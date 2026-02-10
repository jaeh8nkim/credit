# Token-level credit assignment for GRPO training.

# Credit methods:
# - "vanilla": Vanilla DAPO (all tokens weighted equally)
# - "exponential": Exponential increase from first to last reasoning token
# - "attention": Weight based on attention from answer to reasoning tokens
# - "linear": Linear increase from first to last reasoning token
# - "sigmoid": S-curve increase from first to last reasoning token
# - "zero": No credit for reasoning tokens

"""
- Vanilla DAPO, exponential, linear, sigmoid, zero (3 GPUs):
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file accelerate_config.yaml grpo.py

- attention (4 GPUs - GPU 3 reserved for attention extraction):
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file accelerate_config.yaml grpo.py

tmux new -s grpo
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file accelerate_config.yaml grpo.py > grpo.log 2>&1
Press Ctrl+B, then D
tmux attach -t grpo
tmux kill-session -t grpo
"""

import os
from dotenv import load_dotenv
import wandb
import torch
from typing import Optional, Union, Any
from collections import defaultdict
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle

from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward, reasoning_accuracy_reward


# =============================================================================
# NAMING CONFIG
# =============================================================================

OUTPUT_DIR = "Qwen3-0.6B-ATPO"
WANDB_PROJECT = "TRL-Qwen3-0.6B-DAPO"
WANDB_RUN_NAME = "Qwen3-0.6B-ATPO"


# =============================================================================
# SETUP
# =============================================================================

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=True)
os.environ["WANDB_PROJECT"] = WANDB_PROJECT


# =============================================================================
# RESUME CONFIG
# =============================================================================

# RESUME_FROM_CHECKPOINT = "Qwen3-0.6B-DAPO/checkpoint-100"
# WANDB_RUN_ID = "abc123xy"
RESUME_FROM_CHECKPOINT = None
WANDB_RUN_ID = None

if RESUME_FROM_CHECKPOINT and WANDB_RUN_ID:
    os.environ["WANDB_RESUME"] = "must"  # "must" = fail if run doesn't exist, "allow" = resume or create new, "never" = always new
    os.environ["WANDB_RUN_ID"] = WANDB_RUN_ID


# =============================================================================
# ATTENTION EXTRACTION FOR CREDIT ASSIGNMENT (DEDICATED GPU 3)
# =============================================================================

# Global attention extractor instance (singleton pattern)
# This is loaded once on GPU 3 and reused throughout training
_global_attention_extractor = None


class DedicatedAttentionExtractor:
    """
    Dedicated attention extractor that runs on a separate GPU (GPU 3).

    This class loads TWO models on GPU 3:
    1. SDPA model: For efficient prefill of reasoning tokens (builds KV cache)
    2. Eager model: For answer tokens with attention output

    Three-phase pipeline:
    1. vLLM generation (Phase 1) - GPUs 0, 1, 2 (colocate mode)
    2. Attention extraction (Phase 2) - GPU 3 (this class)
    3. TRL training (Phase 3) - GPUs 0, 1, 2

    The models are loaded once and kept in memory throughout training.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda:3",
        dtype: torch.dtype = torch.bfloat16,
        attention_num_layers: int = 8,
        epsilon: float = 0.00001,
        sigmoid_temperature: float = 2.0,
        baseline_credit: float = 0.5,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.attention_num_layers = attention_num_layers
        self.epsilon = epsilon
        self.sigmoid_temperature = sigmoid_temperature
        self.baseline_credit = baseline_credit
        self.model_sdpa = None  # For efficient prefill (no attention output)
        self.model_eager = None  # For attention extraction
        self.tokenizer = None
        self._loaded = False

    def load_model(self):
        """Load both SDPA and eager models on the dedicated GPU."""
        if self._loaded:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[AttentionExtractor] Loading models on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Load SDPA model for efficient prefill (no attention output needed)
        print(f"[AttentionExtractor] Loading SDPA model for prefill...")
        self.model_sdpa = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            attn_implementation="sdpa",  # Flash attention for efficiency
            trust_remote_code=True,
            device_map=None,
        )
        self.model_sdpa = self.model_sdpa.to(self.device)
        self.model_sdpa.eval()

        # Load eager model for attention extraction
        print(f"[AttentionExtractor] Loading eager model for attention extraction...")
        self.model_eager = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            attn_implementation="eager",  # Required for attention output
            trust_remote_code=True,
            device_map=None,
        )
        self.model_eager = self.model_eager.to(self.device)
        self.model_eager.eval()

        self.num_layers = self.model_sdpa.config.num_hidden_layers
        self._loaded = True
        print(f"[AttentionExtractor] Both models loaded on {self.device}. Layers: {self.num_layers}")

    def extract_attention_weights(
        self,
        input_ids: torch.Tensor,  # [seq_len] - single sequence on CPU (completion only)
        think_end_pos: int,  # Position right after </think> within completion
    ) -> torch.Tensor:
        """
        Extract attention-based weights for a single sequence (completion only).

        Uses attention from queries after </think> to keys in reasoning section.
        Applies log transformation and normalizes to [0, 1].

        MEMORY-EFFICIENT TWO-PASS APPROACH:
        The attention matrix we need is [answer_tokens × reasoning_tokens] which is
        typically [500-1500 × up_to_20K] = 30-120MB, much smaller than full [seq × seq].

        However, eager attention computes full [seq × seq] during forward pass.
        To avoid OOM, we use KV-cache:
        1. First pass: Process reasoning tokens to build KV cache (no attention output needed)
        2. Second pass: Process answer tokens with KV cache, get attention to reasoning keys

        This way the attention matrix is only [answer_len × full_seq] which is manageable.

        Args:
            input_ids: Token IDs for the completion [seq_len] (on CPU)
            think_end_pos: Position right after </think> token within completion

        Returns:
            Attention weights [seq_len] on CPU, normalized to [0, 1]
        """
        self.load_model()

        seq_len = input_ids.size(0)

        # Initialize weights to 1.0 (default for all tokens)
        weights = torch.ones(seq_len, dtype=torch.float32)

        if think_end_pos is None or think_end_pos <= 0 or think_end_pos >= seq_len:
            # No valid </think> position, return uniform weights
            return weights

        reasoning_len = think_end_pos
        answer_len = seq_len - think_end_pos

        # Move input to dedicated GPU
        input_ids_device = input_ids.unsqueeze(0).to(self.device)  # [1, seq_len]

        # Split into reasoning and answer portions
        reasoning_ids = input_ids_device[:, :think_end_pos]  # [1, reasoning_len]
        answer_ids = input_ids_device[:, think_end_pos:]  # [1, answer_len]

        with torch.no_grad():
            # Pass 1: Process reasoning tokens with SDPA model to build KV cache
            # SDPA is memory-efficient (no full attention matrix materialization)
            # We don't need attention output here, just the KV cache
            reasoning_outputs = self.model_sdpa(
                reasoning_ids,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = reasoning_outputs.past_key_values

            # Pass 2: Process answer tokens with EAGER model to get attention output
            # The attention matrix will be [answer_len × (reasoning_len + answer_len)]
            # which is much smaller than [seq × seq] for long reasoning
            # Note: We pass the KV cache from SDPA model to eager model - they share the same architecture
            answer_outputs = self.model_eager(
                answer_ids,
                past_key_values=past_key_values,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )

        # Average attention across last N layers, then average across heads
        # Each layer shape: [batch=1, num_heads, answer_len, reasoning_len + answer_len]
        n_layers = min(self.attention_num_layers, len(answer_outputs.attentions))
        layers = answer_outputs.attentions[-n_layers:]
        last_layers_attn = layers[0].mean(dim=1)
        for attn in layers[1:]:
            last_layers_attn = last_layers_attn + attn.mean(dim=1)
        last_layers_attn = (last_layers_attn / n_layers).squeeze(0)  # [answer_len, kv_len]

        # We want attention from answer queries to reasoning keys
        # The first reasoning_len columns are attention to reasoning tokens
        # Shape: [answer_len, reasoning_len]
        attn_to_reasoning = last_layers_attn[:, :reasoning_len]

        # Average across answer queries for each reasoning token
        # Shape: [reasoning_len]
        avg_attention = attn_to_reasoning.mean(dim=0)

        # =============================================================
        # ADAPTIVE SIGMOID TRANSFORMATION
        # =============================================================
        # This approach solves outlier compression by:
        # 1. Log transform: linearizes exponential attention distribution
        # 2. Z-score: centers data (mean=0) and normalizes spread (std=1)
        # 3. Sigmoid: maps to (0,1) with temperature-controlled contrast
        # 4. Min-max rescale: forces exact [0.0, 1.0] range

        # Step 1: Log transform
        log_attn = torch.log(avg_attention + self.epsilon)

        # Step 2: Z-score normalization (adaptive to this sequence)
        seq_mean = log_attn.mean()
        seq_std = log_attn.std()

        # Handle edge case: uniform attention (std=0)
        if seq_std == 0:
            # All tokens have equal attention, give them equal credit
            attn_transformed = torch.full_like(log_attn, 0.5)
        else:
            z_score = (log_attn - seq_mean) / seq_std

            # Step 3: Sigmoid with temperature
            # Higher temperature = sharper separation (more binary)
            # Lower temperature = softer gradients
            attn_transformed = torch.sigmoid(z_score * self.sigmoid_temperature)

        # Step 4: Min-max rescale to force exact [0.0, 1.0] range
        # This ensures lowest-contribution token gets exactly 0 credit
        min_w = attn_transformed.min()
        max_w = attn_transformed.max()
        if max_w > min_w:
            attn_transformed = (attn_transformed - min_w) / (max_w - min_w)
        else:
            # Edge case: all values are the same after sigmoid
            attn_transformed = torch.full_like(attn_transformed, 0.5)

        # Step 5: Apply baseline credit (maps [0,1] to [baseline, 1])
        attn_transformed = attn_transformed * (1.0 - self.baseline_credit) + self.baseline_credit
        # =============================================================

        # Assign weights to reasoning tokens
        weights[:think_end_pos] = attn_transformed.cpu()
        # Tokens after </think> keep weight 1.0 (already initialized)

        # Clean up intermediate tensors
        del reasoning_outputs, answer_outputs, past_key_values, last_layers_attn, attn_to_reasoning, avg_attention, attn_transformed
        torch.cuda.empty_cache()

        return weights


def get_attention_extractor(
    model_name: str,
    attention_num_layers: int = 8,
    epsilon: float = 0.00001,
    sigmoid_temperature: float = 2.0,
    baseline_credit: float = 0.5,
) -> DedicatedAttentionExtractor:
    """
    Get or create the global attention extractor instance.

    This ensures only one model is loaded on GPU 3 throughout training.
    """
    global _global_attention_extractor

    if _global_attention_extractor is None:
        _global_attention_extractor = DedicatedAttentionExtractor(
            model_name=model_name,
            device="cuda:3",
            dtype=torch.bfloat16,
            attention_num_layers=attention_num_layers,
            epsilon=epsilon,
            sigmoid_temperature=sigmoid_temperature,
            baseline_credit=baseline_credit,
        )

    return _global_attention_extractor


# =============================================================================
# VISUALIZATION FUNCTIONS FOR TOKEN-LEVEL CREDIT
# =============================================================================

# Font and style configuration
FONT_SFPRO = "./_inventory/fonts/SFPRODISPLAYREGULAR.OTF"
FONT_SFMONO = "./_inventory/fonts/SFMonoRegular.otf"
FONT_TIMESNEWROMAN = "./_inventory/fonts/Times-New-Roman.otf"
CREDIT_PLOTS_SUBDIR = "credit_plots"  # Subdirectory within OUTPUT_DIR for credit plots
matplotlib.use('Agg')  # Non-interactive backend for server environments
plt.ioff()


def plot_credit_token_highlighting(
    tokens,
    token_weights,
    tokenizer,
    think_start_pos,
    think_end_pos,
    fig_width_in=8.0,
):
    """
    Highlight tokens with credit scores (reasoning section only).

    Args:
        tokens: List of token strings
        token_weights: Tensor of credit weights [seq_len]
        tokenizer: Tokenizer object
        think_start_pos: Position of <think> token
        think_end_pos: Position right after </think> token
        fig_width_in: Figure width in inches

    Returns:
        matplotlib.figure.Figure
    """
    # Style configuration
    bg_color = "#f5f5f7"
    text_color = "black"
    highlight_hex = "#ff3b2f"
    font_size_pt = 10
    line_spacing = 1.4

    # Margins (uniform on all sides)
    MARGIN_PT = 5
    left_margin_pt = MARGIN_PT
    right_margin_pt = MARGIN_PT
    top_margin_pt = MARGIN_PT
    bottom_margin_pt = MARGIN_PT

    pad_x = 0
    pad_y = font_size_pt * 0.10

    # Font setup
    if os.path.exists(FONT_SFPRO):
        fm.fontManager.addfont(FONT_SFPRO)
        fontname = fm.FontProperties(fname=FONT_SFPRO).get_name()
        prop = fm.FontProperties(fname=FONT_SFPRO, size=font_size_pt)
        matplotlib.rcParams['font.family'] = fontname
    else:
        prop = fm.FontProperties(size=font_size_pt)
        matplotlib.rcParams['font.family'] = 'sans-serif'

    # Convert to numpy
    if isinstance(token_weights, torch.Tensor):
        weights_np = token_weights.cpu().float().numpy()
    else:
        weights_np = np.array(token_weights)

    # Extract reasoning section
    if think_start_pos is None or think_end_pos is None or think_start_pos >= think_end_pos:
        # No valid reasoning section
        chunk_ranges = []
        attention_weights = []
    else:
        chunk_start = think_start_pos + 1
        chunk_end = think_end_pos - 1

        if chunk_start > chunk_end:
            chunk_ranges = []
            attention_weights = []
        else:
            chunk_ranges = [(chunk_start, chunk_end)]
            attention_weights = weights_np.tolist()

    # Build paragraphs
    def build_chunks(tokens, chunk_ranges, attention_weights):
        paras = []
        for start, end in chunk_ranges:
            chunk_tokens = [str(tokens[i]) for i in range(start, end + 1)]
            chunk_token_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in chunk_tokens]
            full_text = tokenizer.decode(chunk_token_ids, skip_special_tokens=False)

            items = []
            char_pos = 0

            for idx, tok_id in enumerate(chunk_token_ids):
                decoded_so_far = tokenizer.decode(chunk_token_ids[:idx+1], skip_special_tokens=False)
                next_char_pos = len(decoded_so_far)
                token_text = full_text[char_pos:next_char_pos]
                weight = attention_weights[start + idx]

                parts = re.split(r"(\n)", token_text)
                for p in parts:
                    if not p:
                        continue
                    if p == "\n":
                        items.append(("\\n", weight, True))
                    else:
                        items.append((p, weight, False))
                char_pos = next_char_pos
            paras.append(items)
        return paras

    paragraphs = build_chunks(tokens, chunk_ranges, attention_weights)

    # Helper functions for rendering
    line_step = font_size_pt * line_spacing

    def sanitize(t: str) -> str:
        return t.replace("$", r"\$")

    # First pass: measure actual content height using a temporary figure
    temp_fig_height_in = 100.0  # Large enough for any content
    temp_fig, temp_ax = plt.subplots(figsize=(fig_width_in, temp_fig_height_in))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    temp_ax.set_position([0, 0, 1, 1])
    temp_ax.axis('off')
    temp_fig.canvas.draw()
    temp_renderer = temp_fig.canvas.get_renderer()
    PX2PT = 72.0 / temp_fig.dpi

    def measure_text_pts(t: str, renderer):
        w_px, h_px, d_px = renderer.get_text_width_height_descent(t, prop, ismath=False)
        return (w_px * PX2PT, (h_px - d_px) * PX2PT, d_px * PX2PT)

    temp_axbb = temp_ax.get_window_extent(renderer=temp_renderer)
    W_pts = temp_axbb.width * PX2PT
    max_line_w = W_pts - left_margin_pt - right_margin_pt

    # Measure content height by simulating layout
    # Get text metrics for calculating actual text height
    _, temp_asc, temp_des = measure_text_pts(sanitize("Mg"), temp_renderer)

    y_start = 0.0
    y_current = y_start
    x = 0.0

    for items in paragraphs:
        for raw_text, weight, _is_nl in items:
            t_disp = sanitize(raw_text)
            w_pt, _a, _d = measure_text_pts(t_disp, temp_renderer)

            if (x + w_pt) > max_line_w and x > 0:
                y_current -= line_step
                x = 0.0

            x += w_pt

    plt.close(temp_fig)

    # Calculate actual content height
    # y_current tracks baseline positions, descending by line_step for each line
    # Content spans from first baseline (y_start) to last baseline (y_current)
    # Above first baseline: ascender height (temp_asc)
    # Below last baseline: descender height (temp_des)
    content_height_pt = abs(y_current - y_start) + temp_asc + temp_des

    # Create figure with dynamic height: equal margins on top and bottom
    fig_height_pt = content_height_pt + top_margin_pt + bottom_margin_pt
    fig_height_in = fig_height_pt / 72.0

    # Create final figure
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_position([0, 0, 1, 1])

    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.axis('off')

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    axbb = ax.get_window_extent(renderer=renderer)
    W_pts = axbb.width * PX2PT
    H_pts = axbb.height * PX2PT
    ax.set_xlim(0, W_pts)
    ax.set_ylim(0, H_pts)

    _, FIX_ASC, FIX_DES = measure_text_pts(sanitize("Mg"), renderer)
    FIX_HEIGHT = FIX_ASC + FIX_DES

    x0 = left_margin_pt
    # Position first baseline so text ascender touches the top margin
    y = H_pts - top_margin_pt - FIX_ASC
    max_line_w = W_pts - left_margin_pt - right_margin_pt

    def new_line():
        nonlocal y
        y -= line_step

    def render_items(items):
        nonlocal y
        x = x0
        for raw_text, weight, _is_nl in items:
            t_disp = sanitize(raw_text)
            w_pt, _a, _d = measure_text_pts(t_disp, renderer)

            if (x - x0 + w_pt) > max_line_w and (x > x0):
                new_line()
                x = x0

            if weight > 0.0:
                rect_x = x
                rect_y = y - FIX_DES - pad_y
                rect_w = w_pt
                rect_h = FIX_HEIGHT + 2 * pad_y
                alpha = weight * 0.9

                ax.add_patch(Rectangle((rect_x, rect_y), rect_w, rect_h,
                                       facecolor=highlight_hex, alpha=alpha,
                                       edgecolor='none'))

            ax.text(x, y, t_disp, color=text_color, fontproperties=prop,
                    va='baseline', ha='left')
            x += w_pt

    for items in paragraphs:
        render_items(items)

    return fig


def plot_credit_scores(
    tokens,
    token_weights,
    think_start_pos,
    think_end_pos,
):
    """
    Plot token-level credit scores as a line graph.

    Args:
        tokens: List of token strings
        token_weights: Tensor of credit weights [seq_len]
        think_start_pos: Position of <think> token
        think_end_pos: Position right after </think> token

    Returns:
        matplotlib.figure.Figure
    """
    # Dimensions
    FIG_WIDTH = 8
    FIG_HEIGHT = 5

    # Line Widths
    AXIS_LINE_WIDTH = 1
    TICK_WIDTH = 1
    MAIN_LINE_WIDTH = 0.75

    # Colors
    BG_COLOR = 'white'
    AXIS_COLOR = 'black'
    TEXT_COLOR = 'black'

    # Font Sizes
    FONT_SIZE_TITLE = 16
    FONT_SIZE_LABEL = 14
    FONT_SIZE_TICK = 10

    # Prepare Font Properties
    prop_title = None
    prop_label = None
    prop_mono = None

    if os.path.exists(FONT_TIMESNEWROMAN):
        prop_title = fm.FontProperties(fname=FONT_TIMESNEWROMAN, size=FONT_SIZE_TITLE)
        prop_label = fm.FontProperties(fname=FONT_TIMESNEWROMAN, size=FONT_SIZE_LABEL)
    else:
        prop_title = fm.FontProperties(family='serif', size=FONT_SIZE_TITLE)
        prop_label = fm.FontProperties(family='serif', size=FONT_SIZE_LABEL)

    if os.path.exists(FONT_SFMONO):
        prop_mono = fm.FontProperties(fname=FONT_SFMONO, size=FONT_SIZE_TICK)
    else:
        prop_mono = fm.FontProperties(family='monospace', size=FONT_SIZE_TICK)

    # Convert to numpy
    if isinstance(token_weights, torch.Tensor):
        credit_scores = token_weights.cpu().float().numpy()
    else:
        credit_scores = np.array(token_weights)

    seq_len = len(tokens)

    # Setup Figure
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Configure Spines
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(AXIS_COLOR)
        ax.spines[spine].set_linewidth(AXIS_LINE_WIDTH)
        ax.spines[spine].set_zorder(10)

    # Handle Axis Limits
    ax.set_xlim(0, seq_len)
    ax.set_ylim(0, 1.05)

    # Plot Data
    token_positions = np.arange(seq_len)
    ax.plot(token_positions, credit_scores, color='black', linewidth=MAIN_LINE_WIDTH, zorder=2)

    # Add markers for <think> and </think> positions
    # To get visually equal width/height, account for data range and figure aspect ratio
    # Data ranges: x=[0, seq_len], y=[0, 1.05]. Figure: 8x5 inches.
    # Pixels per data unit: x_scale = 8/seq_len, y_scale = 5/1.05
    # To make triangle look square: triangle_size_y = triangle_size_x * (x_scale / y_scale)
    triangle_size_x = max(seq_len * 0.02, 1)
    triangle_size_y = triangle_size_x * (1.05 / seq_len) * (FIG_WIDTH / FIG_HEIGHT)

    for pos in [think_start_pos, think_end_pos]:
        if pos is not None and pos < seq_len:
            y = credit_scores[pos]
            # Downward triangle with tip at (pos, y)
            x_coords = [pos - triangle_size_x/2, pos, pos + triangle_size_x/2, pos - triangle_size_x/2]
            y_coords = [y + triangle_size_y, y, y + triangle_size_y, y + triangle_size_y]
            ax.fill(x_coords, y_coords, 'r', alpha=1.0, zorder=10, edgecolor='none')

    # Typography & Labels
    ax.set_title('Token-Level Credit Assignment', fontproperties=prop_title, color=TEXT_COLOR, pad=10)
    ax.set_xlabel('Token Position', fontproperties=prop_label, color=TEXT_COLOR, labelpad=5)
    ax.set_ylabel('Credit Score', fontproperties=prop_label, color=TEXT_COLOR, labelpad=10)

    ax.tick_params(axis='both', colors=AXIS_COLOR, direction='out', width=TICK_WIDTH, length=4)
    if prop_mono:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(prop_mono)

    plt.tight_layout()

    return fig


# =============================================================================
# DEBUG REWARD WRAPPER
# =============================================================================

_debug_completions = []
_debug_step = 0
_max_debug_steps = 5


def wrap_reward(reward_func, max_debug_steps=5):
    """Wrap a reward function to capture completions for debugging."""
    global _max_debug_steps
    _max_debug_steps = max_debug_steps

    def wrapper(completions, solution, **kwargs):
        global _debug_completions, _debug_step

        if _debug_step <= _max_debug_steps and len(completions) > 0:
            first = completions[0]
            if isinstance(first, list) and len(first) > 0:
                if isinstance(first[0], dict):
                    _debug_completions = [first[0].get("content", str(first[0]))]
                else:
                    _debug_completions = [str(first[0])]
            else:
                _debug_completions = [str(first)]

        return reward_func(completions, solution, **kwargs)

    return wrapper


def add_debug_callback(trainer, train_dataset):
    """Attach debug callback to trainer."""
    from transformers import TrainerCallback

    class DebugPrintCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            global _debug_step
            _debug_step = state.global_step + 1

        def on_step_end(self, args, state, control, **kwargs):
            global _debug_completions

            if state.global_step > _max_debug_steps:
                return

            if not trainer.accelerator.is_main_process:
                return

            tokenizer = trainer.processing_class
            sample_idx = (state.global_step - 1) % len(train_dataset)
            sample = train_dataset[sample_idx]

            tokenized = tokenizer.apply_chat_template(
                sample["prompt"], tokenize=True, add_generation_prompt=True,
            )
            decoded = tokenizer.decode(tokenized, skip_special_tokens=False)

            print(f"\n{'='*80}")
            print(f"FULL CONTEXT DEBUG - STEP {state.global_step} (First Sample)")
            print(f"{'='*80}")

            print(f"\n[RAW PROMPT FROM DATASET]:")
            print(sample["prompt"])

            print(f"\n[FULL PROMPT WITH SPECIAL TOKENS]:")
            print(decoded)

            if _debug_completions:
                print(f"\n[FIRST GENERATED COMPLETION]:")
                print(_debug_completions[0])

            print(f"\n[GROUND TRUTH SOLUTION]: {sample.get('solution', 'N/A')}")
            print(f"{'='*80}\n")

            _debug_completions = []

    trainer.add_callback(DebugPrintCallback())


# =============================================================================
# TOKEN-LEVEL CREDIT ASSIGNMENT TRAINER
# =============================================================================

class TokenLevelGRPOTrainer(GRPOTrainer):
    """
    Custom GRPO trainer with token-level credit assignment.

    Supports credit assignment methods:
    - "exponential": Exponential increase from first to last reasoning token
    - "attention": Weight based on attention from answer to reasoning tokens
    - "linear": Linear increase from first to last reasoning token
    - "sigmoid": S-curve increase from first to last reasoning token
    - "zero": No credit for reasoning tokens

    Three-phase pipeline for attention method:
    1. vLLM generation (handled by parent class)
    2. Attention extraction (Phase 2)
    3. TRL training (Phase 3)

    Reasoning tokens (before </think>) get method-specific weights.
    Answer tokens (after </think>) get normal weights (1.0).

    This implementation modifies the inputs to include per-token weighted advantages.
    """

    def __init__(
        self,
        *args,
        think_end_token: str = "</think>",
        decay_base: float = 0.995,  # Exponential decay base
        credit_method: str = "exponential",  # "exponential", "attention", "linear", "sigmoid", "zero"
        attention_num_layers: int = 8,  # Number of last layers to average attention from
        attention_epsilon: float = 0.00001,  # Epsilon for attention log transform
        sigmoid_temperature: float = 2.0,  # Temperature for sigmoid contrast (higher = more binary)
        baseline_credit: float = 0.5,  # Baseline credit (shifts credit floor)
        debug_token_weights: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.think_end_token = think_end_token
        self.decay_base = decay_base
        self.credit_method = credit_method
        self.attention_num_layers = attention_num_layers
        self.attention_epsilon = attention_epsilon
        self.sigmoid_temperature = sigmoid_temperature
        self.baseline_credit = baseline_credit
        self.debug_token_weights = debug_token_weights
        self._last_debug_step = -1  # Track which step we've already printed for
        self._fallback_debug_data = None  # Store fallback data (zero reward sample)
        self._minibatch_count = 0  # Track minibatch count within current step
        self._minibatch_step = -1  # Which step we're counting minibatches for

        # Cache the </think> token IDs after tokenizer is available
        self._think_end_ids = None

        # Cache for attention weights (stored on CPU to save GPU memory)
        # Key: tuple of completion token ids, Value: attention weights tensor on CPU
        # This cache is used in attention mode to avoid recomputing
        # attention weights for the same completion
        self._attention_weights_cache = {}

        # Validate credit method
        if self.credit_method not in ["vanilla", "exponential", "attention", "linear", "sigmoid", "zero"]:
            raise ValueError(f"Invalid credit_method: {credit_method}. Must be 'vanilla', 'exponential', 'attention', 'linear', 'sigmoid', or 'zero'")

        # Setup credit plots directory for visualization
        self._credit_plots_dir = None
        if self.credit_method != "vanilla" and self.debug_token_weights:
            self._setup_credit_plots_dir()

    def _clear_attention_cache(self):
        """Clear the attention weights cache to free CPU memory."""
        if self._attention_weights_cache:
            self._attention_weights_cache.clear()

    def _setup_credit_plots_dir(self):
        """Setup directory for credit visualization plots."""
        import glob

        # Only rank 0 should create directories
        if not self.accelerator.is_main_process:
            return

        # Create base directory within OUTPUT_DIR
        credit_plots_base = os.path.join(self.args.output_dir, CREDIT_PLOTS_SUBDIR)
        os.makedirs(credit_plots_base, exist_ok=True)

        # Find next run number
        existing_runs = glob.glob(os.path.join(credit_plots_base, "credit_plots_run_*"))
        if existing_runs:
            run_numbers = []
            for run_dir in existing_runs:
                try:
                    num = int(run_dir.split('_')[-1])
                    run_numbers.append(num)
                except ValueError:
                    continue
            next_run = max(run_numbers) + 1 if run_numbers else 1
        else:
            next_run = 1

        # Create run-specific directory
        self._credit_plots_dir = os.path.join(credit_plots_base, f"credit_plots_run_{next_run}")
        os.makedirs(self._credit_plots_dir, exist_ok=True)
        print(f"[Credit Plots] Output directory: {self._credit_plots_dir}")

    def _get_think_end_ids(self):
        """Get and cache the token IDs for </think>"""
        if self._think_end_ids is None:
            self._think_end_ids = self.processing_class.encode(
                self.think_end_token, 
                add_special_tokens=False
            )
            self._think_end_ids = torch.tensor(self._think_end_ids)
        return self._think_end_ids
        
    def _find_think_end_position(
        self,
        completion_ids: torch.Tensor,  # [seq_len]
    ) -> Optional[int]:
        """
        Find the token position where </think> ends.
        Returns the index right after </think>, or None if not found.
        """
        think_end_ids = self._get_think_end_ids().to(completion_ids.device)

        seq_len = completion_ids.size(0)
        pattern_len = think_end_ids.size(0)

        # Search for the pattern in completion_ids
        for i in range(seq_len - pattern_len + 1):
            if torch.equal(completion_ids[i:i + pattern_len], think_end_ids):
                # Return the position right after </think>
                return i + pattern_len

        return None

    def _find_think_start_position(
        self,
        completion_ids: torch.Tensor,  # [seq_len]
    ) -> Optional[int]:
        """
        Find the token position where <think> starts.
        Returns the index of <think>, or None if not found.
        """
        # Encode <think> token
        think_start_token = "<think>"
        think_start_ids = self.processing_class.encode(think_start_token, add_special_tokens=False)
        think_start_ids = torch.tensor(think_start_ids).to(completion_ids.device)

        seq_len = completion_ids.size(0)
        pattern_len = think_start_ids.size(0)

        # Search for the pattern in completion_ids
        for i in range(seq_len - pattern_len + 1):
            if torch.equal(completion_ids[i:i + pattern_len], think_start_ids):
                return i

        return None

    def _get_model_name(self) -> str:
        """Get the model name for the attention extractor."""
        if hasattr(self.model, 'name_or_path'):
            return self.model.name_or_path
        elif hasattr(self.model, 'config') and hasattr(self.model.config, '_name_or_path'):
            return self.model.config._name_or_path
        elif hasattr(self.model, 'module'):
            # Handle wrapped models
            if hasattr(self.model.module, 'name_or_path'):
                return self.model.module.name_or_path
            elif hasattr(self.model.module, 'config') and hasattr(self.model.module.config, '_name_or_path'):
                return self.model.module.config._name_or_path
        # Fallback to a default - this should be updated if using a different model
        return "Qwen/Qwen3-0.6B"

    def _extract_attention_weights_for_batch(
        self,
        prompt_ids: torch.Tensor,  # [batch_size, prompt_len]
        completion_ids: torch.Tensor,  # [batch_size, completion_len]
        completion_mask: torch.Tensor,  # [batch_size, completion_len]
    ) -> torch.Tensor:
        """
        Extract attention-based weights for a batch of completions.

        This is Phase 2 of the three-phase pipeline:
        1. vLLM generation (Phase 1) - GPUs 0, 1, 2 (colocate mode)
        2. Attention extraction (Phase 2) - GPU 3 (dedicated eager model, rank 0 only)
        3. TRL training (Phase 3) - GPUs 0, 1, 2

        DISTRIBUTED COORDINATION:
        - Only rank 0 loads the attention model on GPU 3
        - All ranks gather their data to rank 0
        - Rank 0 processes all sequences and broadcasts results back
        - This avoids multiple processes loading the model on the same GPU

        Args:
            prompt_ids: Prompt token IDs [batch_size, prompt_len]
            completion_ids: Completion token IDs [batch_size, completion_len]
            completion_mask: Completion attention mask [batch_size, completion_len]

        Returns:
            Attention weights [batch_size, completion_len] on the same device as completion_ids
        """
        import torch.distributed as dist

        batch_size, seq_len = completion_ids.shape
        device = completion_ids.device

        # Get distributed info
        is_distributed = dist.is_initialized()
        world_size = dist.get_world_size() if is_distributed else 1
        rank = dist.get_rank() if is_distributed else 0

        if not is_distributed or world_size == 1:
            # Single process mode - extract directly
            return self._extract_attention_weights_single_process(
                prompt_ids, completion_ids, completion_mask
            )

        # =====================================================
        # DISTRIBUTED MODE: Gather to rank 0, process, scatter
        # =====================================================

        # Step 1: All ranks send their shapes to rank 0 so we know tensor sizes
        local_batch_size = torch.tensor([batch_size], device=device, dtype=torch.long)
        local_prompt_len = torch.tensor([prompt_ids.size(1)], device=device, dtype=torch.long)
        local_seq_len = torch.tensor([seq_len], device=device, dtype=torch.long)

        # Gather shapes from all ranks
        if rank == 0:
            all_batch_sizes = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(world_size)]
            all_prompt_lens = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(world_size)]
            all_seq_lens = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(world_size)]
        else:
            all_batch_sizes = None
            all_prompt_lens = None
            all_seq_lens = None

        dist.gather(local_batch_size, gather_list=all_batch_sizes, dst=0)
        dist.gather(local_prompt_len, gather_list=all_prompt_lens, dst=0)
        dist.gather(local_seq_len, gather_list=all_seq_lens, dst=0)

        # Step 2: Gather actual tensors to rank 0
        # Use padded gather since tensors might have different shapes
        # First, broadcast max sizes
        max_batch = torch.tensor([batch_size], device=device, dtype=torch.long)
        max_prompt = torch.tensor([prompt_ids.size(1)], device=device, dtype=torch.long)
        max_seq = torch.tensor([seq_len], device=device, dtype=torch.long)

        dist.all_reduce(max_batch, op=dist.ReduceOp.MAX)
        dist.all_reduce(max_prompt, op=dist.ReduceOp.MAX)
        dist.all_reduce(max_seq, op=dist.ReduceOp.MAX)

        max_batch_size = max_batch.item()
        max_prompt_len = max_prompt.item()
        max_seq_len = max_seq.item()

        # Pad local tensors to max size
        padded_prompt_ids = torch.zeros(max_batch_size, max_prompt_len, device=device, dtype=prompt_ids.dtype)
        padded_completion_ids = torch.zeros(max_batch_size, max_seq_len, device=device, dtype=completion_ids.dtype)
        padded_completion_mask = torch.zeros(max_batch_size, max_seq_len, device=device, dtype=completion_mask.dtype)

        padded_prompt_ids[:batch_size, :prompt_ids.size(1)] = prompt_ids
        padded_completion_ids[:batch_size, :seq_len] = completion_ids
        padded_completion_mask[:batch_size, :seq_len] = completion_mask

        # Gather all tensors to rank 0
        if rank == 0:
            gathered_prompt_ids = [torch.zeros_like(padded_prompt_ids) for _ in range(world_size)]
            gathered_completion_ids = [torch.zeros_like(padded_completion_ids) for _ in range(world_size)]
            gathered_completion_mask = [torch.zeros_like(padded_completion_mask) for _ in range(world_size)]
        else:
            gathered_prompt_ids = None
            gathered_completion_ids = None
            gathered_completion_mask = None

        dist.gather(padded_prompt_ids, gather_list=gathered_prompt_ids, dst=0)
        dist.gather(padded_completion_ids, gather_list=gathered_completion_ids, dst=0)
        dist.gather(padded_completion_mask, gather_list=gathered_completion_mask, dst=0)

        # Step 3: Rank 0 extracts attention weights for all sequences
        if rank == 0:
            all_weights = []

            for r in range(world_size):
                r_batch_size = all_batch_sizes[r].item()
                r_prompt_len = all_prompt_lens[r].item()
                r_seq_len = all_seq_lens[r].item()

                if r_batch_size == 0:
                    all_weights.append(torch.ones(max_batch_size, max_seq_len, device=device, dtype=torch.float32))
                    continue

                # Extract actual tensors for this rank
                r_prompt_ids = gathered_prompt_ids[r][:r_batch_size, :r_prompt_len]
                r_completion_ids = gathered_completion_ids[r][:r_batch_size, :r_seq_len]
                r_completion_mask = gathered_completion_mask[r][:r_batch_size, :r_seq_len]

                # Extract weights using the dedicated model on GPU 3
                r_weights = self._extract_attention_weights_single_process(
                    r_prompt_ids, r_completion_ids, r_completion_mask
                )

                # Pad weights back to max size for scatter
                padded_weights = torch.ones(max_batch_size, max_seq_len, device=device, dtype=torch.float32)
                padded_weights[:r_batch_size, :r_seq_len] = r_weights
                all_weights.append(padded_weights)
        else:
            all_weights = None

        # Step 4: Scatter weights back to each rank
        local_weights = torch.ones(max_batch_size, max_seq_len, device=device, dtype=torch.float32)
        dist.scatter(local_weights, scatter_list=all_weights, src=0)

        # Extract only the portion we need
        weights = local_weights[:batch_size, :seq_len]

        return weights

    def _extract_attention_weights_single_process(
        self,
        prompt_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract attention weights for a single process (no distributed coordination).
        This is called by rank 0 to do the actual extraction work.

        MEMORY OPTIMIZATION:
        - Only processes completion tokens (not prompt+completion)
        - The attention we need is between answer tokens (after </think>) and
          reasoning tokens (before </think>), both within the completion
        - This drastically reduces memory usage for long prompts
        """
        batch_size, seq_len = completion_ids.shape
        device = completion_ids.device

        # Initialize weights to 1.0
        weights = torch.ones(batch_size, seq_len, device=device, dtype=torch.float32)

        # Get the dedicated attention extractor on GPU 3
        model_name = self._get_model_name()
        extractor = get_attention_extractor(
            model_name,
            attention_num_layers=self.attention_num_layers,
            epsilon=self.attention_epsilon,
            sigmoid_temperature=self.sigmoid_temperature,
            baseline_credit=self.baseline_credit,
        )

        for i in range(batch_size):
            # Get actual sequence length
            mask = completion_mask[i]
            actual_len = mask.sum().int().item()

            if actual_len == 0:
                continue

            # Create cache key from completion token IDs
            completion_tuple = tuple(completion_ids[i, :actual_len].cpu().tolist())

            # Check cache first
            if completion_tuple in self._attention_weights_cache:
                cached_weights = self._attention_weights_cache[completion_tuple]
                weights[i, :len(cached_weights)] = cached_weights.to(device)
                continue

            # Find </think> position in completion
            think_end_pos = self._find_think_end_position(completion_ids[i, :actual_len])

            if think_end_pos is None or think_end_pos <= 0:
                # No reasoning section found, all tokens get weight 1.0
                continue

            # MEMORY OPTIMIZATION: Only process completion tokens
            # We don't need the prompt - we're measuring attention from answer tokens
            # (after </think>) to reasoning tokens (before </think>), all within completion
            completion_only = completion_ids[i, :actual_len].cpu()

            # Extract attention weights using the dedicated GPU 3 model
            # think_end_pos is already relative to completion start
            completion_weights = extractor.extract_attention_weights(
                input_ids=completion_only,
                think_end_pos=think_end_pos,
            )

            # Cache on CPU
            self._attention_weights_cache[completion_tuple] = completion_weights

            # Assign to batch weights
            weights[i, :actual_len] = completion_weights.to(device)

        return weights

    def _compute_token_weights(
        self,
        completion_ids: torch.Tensor,  # [batch_size, seq_len]
        completion_mask: torch.Tensor,  # [batch_size, seq_len]
        prompt_ids: Optional[torch.Tensor] = None,  # [batch_size, prompt_len] - needed for attention
    ) -> torch.Tensor:
        """
        Compute per-token weights based on credit_method.

        For "vanilla" (vanilla DAPO):
        - All tokens get weight = 1.0

        For "exponential":
        - Tokens before </think>: exponential increase (early tokens get less weight)
        - Tokens after </think>: weight = 1.0 (normal)

        For "attention":
        - Tokens before </think>: weighted by attention from answer tokens
        - Tokens after </think>: weight = 1.0 (normal)

        For "linear":
        - Tokens before </think>: linear increase from 0 to 1
        - Tokens after </think>: weight = 1.0 (normal)

        For "sigmoid":
        - Tokens before </think>: S-curve increase from ~0 to ~1
        - Tokens after </think>: weight = 1.0 (normal)

        For "zero":
        - Tokens before </think>: weight = 0.0 (no credit)
        - Tokens after </think>: weight = 1.0 (normal)

        Returns: [batch_size, seq_len] tensor of weights
        """
        batch_size, seq_len = completion_ids.shape
        device = completion_ids.device

        # Vanilla DAPO - no token-level credit assignment
        if self.credit_method == "vanilla":
            return torch.ones(batch_size, seq_len, device=device, dtype=torch.float32)

        if self.credit_method == "attention":
            if prompt_ids is None:
                raise ValueError("prompt_ids required for attention credit method")
            return self._extract_attention_weights_for_batch(
                prompt_ids, completion_ids, completion_mask
            )

        # Initialize weights to 1.0 (normal)
        weights = torch.ones(batch_size, seq_len, device=device, dtype=torch.float32)

        for i in range(batch_size):
            # Get actual sequence length (without padding)
            mask = completion_mask[i]
            actual_len = mask.sum().int().item()

            if actual_len == 0:
                continue

            # Find </think> position
            think_end_pos = self._find_think_end_position(completion_ids[i, :actual_len])

            if think_end_pos is not None and think_end_pos > 0:
                reasoning_len = think_end_pos

                if self.credit_method == "exponential":
                    # Exponential increase: weight[t] = decay_base^(reasoning_len - 1 - t)
                    # First token gets decay_base^(reasoning_len-1), last gets decay_base^0 = 1.0
                    positions = torch.arange(reasoning_len, device=device, dtype=torch.float32)
                    decay_weights = self.decay_base ** (reasoning_len - 1 - positions)
                    # Apply baseline credit (maps [0,1] to [baseline, 1])
                    decay_weights = decay_weights * (1.0 - self.baseline_credit) + self.baseline_credit
                    weights[i, :reasoning_len] = decay_weights

                elif self.credit_method == "linear":
                    # Linear increase from 0 (first) to 1 (last reasoning token)
                    if reasoning_len == 1:
                        linear_weights = torch.tensor([1.0], device=device)
                    else:
                        positions = torch.arange(reasoning_len, device=device, dtype=torch.float32)
                        linear_weights = positions / (reasoning_len - 1)
                    # Apply baseline credit (maps [0,1] to [baseline, 1])
                    linear_weights = linear_weights * (1.0 - self.baseline_credit) + self.baseline_credit
                    weights[i, :reasoning_len] = linear_weights

                elif self.credit_method == "sigmoid":
                    # S-curve increase controlled by sigmoid_temperature
                    if reasoning_len == 1:
                        sigmoid_weights = torch.tensor([0.5], device=device)
                    else:
                        positions = torch.arange(reasoning_len, device=device, dtype=torch.float32)
                        # Normalize to [-1, 1]
                        normalized = 2.0 * positions / (reasoning_len - 1) - 1.0
                        # Apply sigmoid with temperature scaling
                        sigmoid_weights = torch.sigmoid(normalized * self.sigmoid_temperature * 3.0)
                        # Rescale to exact [0, 1]
                        min_w = sigmoid_weights.min()
                        max_w = sigmoid_weights.max()
                        if max_w > min_w:
                            sigmoid_weights = (sigmoid_weights - min_w) / (max_w - min_w)
                    # Apply baseline credit (maps [0,1] to [baseline, 1])
                    sigmoid_weights = sigmoid_weights * (1.0 - self.baseline_credit) + self.baseline_credit
                    weights[i, :reasoning_len] = sigmoid_weights

                elif self.credit_method == "zero":
                    # No credit for reasoning tokens
                    zero_weights = torch.zeros(reasoning_len, device=device, dtype=torch.float32)
                    # Apply baseline credit (maps [0,1] to [baseline, 1])
                    zero_weights = zero_weights * (1.0 - self.baseline_credit) + self.baseline_credit
                    weights[i, :reasoning_len] = zero_weights

                # Tokens after </think> keep weight = 1.0 (already initialized)

        return weights
    
    def _print_token_weights_debug(
        self,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        weights: torch.Tensor,
        advantages: torch.Tensor,
    ):
        """Print detailed debug info about token-level advantages."""
        
        if not self.accelerator.is_main_process:
            return
        
        # Get current step number
        current_step = getattr(self.state, 'global_step', 0) + 1
        
        # Reset minibatch count on new step
        if current_step != self._minibatch_step:
            self._minibatch_count = 0
            self._minibatch_step = current_step
            self._fallback_debug_data = None
        
        # Increment minibatch count
        self._minibatch_count += 1
        
        # If already printed for current step, return
        if current_step == self._last_debug_step:
            return
        
        batch_size = completion_ids.size(0)
        
        # Find first sample with non-zero advantage (reward)
        batch_idx = None
        for i in range(batch_size):
            if advantages[i].item() != 0.0:
                batch_idx = i
                break
        
        if batch_idx is not None:
            # Found non-zero, print immediately
            self._do_print_debug(current_step, completion_ids, completion_mask, weights, advantages, batch_idx)
            self._last_debug_step = current_step
            self._fallback_debug_data = None
        else:
            # No non-zero found, save as fallback
            self._fallback_debug_data = (
                completion_ids.detach().clone(),
                completion_mask.detach().clone(),
                weights.detach().clone(),
                advantages.detach().clone(),
            )
            
            # If last minibatch of this step and still no non-zero, print fallback now
            if self._minibatch_count >= self.args.gradient_accumulation_steps:
                self._do_print_debug(current_step, *self._fallback_debug_data, batch_idx=0)
                self._last_debug_step = current_step
                self._fallback_debug_data = None
    
    def _do_print_debug(
        self,
        step: int,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        weights: torch.Tensor,
        advantages: torch.Tensor,
        batch_idx: int,
    ):
        """Actually print the debug output."""
        tokenizer = self.processing_class
        
        print(f"\n{'='*80}")
        print(f"TOKEN CREDIT DEBUG - STEP {step} (First Nonzero Reward Sample)")
        print(f"Credit Method: {self.credit_method}")
        print(f"{'='*80}")
        
        seq = completion_ids[batch_idx]
        mask = completion_mask[batch_idx]
        w = weights[batch_idx]
        adv = advantages[batch_idx].item()
        
        actual_len = mask.sum().int().item()

        print(f"\n[Sample {batch_idx}]")
        print(f"  Original advantage (scalar): {adv:.6f}")
        print(f"  Total tokens: {actual_len}")
        if self.credit_method == "exponential":
            print(f"  Decay base: {self.decay_base}")
        elif self.credit_method == "attention":
            print(f"  Attention epsilon: {self.attention_epsilon}")
            print(f"  Sigmoid temperature: {self.sigmoid_temperature}")
        elif self.credit_method == "sigmoid":
            print(f"  Sigmoid temperature: {self.sigmoid_temperature}")

        # Handle empty sequences
        if actual_len == 0:
            print(f"  [Empty sequence - no tokens to display]")
            print(f"\n{'='*80}\n")
            return

        # Find </think> position
        think_end_pos = self._find_think_end_position(seq[:actual_len])

        if think_end_pos is not None:
            print(f"  </think> found at token position: {think_end_pos}")
            print(f"  Reasoning tokens: {think_end_pos}")
            print(f"  Answer tokens: {actual_len - think_end_pos}")
        else:
            print(f"  </think> NOT FOUND - all tokens get weight 1.0")

        # Compute token-level advantages
        token_advantages = w[:actual_len] * adv

        print(f"\n  Token-level advantages summary:")
        print(f"    Total tokens: {actual_len}")
        print(f"    Total token-level advantages: {token_advantages.size(0)}")
        print(f"    Sum of weights: {w[:actual_len].sum().item():.4f}")
        print(f"    Min weight: {w[:actual_len].min().item():.6f}")
        print(f"    Max weight: {w[:actual_len].max().item():.6f}")
        print(f"    Mean weight: {w[:actual_len].mean().item():.6f}")
        
        # Show first and last few tokens with their weights and advantages
        print(f"\n  First 10 tokens (reasoning start):")
        print(f"    {'Pos':<5} {'Token':<20} {'Weight':<12} {'Advantage':<12}")
        print(f"    {'-'*49}")
        for t in range(min(10, actual_len)):
            token_str = tokenizer.decode([seq[t].item()]).replace('\n', '\\n')[:18]
            print(f"    {t:<5} {token_str:<20} {w[t].item():<12.6f} {token_advantages[t].item():<12.6f}")
        
        if think_end_pos is not None and think_end_pos > 10:
            print(f"\n  Tokens around </think> (pos {think_end_pos}):")
            print(f"    {'Pos':<5} {'Token':<20} {'Weight':<12} {'Advantage':<12}")
            print(f"    {'-'*49}")
            start = max(0, think_end_pos - 5)
            end = min(actual_len, think_end_pos + 5)
            for t in range(start, end):
                token_str = tokenizer.decode([seq[t].item()]).replace('\n', '\\n')[:18]
                marker = " <-- </think> ends" if t == think_end_pos else ""
                print(f"    {t:<5} {token_str:<20} {w[t].item():<12.6f} {token_advantages[t].item():<12.6f}{marker}")
        
        if actual_len > 20:
            print(f"\n  Last 10 tokens (answer end):")
            print(f"    {'Pos':<5} {'Token':<20} {'Weight':<12} {'Advantage':<12}")
            print(f"    {'-'*49}")
            for t in range(actual_len - 10, actual_len):
                token_str = tokenizer.decode([seq[t].item()]).replace('\n', '\\n')[:18]
                print(f"    {t:<5} {token_str:<20} {w[t].item():<12.6f} {token_advantages[t].item():<12.6f}")

        # Generate visualization plots
        if self._credit_plots_dir is not None:
            try:
                # Get tokens for the completion
                tokens = tokenizer.convert_ids_to_tokens(seq[:actual_len].cpu().tolist())

                # Find <think> start position
                think_start_pos = self._find_think_start_position(seq[:actual_len])

                # Method prefix for filename
                method_prefix = self.credit_method

                # Generate token highlighting plot
                print("\n  Generating credit visualization plots...")
                fig_token = plot_credit_token_highlighting(
                    tokens=tokens,
                    token_weights=w[:actual_len],
                    tokenizer=tokenizer,
                    think_start_pos=think_start_pos,
                    think_end_pos=think_end_pos,
                )
                token_plot_path = os.path.join(
                    self._credit_plots_dir,
                    f"credit_{method_prefix}_token_step_{step}.pdf"
                )
                fig_token.savefig(token_plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig_token)
                print(f"    Saved: {token_plot_path}")

                # Generate credit score line plot
                fig_score = plot_credit_scores(
                    tokens=tokens,
                    token_weights=w[:actual_len],
                    think_start_pos=think_start_pos,
                    think_end_pos=think_end_pos,
                )
                score_plot_path = os.path.join(
                    self._credit_plots_dir,
                    f"credit_{method_prefix}_score_step_{step}.pdf"
                )
                fig_score.savefig(score_plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig_score)
                print(f"    Saved: {score_plot_path}")

            except Exception as e:
                print(f"\n  Warning: Failed to generate plots: {e}")

        print(f"\n{'='*80}\n")
    
    def _compute_loss(self, model, inputs):
        """
        Override to implement token-level credit assignment.
        
        This directly computes the loss with per-token weighted advantages.
        """
        # Extract tensors
        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        old_per_token_logps = inputs["old_per_token_logps"]
        advantages = inputs["advantages"]  # [batch_size] - scalar per completion
        
        # Compute per-token weights
        # For attention method, this triggers Phase 2 (attention extraction)
        token_weights = self._compute_token_weights(completion_ids, completion_mask, prompt_ids=prompt_ids)
        
        # Debug printing (only main process)
        if self.debug_token_weights:
            self._print_token_weights_debug(
                completion_ids,
                completion_mask,
                token_weights,
                advantages,
            )
        
        # Log token weight statistics (local only - no distributed gather to avoid deadlock)
        mode = "train" if model.training else "eval"
        try:
            active_mask = completion_mask.bool()
            if active_mask.any():
                active_weights = token_weights[active_mask]
                if active_weights.numel() > 0:
                    if "token_weights/mean" not in self._metrics[mode]:
                        self._metrics[mode]["token_weights/mean"] = []
                        self._metrics[mode]["token_weights/min"] = []
                        self._metrics[mode]["token_weights/max"] = []
                    # Log local metrics only (no gather to avoid NCCL sync issues)
                    self._metrics[mode]["token_weights/mean"].append(active_weights.mean().item())
                    self._metrics[mode]["token_weights/min"].append(active_weights.min().item())
                    self._metrics[mode]["token_weights/max"].append(active_weights.max().item())
        except Exception as e:
            # Silently ignore logging errors to prevent training interruption
            pass
        
        # Concatenate prompt and completion for forward pass
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        
        # Forward pass to get logits
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = output.logits
        
        # Get logits for completion tokens only (the last logits_to_keep positions)
        logits = logits[:, -logits_to_keep:, :]
        
        # Compute per-token log probabilities using next-token prediction
        # logits[:, t, :] predicts token at position t+1
        # So we shift: logits[:, :-1, :] predicts completion_ids[:, 1:]
        logits_for_loss = logits[:, :-1, :].contiguous()
        labels = completion_ids[:, 1:].contiguous()
        
        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(logits_for_loss, dim=-1)
        per_token_logps = torch.gather(
            log_probs, 
            dim=-1, 
            index=labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Adjust masks and weights for the shifted sequence (one less position)
        completion_mask_for_loss = completion_mask[:, 1:].contiguous()
        token_weights_for_loss = token_weights[:, 1:].contiguous()
        
        # Handle old_per_token_logps length
        # It should match per_token_logps length
        if old_per_token_logps.size(1) == per_token_logps.size(1) + 1:
            # old_per_token_logps includes one extra position, shift it
            old_logps = old_per_token_logps[:, 1:].contiguous()
        elif old_per_token_logps.size(1) == per_token_logps.size(1):
            old_logps = old_per_token_logps
        else:
            # Truncate or pad to match
            min_len = min(old_per_token_logps.size(1), per_token_logps.size(1))
            old_logps = old_per_token_logps[:, :min_len]
            per_token_logps = per_token_logps[:, :min_len]
            completion_mask_for_loss = completion_mask_for_loss[:, :min_len]
            token_weights_for_loss = token_weights_for_loss[:, :min_len]
        
        # Compute importance sampling ratio
        log_ratio = per_token_logps - old_logps
        ratio = torch.exp(log_ratio)
        
        # ===========================================
        # TOKEN-LEVEL CREDIT ASSIGNMENT
        # ===========================================
        # Apply per-token weights to advantages
        # Shape: [batch_size] unsqueeze -> [batch_size, 1]
        # Multiply with weights: [batch_size, seq_len]
        per_token_advantages = advantages.unsqueeze(-1) * token_weights_for_loss
        
        # Get clipping bounds from args
        eps_low = self.args.epsilon
        eps_high = getattr(self.args, 'epsilon_high', None)
        if eps_high is None:
            eps_high = self.args.epsilon
        
        # Clipped surrogate objective with per-token advantages
        coef_1 = ratio * per_token_advantages
        coef_2 = torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * per_token_advantages
        per_token_loss = -torch.min(coef_1, coef_2)
        
        # KL penalty if beta > 0
        if self.args.beta != 0.0 and "ref_per_token_logps" in inputs:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            # Match lengths
            if ref_per_token_logps.size(1) == per_token_logps.size(1) + 1:
                ref_logps = ref_per_token_logps[:, 1:per_token_logps.size(1)+1]
            elif ref_per_token_logps.size(1) >= per_token_logps.size(1):
                ref_logps = ref_per_token_logps[:, :per_token_logps.size(1)]
            else:
                ref_logps = ref_per_token_logps
                per_token_logps_kl = per_token_logps[:, :ref_logps.size(1)]
                per_token_kl = (
                    torch.exp(ref_logps - per_token_logps_kl)
                    - (ref_logps - per_token_logps_kl)
                    - 1
                )
                # Pad to match per_token_loss size
                pad_size = per_token_loss.size(1) - per_token_kl.size(1)
                if pad_size > 0:
                    per_token_kl = torch.nn.functional.pad(per_token_kl, (0, pad_size), value=0)
                per_token_loss = per_token_loss + self.args.beta * per_token_kl
                ref_logps = None  # Mark as handled
            
            if ref_logps is not None:
                per_token_kl = (
                    torch.exp(ref_logps - per_token_logps)
                    - (ref_logps - per_token_logps)
                    - 1
                )
                per_token_loss = per_token_loss + self.args.beta * per_token_kl
        
        # Handle truncated completions mask if enabled
        if self.args.mask_truncated_completions:
            truncation_mask = inputs.get("truncation_mask", None)
            if truncation_mask is not None:
                # truncation_mask is [batch_size], need to expand
                completion_mask_for_loss = completion_mask_for_loss * truncation_mask.unsqueeze(-1).float()
        
        # Apply completion mask
        masked_loss = per_token_loss * completion_mask_for_loss
        
        # Aggregate loss based on loss_type
        loss_type = self.args.loss_type
        
        if loss_type == "grpo":
            # Normalize by sequence length (original GRPO)
            per_seq_loss = masked_loss.sum(dim=1)
            per_seq_length = completion_mask_for_loss.sum(dim=1).clamp(min=1)
            loss = (per_seq_loss / per_seq_length).mean()
            
        elif loss_type == "dapo":
            # Normalize by total active tokens (DAPO style)
            total_tokens = completion_mask_for_loss.sum().clamp(min=1)
            loss = masked_loss.sum() / total_tokens

        else:
            # Default to DAPO
            total_tokens = completion_mask_for_loss.sum().clamp(min=1)
            loss = masked_loss.sum() / total_tokens
        
        return loss / self.args.gradient_accumulation_steps


# =============================================================================
# DATASET SETUP
# =============================================================================

SYSTEM_PROMPT = "You are a helpful assistant."
# SYSTEM_PROMPT = r"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process is enclosed within <think>...</think> tags, and the answer follows directly after with the final answer in \boxed{}, i.e., <think> reasoning process here </think> answer here \boxed{final answer here}."

def add_system_prompt(example):
    example["prompt"] = [{"role": "system", "content": SYSTEM_PROMPT}] + example["prompt"]
    return example

# train_dataset = load_dataset("trl-lib/DeepMath-103K", split="train").shuffle(seed=42)
train_dataset = (
    load_dataset("allenai/Dolci-RL-Zero-Math-7B", split="train")
    .rename_column("prompt", "question")
    .rename_column("messages", "prompt")
    .rename_column("ground_truth", "solution")
    .map(add_system_prompt)
    .shuffle(seed=42)
)


# =============================================================================
# TRAINING CONFIG
# =============================================================================

training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    
    # Max steps takes precedence over epochs
    num_train_epochs=1,
    max_steps=500,
    
    per_device_train_batch_size=1,   # Per device batch size
    gradient_accumulation_steps=32,  # Step size
    num_generations=8,               # Group size
    # 3 (gpus) × 1 (batch) × 32 (accum) = 96 unique prompts per gradient update
    # 3 (gpus) × 1 (batch) × 32 (accum) × 8 (gens) = 768 completions per gradient update
    # CONCURRENT_SEQUENCES = per_device_train_batch_size × num_generations = 1 × 8 = 8

    # ===================================================================================
    # MEMORY ANALYSIS PER GPU (4× 80GB H100)
    # ===================================================================================
    # GPU ALLOCATION:
    # - GPUs 0, 1, 2: vLLM generation (colocate) + TRL training (ZeRO-2)
    # - GPU 3: Attention extraction only (when credit_method="attention")
    #
    # SHARDING BEHAVIOR:
    # - vLLM: NOT sharded - each GPU runs full model + KV cache independently
    # - TRL ZeRO-2: Gradients + Optimizer sharded across 3 GPUs, Weights replicated
    # - Logits: NOT sharded - each GPU computes logits for its local batch
    # ===================================================================================

    # VLLM MEMORY PER GPU (not sharded, each GPU has full copy):
    # VLLM_MEMORY = MODEL_WEIGHTS (parameters × 2 bf16) + KV_CACHE
    # KV_CACHE_PER_TOKEN = 2 (K+V tensors) × layers × kv_heads × head_dim × 2 (bf16)
    # KV_CACHE_TOTAL = KV_CACHE_PER_TOKEN × max_seq_length × CONCURRENT_SEQUENCES
    # Qwen3-0.6B: 1.2GB + 28KB (2 × 28 × 4 × 64 × 2 = 28,672 bytes) × 25600 × 8 = 1.2GB + 5.5GB → VLLM_PER_GPU = 6.7GB
    # Qwen3-1.7B: 3.4GB + 112KB (2 × 28 × 8 × 128 × 2 = 114,688 bytes) × 25600 × 8 = 3.4GB + 21.9GB → VLLM_PER_GPU = 25.3GB

    # Sequence lengths
    max_prompt_length=1024,
    max_completion_length=24576,

    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.4,  # Sets 32GB ceiling; vLLM allocates min(ceiling, actual_need) → ~6.7GB for 0.6B, ~25.3GB for 1.7B
    vllm_enable_sleep_mode=False,
    vllm_max_model_length=25600,

    # TRL MEMORY PER GPU (ZeRO-2: weights replicated, gradients + optimizer sharded across 3 GPUs):
    # Qwen3-0.6B FULL_FT: WEIGHTS (bf16, replicated) = 0.6B × 2 = 1.2GB, GRADIENTS (bf16, sharded) = 0.6B × 2 ÷ 3 = 0.4GB, OPTIMIZER (fp32 AdamW, sharded) = 0.6B × 12 ÷ 3 = 2.4GB, ACTIVATIONS ≈ 1.5GB → TRL_PER_GPU = 5.5GB
    # Qwen3-1.7B FULL_FT: WEIGHTS (bf16, replicated) = 1.7B × 2 = 3.4GB, GRADIENTS (bf16, sharded) = 1.7B × 2 ÷ 3 = 1.1GB, OPTIMIZER (fp32 AdamW, sharded) = 1.7B × 12 ÷ 3 = 6.8GB, ACTIVATIONS ≈ 2.5GB → TRL_PER_GPU = 13.8GB
    # Qwen3-1.7B LORA r=128 (~51M trainable): BASE_WEIGHTS (bf16, frozen) = 3.4GB, LORA_WEIGHTS (bf16) = 51M × 2 = 0.1GB, LORA_GRADIENTS (bf16) = 0.1GB, LORA_OPTIMIZER (fp32 AdamW) = 51M × 12 = 0.6GB, ACTIVATIONS ≈ 2.5GB → TRL_PER_GPU = 6.7GB

    # LOGITS TENSOR PER GPU (not sharded, each GPU computes its own local batch):
    # LOGITS_PER_GPU = CONCURRENT_SEQUENCES × completion_length × vocab_size × 2 (bf16) = 8 × 24576 × 151936 × 2 = 55.8GB

    # TOTAL PEAK MEMORY PER GPU (GPUs 0, 1, 2):
    # Qwen3-0.6B FULL_FT: VLLM (6.7GB) + TRL (5.5GB) + LOGITS (55.8GB) = 68.0GB per GPU
    # Qwen3-1.7B FULL_FT: VLLM (25.3GB) + TRL (13.8GB) + LOGITS (55.8GB) = 94.9GB per GPU
    # Qwen3-1.7B LORA r=128: VLLM (25.3GB) + TRL (6.7GB) + LOGITS (55.8GB) = 87.8GB per GPU
    #
    # WHY BATCH_SIZE=2 WOULD OOM (even for 0.6B):
    # - CONCURRENT_SEQUENCES doubles: 8 → 16
    # - LOGITS_PER_GPU doubles: 55.8GB → 111.6GB → exceeds 80GB
    #
    # WHY 1.7B WOULD OOM:
    # - vLLM KV cache 4× larger (112KB vs 28KB per token): 6.7GB → 25.3GB
    # - Total: VLLM (25.3GB) + TRL (13.8GB) + LOGITS (55.8GB) = 94.9GB → exceeds 80GB
    #
    # WHY LORA DOESN'T HELP FOR 1.7B:
    # - LoRA saves only TRL optimizer/gradients (~7GB savings)
    # - But VLLM (25.3GB) + LOGITS (55.8GB) = 81.1GB → exceeds 80GB

    # GPU 3 MEMORY (Attention Extraction, only for credit_method="attention"):
    # Qwen3-0.6B: MODEL (bf16, eager) = 1.2GB, ATTENTION_MATRIX (24K seq) = 24K × 24K × 4 = 2.3GB → ATTN_PER_GPU = 3.5GB

    learning_rate=1e-6,
    lr_scheduler_type="constant",
    warmup_ratio=0.0,
    
    temperature=1.0,
    
    logging_steps=1,
    save_steps=25,
    eval_strategy="no",
    report_to="wandb",
    run_name=WANDB_RUN_NAME,
    
    bf16=True,
    gradient_checkpointing=True,
    torch_compile=True,

    # DAPO
    loss_type="dapo",                 # Token-level normalization
    epsilon=0.2,                      # Lower clipping bound (default)
    epsilon_high=0.28,                # Upper clipping bound (DAPO recommends 0.28)
    mask_truncated_completions=True,  # Exclude truncated completions from loss
    scale_rewards=False,              # No reward scaling (removes difficulty bias)
)


# =============================================================================
# TRAINER SETUP
# =============================================================================

REWARD_FUNC = reasoning_accuracy_reward

debug_contexts = True       # Print prompts, completions, and solutions per training step
debug_token_weights = True  # Print token-level credit assignment info per training step

if debug_contexts:
    reward_func = wrap_reward(REWARD_FUNC)
else:
    reward_func = REWARD_FUNC

trainer = TokenLevelGRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    # model="Qwen/Qwen3-1.7B",
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=train_dataset,
    # Token-level credit assignment parameters
    think_end_token="</think>",
    credit_method="attention",  # "vanilla", "exponential", "attention", "linear", "sigmoid", "zero"
    decay_base=0.995,  # Exponential decay base for reasoning tokens (used when credit_method="exponential")
    attention_num_layers=1,  # Number of last layers to average attention from (used when credit_method="attention")
    attention_epsilon=1e-6,  # Log transform epsilon (used when credit_method="attention")
    sigmoid_temperature=3.0,  # Temperature for sigmoid contrast (used when credit_method="attention" or "sigmoid")
    baseline_credit=0.0,  # Baseline credit (shifts credit floor)
    debug_token_weights=debug_token_weights,
)

if debug_contexts:
    add_debug_callback(trainer, train_dataset)

trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
