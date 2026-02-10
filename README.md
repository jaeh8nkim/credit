<h1 align="center">
  Credit Where Credit Is Due:<br>
  Attention-weighted Token-level<br>
  Credit Assignment in Policy Gradient
</h1>

<img width="500" height="339" alt="atpo_overview" src="https://github.com/user-attachments/assets/af1d3c0b-bb9f-4b7d-887f-42aab66ee548" />

### Figure 1. Schematic comparison of the GRPO and ATPO training pipelines.

GRPO calculates a group-level advantage and applies it uniformly to all tokens, while ATPO inserts an attention-weighted advantage module that re-weights credit per token using the policy model's own attention maps. 
This enables token-level credit assignment without auxiliary models or additional forward passes.

---

<img width="500" height="438" alt="image" src="https://github.com/user-attachments/assets/9bce60f8-3299-4410-9e5d-a38bbc72cb96" />

### Figure 2. Head-averaged attention map from the last transformer layer for a single reasoning trace.

Red triangles mark the `<think>` and `</think>` delimiters; the boxed region highlights attention from answer tokens to reasoning tokens. 
Compressing this region vertically yields a per-token credit score, capturing how much each reasoning token contributed to the final answer.

---

<img width="500" height="837" alt="fig 2" src="https://github.com/user-attachments/assets/a1280717-1435-4d3c-b1a8-0fe465f34769" />

### Figure 3. Token-level credit scores for three reasoning traces of varying length, computed via log-transform, z-score normalization, and sigmoid rescaling. 

Red triangles mark `<think>`/`</think>` delimiters; heatmap strips on the right visualize the same scores spatially, with brighter red indicating higher credit. 
Credit distributes non-uniformly according to the semantic progression of each trace, with some traces front-loading high-credit tokens and others concentrating them later.

---

<img width="1000" height="262" alt="fig 3" src="https://github.com/user-attachments/assets/ae3bee74-93ee-47c2-9ac0-fa00acb2b4e9" />

### Figure 4. Training reward (left) and AIME25 evaluation accuracy (avg@8, right) across ablation variants on Qwen3-0.6B.

DAPO serves as the uniform-credit baseline. 
ATPO variants explore sigmoid temperature (2.0), a 0.5 baseline floor that halves attention scores before adding them to a constant credit, and averaging over the last 8 of 28 layers instead of the final layer alone. 
ATPO with a 0.5 baseline floor maintains competitive accuracy with DAPO while the pure attention-only variant degrades, suggesting that a minimum credit floor prevents overly aggressive down-weighting of tokens.

---

<img width="1000" height="262" alt="fig 4" src="https://github.com/user-attachments/assets/b91fc369-c52a-4a7e-b0a6-27d43790d8ea" />

### Figure 5. Mean completion length during training (left) and at evaluation (right) for the same ablation variants.

DAPO maintains relatively stable lengths, while pure ATPO initially increases then sharply collapses, mirroring its accuracy degradation. 
The 0.5 baseline variant produces moderately longer responses that remain stable, indicating that the credit floor prevents the length collapse associated with overly aggressive token-level weighting.

---

### Table 1. Token-level credit weights and resulting advantages at three positions in a correct reasoning trace (group advantage = 0.75).

Reasoning tokens near `<think>` receive fractional weights (0.25–0.59), reducing their advantage proportionally, while tokens approaching `</think>` and all answer tokens receive near-maximum weight. 
This illustrates how ATPO preserves the full learning signal for answer tokens while selectively modulating credit within the reasoning trace.

```text
  First 10 tokens (reasoning start):
    Pos   Token                Weight       Advantage   
    -------------------------------------------------
    0     <think>              1.000000     0.750000    
    1     \n                   0.324219     0.243164    
    2     Okay                 0.539062     0.404297    
    3     ,                    0.343750     0.257812    
    4      so                  0.247070     0.185303    
    5      there               0.304688     0.228516    
    6     's                   0.343750     0.257812    
    7      this                0.589844     0.442383    
    8      problem             0.304688     0.228516    
    9      about               0.498047     0.373535    

  Tokens around </think> (pos 7329):
    Pos   Token                Weight       Advantage   
    -------------------------------------------------
    7324  boxed                0.910156     0.682617    
    7325  {                    0.871094     0.653320    
    7326  9                    0.968750     0.726562    
    7327  }.\n                 0.949219     0.711914    
    7328  </think>             0.980469     0.735352    
    7329  \n\n                 1.000000     0.750000     <-- </think> ends
    7330  To                   1.000000     0.750000    
    7331   determine           1.000000     0.750000    
    7332   the                 1.000000     0.750000    
    7333   probability         1.000000     0.750000    

  Last 10 tokens (answer end):
    Pos   Token                Weight       Advantage   
    -------------------------------------------------
    7742   $\n\n               1.000000     0.750000    
    7743  $$                   1.000000     0.750000    
    7744  \n                   1.000000     0.750000    
    7745  \                    1.000000     0.750000    
    7746  boxed                1.000000     0.750000    
    7747  {                    1.000000     0.750000    
    7748  9                    1.000000     0.750000    
    7749  }\n                  1.000000     0.750000    
    7750  $$                   1.000000     0.750000    
    7751  <|im_end|>           1.000000     0.750000  
```

---

## Abstract

In Reinforcement Learning with Verifiable Reward (RLVR), policy gradient methods assign a single reward based on final answer correctness and distribute it uniformly across all tokens in a generated sequence. 
This uniform credit assignment creates a fundamental inefficiency: all tokens receive equal credit regardless of their actual contribution to the solution, meaning the learning signal conflates productive reasoning steps with redundant or irrelevant ones.
Recent approaches to address this limitation either rely on external statistical methods that lack access to the policy model's internal representations, or require maintaining separate reward models that double computational costs during training.
We propose Attention-weighted Token-level Policy Optimization (ATPO), a method that leverages the policy model's own self-attention mechanisms for fine-grained, token-level credit assignment.
ATPO extracts credit signals directly from the policy model's attention patterns, systematically identifying each token's contribution to reasoning success. 
These attention-derived weights are then used to adaptively scale the policy gradient update for each token.
Our experiments demonstrate that ATPO provides a more precise learning signal compared to uniform credit assignment, leading to improved accuracy on reasoning benchmarks.
These findings suggest that the policy model's internal representations contain sufficient information for effective credit assignment, offering a computationally efficient path toward higher-quality outputs in RLVR.

## Related Work

### Credit Assignment

**Step-Level Credit Assignment**

Reasoning traces can be decomposed into intermediate steps, motivating methods that provide feedback at the step level rather than the trajectory level. Some work estimates each step's value through Monte Carlo rollouts, sampling multiple continuations from intermediate states and averaging their success rates to predict expected future reward [1]. This eliminates the need for a separate value network but requires substantial inference overhead due to multiple rollouts per step. Subsequent work extends this direction by performing branching at high-entropy tokens rather than relying on predefined linguistic segmentation [2]. While these step-level methods provide finer-grained credit assignment than trajectory-level rewards, they still distribute credit uniformly within each step, potentially reinforcing unproductive tokens alongside productive ones.

**Token-Level Credit Assignment**

Recent work has pursued even finer granularity at the token level. Some approaches train a separate discriminator to predict token-level preferences using synthetic labels [3]. Others achieve token-level credit assignment through the reward model's incremental outputs, running the reward model on partial sequences up to each token position and computing rewards as the difference between consecutive evaluations [4]. While requiring no additional training of the reward model, this approach necessitates numerous forward passes during training.

Statistical methods offer an alternative path to token-level signals without auxiliary models. One approach constructs contingency tables for each unique token and applies statistical tests to quantify associations between tokens and correctness [5]. Another line of work identifies that only a minority of tokens exhibit high entropy and act as critical decision points, restricting policy gradient updates to only these high-entropy tokens during training [6]. While entropy provides some insight into the model's internal uncertainty, we posit that attention patterns may offer a more direct view into how the model attributes importance across tokens.

**Attention-Based Credit Assignment**

Attention mechanisms offer a principled approach to estimating token contributions by examining how models aggregate information. Prior work has used the reward model's attention weights to redistribute rewards at the token level, providing dense rewards without additional training [7]. Similar approaches have been explored in sequential decision-making domains beyond language modeling, training a separate transformer to predict episodic returns and leveraging its attention patterns to redistribute sparse rewards [8]. While these methods demonstrate the viability of attention-based reward redistribution, they require maintaining a separate reward model typically comparable in size to the policy model, effectively doubling computational costs during training.

Recent work demonstrates that the policy model's own attention patterns can identify token-level importance, using attention scores to prune redundant reasoning at inference time [9]. However, this applies attention-based analysis for inference-time optimization rather than training-time credit assignment. Our work bridges this gap by leveraging the policy model's own self-attention mechanisms directly for credit assignment during training, avoiding the computational overhead of auxiliary models while incorporating the policy's internal representations into the learning signal.

---

### References

[1] Kazemnejad et al. "VinePPO: Refining Credit Assignment in RL Training of LLMs." ICML, 2025.

[2] Hou et al. "TreeRL: LLM Reinforcement Learning with On-Policy Tree Search." ACL, 2025.

[3] Yoon et al. "TLCR: Token-Level Continuous Reward for Fine-grained Reinforcement Learning from Human Feedback." Findings of ACL, 2024.

[4] Li et al. "RED: Unleashing Token-Level Rewards from Holistic Feedback via Reward Redistribution." EMNLP, 2025.

[5] Sun et al. "KTAE: A Model-Free Algorithm to Key-Tokens Advantage Estimation in Mathematical Reasoning." NeurIPS, 2025.

[6] Wang et al. "Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning." arXiv:2506.01939, 2025.

[7] Chan et al. "Dense Reward for Free in Reinforcement Learning from Human Feedback." ICML, 2024.

[8] Holmes and Chi. "Attention-Based Reward Shaping for Sparse and Delayed Rewards." arXiv:2505.10802, 2025.

[9] Choi et al. "Think Clearly: Improving Reasoning via Redundant Token Pruning." ES-FoMo III Workshop, 2025.

---

## Why We Dropped This Research Direction

ATPO demonstrated that attention patterns correlate with token importance, and the baseline-floor variant maintained competitive accuracy with DAPO. 
However, several fundamental limitations led us to discontinue this line of work.

- **Attention is not causal contribution.** High attention weight means the model referenced a token during generation, not that removing it would have changed the outcome. Formatting tokens and delimiters frequently receive high attention despite carrying no semantic load, while decisive reasoning steps may receive low attention if processed efficiently. This gap between "information reference" and "counterfactual responsibility" undermines the core premise.

- **Arbitrary bounds on credit scores lack theoretical grounding.** The credit weights are clipped to [-1, 1], but this range is an ad hoc choice with no connection to the reward structure or advantage scale. Any other bounds would be equally defensible (or indefensible), and the training dynamics shift meaningfully depending on this choice. Without a principled way to set these limits, the method introduces a sensitive hyperparameter that cannot be calibrated from first principles.

- **Sigmoid scaling discards half the credit spectrum.** The sigmoid transformation centers around the median, so roughly half of all tokens always receive below-baseline credit. This is a structural artifact of the normalization, not a property of the reasoning trace. Tokens below median attention are systematically down-weighted regardless of their actual contribution.

- **Single-layer extraction is insufficient, but alternatives are equally arbitrary.** Our default implementation uses only the last transformer layer's attention map, yet reasoning causality likely spans multiple layers. The attention tensor is inherently multi-dimensional (layers × heads × positions), but credit assignment requires a single scalar per token. Every reduction strategy, whether last-layer or multi-layer averaging, introduces unjustified design choices.

These issues compound. A causally ungrounded signal susceptible to optimization pressure does not provide a reliable foundation for credit assignment. The baseline-floor variant mitigated failure modes precisely by dampening the attention signal, suggesting the attention component contributed noise rather than genuine learning signal.


