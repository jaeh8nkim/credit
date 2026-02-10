# Evaluation script for GRPO checkpoints with avg@k support.

"""
- Evaluate with pass@1:
python eval.py --checkpoint Qwen3-0.6B-ATPO/checkpoint-50 --dataset aime25 > eval.log 2>&1

- Evaluate with avg@8:
python eval.py --checkpoint Qwen3-0.6B-ATPO/checkpoint-50 --dataset aime25 --k 8 > eval.log 2>&1

- Evaluate all checkpoints:
python eval.py --checkpoint Qwen3-0.6B-ATPO --all --dataset aime25 --k 8 > eval.log 2>&1
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from trl.rewards import accuracy_reward
from vllm import LLM, SamplingParams


# ================================================================================
# Dataset Registry - Add new datasets here
# ================================================================================
DATASET_REGISTRY = {
    "aime25": {
        "path": "math-ai/aime25",
        "split": "test",
        "prompt_column": "problem",
        "answer_column": "answer",
    },
    # Add more datasets here:
    # "math500": {
    #     "path": "HuggingFaceH4/MATH-500",
    #     "split": "test",
    #     "prompt_column": "problem",
    #     "answer_column": "answer",
    # },
    # "amc23": {
    #     "path": "math-ai/amc23",
    #     "split": "test",
    #     "prompt_column": "problem",
    #     "answer_column": "answer",
    # },
}


def load_eval_dataset(dataset_name: str):
    """Load evaluation dataset from registry."""
    if dataset_name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    
    config = DATASET_REGISTRY[dataset_name]
    dataset = load_dataset(config["path"], split=config["split"])
    
    # Rename columns to standard names
    if config["prompt_column"] != "prompt":
        dataset = dataset.rename_column(config["prompt_column"], "prompt")
    if config["answer_column"] != "answer":
        dataset = dataset.rename_column(config["answer_column"], "answer")
    
    return dataset


def evaluate_checkpoint(
    checkpoint_path: str,
    eval_dataset,
    tokenizer,
    k: int = 1,
    max_tokens: int = 24576,
    temperature: float = 1.0,
    seed: int = 42,
):
    """
    Evaluate a checkpoint with pass@k and avg@k.
    
    pass@k: For each problem, 1 if at least one of k completions is correct, else 0.
            Average across all problems.
    
    avg@k: Generate k completions per problem, compute accuracy for each,
           then average across all problems.
    
    Args:
        checkpoint_path: Path to model checkpoint
        eval_dataset: Dataset with 'prompt' and 'answer' columns
        tokenizer: Tokenizer for formatting prompts
        k: Number of completions per problem
        max_tokens: Max generation tokens
        temperature: Sampling temperature
        seed: Random seed for reproducibility
    
    Returns:
        dict with evaluation results
    """
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {checkpoint_path}")
    print(f"avg@{k} evaluation")
    print(f"{'='*80}")
    
    # Load model with vLLM
    llm = LLM(
        model=checkpoint_path,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        max_model_len=32768,
    )
    
    # Temperature 0 for k=1 (greedy), otherwise use specified temperature
    if k == 1:
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,  # Greedy for pass@1
            seed=seed,
        )
    else:
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            n=k,  # Generate k completions per prompt
            seed=seed,
        )
    
    # Prepare samples
    samples = list(eval_dataset)
    
    # Format prompts
    prompts = []
    answers = []
    for example in samples:
        messages = [{"role": "user", "content": example["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt_text)
        answers.append(example["answer"])
    
    print(f"Generating {len(prompts)} prompts x {k} completions = {len(prompts) * k} total completions...")
    
    # Generate
    outputs = llm.generate(prompts, sampling_params)
    
    # Score with avg@k and pass@k
    problem_results = []
    total_correct = 0
    total_completions = 0
    total_output_tokens = 0
    total_passed = 0  # For pass@k: count problems with at least one correct
    
    for i, (output, answer) in enumerate(zip(outputs, answers)):
        # Get all k completions for this problem
        completions = [o.text for o in output.outputs]
        
        # Score each completion and count tokens
        correct_count = 0
        problem_tokens = 0
        for completion in completions:
            # Count tokens in this completion
            num_tokens = len(tokenizer.encode(completion, add_special_tokens=False))
            problem_tokens += num_tokens
            
            reward = accuracy_reward(
                completions=[[{"content": completion}]],
                solution=[answer],
            )[0]
            if reward > 0:
                correct_count += 1
        
        # avg@k for this problem
        problem_accuracy = correct_count / len(completions)
        avg_tokens_for_problem = problem_tokens / len(completions)
        
        # pass@k for this problem: 1 if at least one correct, 0 otherwise
        problem_passed = 1 if correct_count > 0 else 0
        total_passed += problem_passed
        
        problem_results.append({
            "problem_idx": i,
            "answer": answer,
            "k": len(completions),
            "correct": correct_count,
            "accuracy": problem_accuracy,
            "passed": problem_passed,
            "total_tokens": problem_tokens,
            "avg_tokens": avg_tokens_for_problem,
        })
        
        total_correct += correct_count
        total_completions += len(completions)
        total_output_tokens += problem_tokens
    
    # Compute overall avg@k
    avg_at_k = sum(r["accuracy"] for r in problem_results) / len(problem_results)
    
    # Compute pass@k (fraction of problems with at least one correct completion)
    pass_at_k = total_passed / len(problem_results)
    
    # Also compute raw accuracy (total correct / total completions)
    raw_accuracy = total_correct / total_completions
    
    # Compute average output length
    avg_output_length = total_output_tokens / total_completions
    
    print(f"\nResults:")
    print(f"  pass@{k}: {pass_at_k:.1%} ({total_passed}/{len(problem_results)} problems)")
    print(f"  avg@{k}: {avg_at_k:.1%}")
    print(f"  Raw accuracy: {total_correct}/{total_completions} = {raw_accuracy:.1%}")
    print(f"  Avg output length: {avg_output_length:.1f} tokens")
    
    # Cleanup
    del llm
    torch.cuda.empty_cache()
    
    return {
        "checkpoint": checkpoint_path,
        "k": k,
        "pass_at_k": pass_at_k,
        "avg_at_k": avg_at_k,
        "raw_accuracy": raw_accuracy,
        "total_correct": total_correct,
        "total_completions": total_completions,
        "num_problems": len(problem_results),
        "num_passed": total_passed,
        "total_output_tokens": total_output_tokens,
        "avg_output_length": avg_output_length,
        "problem_results": problem_results,
    }


def find_checkpoints(base_path: str) -> list[str]:
    """Find all checkpoint directories in a path."""
    base = Path(base_path)
    checkpoints = []
    
    for item in sorted(base.iterdir()):
        if item.is_dir() and item.name.startswith("checkpoint-"):
            checkpoints.append(str(item))
    
    return checkpoints


def main():
    parser = argparse.ArgumentParser(description="Evaluate GRPO checkpoints with avg@k")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path or base directory")
    parser.add_argument("--all", action="store_true", help="Evaluate all checkpoints in directory")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY.keys()), 
                        help="Evaluation dataset")
    parser.add_argument("--k", type=int, default=1, help="Number of completions per problem (avg@k)")
    parser.add_argument("--max-tokens", type=int, default=24576, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (for k>1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None, help="Output file (default: eval_{dataset}_k{k}.json)")
    args = parser.parse_args()
    
    # Default output filename
    if args.output is None:
        args.output = f"eval_{args.dataset}_k{args.k}.json"
    
    # Load eval dataset
    print(f"Loading evaluation dataset: {args.dataset}")
    eval_dataset = load_eval_dataset(args.dataset)
    print(f"Loaded {len(eval_dataset)} problems")
    
    # Load tokenizer from checkpoint or base model
    if args.all:
        checkpoints = find_checkpoints(args.checkpoint)
        if not checkpoints:
            print(f"No checkpoints found in {args.checkpoint}")
            return
        tokenizer_path = checkpoints[0]
    else:
        checkpoints = [args.checkpoint]
        tokenizer_path = args.checkpoint
    
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Evaluate each checkpoint
    all_results = []
    for ckpt in checkpoints:
        result = evaluate_checkpoint(
            ckpt,
            eval_dataset,
            tokenizer,
            k=args.k,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )
        all_results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Summary (avg@{args.k} on {args.dataset})")
    print(f"{'='*80}")
    for r in all_results:
        print(f"{r['checkpoint']}: pass@{args.k}={r['pass_at_k']:.1%}, avg@{args.k}={r['avg_at_k']:.1%}, avg_len={r['avg_output_length']:.1f} tokens")
    
    # Save results (without per-problem details for cleaner output)
    save_results = []
    for r in all_results:
        save_results.append({
            "checkpoint": r["checkpoint"],
            "dataset": args.dataset,
            "k": r["k"],
            "pass_at_k": r["pass_at_k"],
            "avg_at_k": r["avg_at_k"],
            "raw_accuracy": r["raw_accuracy"],
            "total_correct": r["total_correct"],
            "total_completions": r["total_completions"],
            "num_problems": r["num_problems"],
            "num_passed": r["num_passed"],
            "total_output_tokens": r["total_output_tokens"],
            "avg_output_length": r["avg_output_length"],
        })
    
    with open(args.output, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()