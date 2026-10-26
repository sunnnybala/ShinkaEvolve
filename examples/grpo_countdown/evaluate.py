"""
Evaluator for GRPO Countdown task.
Optimizes combined_score = accuracy (%) at max_new_tokens = 512.
"""

import os
import argparse
from typing import Any, Dict, List, Tuple

from shinka.core import run_shinka_eval


def _validate_result(run_output: Dict[str, Any]) -> Tuple[bool, str | None]:
    if not isinstance(run_output, dict):
        return False, "Run output is not a dict"
    required = ["accuracy", "avg_output_tokens"]
    for k in required:
        if k not in run_output:
            return False, f"Missing key: {k}"
    return True, None


def aggregate_grpo_metrics(results: List[Dict[str, Any]], results_dir: str) -> Dict[str, Any]:
    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}

    # Single-run by default; average if multiple
    mean_acc = sum(r.get("accuracy", 0.0) for r in results) / len(results)
    mean_reward = sum(r.get("mean_reward", 0.0) for r in results) / len(results)
    mean_tokens = sum(r.get("avg_output_tokens", 0.0) for r in results) / len(results)

    public_metrics = {
        "accuracy": float(mean_acc),
        "mean_reward": float(mean_reward),
        "avg_output_tokens": float(mean_tokens),
    }
    private_metrics = {
        "subset_size": results[0].get("subset_size", 0),
        "count_correct": results[0].get("count_correct", 0),
        "count_partial": results[0].get("count_partial", 0),
        "count_failed": results[0].get("count_failed", 0),
    }

    return {
        "combined_score": float(mean_acc),  # fitness = accuracy (%)
        "public": public_metrics,
        "private": private_metrics,
    }


def main(program_path: str, results_dir: str):
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    def _get_kwargs(_: int) -> Dict[str, Any]:
        return {"seed": 42, "subset_size": int(os.environ.get("GRPO_EVAL_SUBSET", 64))}

    def _aggregator_with_context(r: List[Dict[str, Any]]):
        return aggregate_grpo_metrics(r, results_dir)

    def _validator(result: Dict[str, Any]):
        return _validate_result(result)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_countdown_eval",
        num_runs=1,
        get_experiment_kwargs=_get_kwargs,
        validate_fn=_validator,
        aggregate_metrics_fn=_aggregator_with_context,
    )

    if correct:
        print("Evaluation completed successfully.")
    else:
        print(f"Evaluation failed: {error_msg}")

    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO Countdown evaluator")
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to program to evaluate (must contain 'run_countdown_eval')",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Dir to save results (metrics.json, correct.json)",
    )
    args = parser.parse_args()
    main(args.program_path, args.results_dir)


