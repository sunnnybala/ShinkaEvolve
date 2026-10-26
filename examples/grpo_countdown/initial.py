import os
import random
import math
import re
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from collections import Counter


# EVOLVE-BLOCK-START
# Prompt and decoding configuration that Shinka can optimize.
TEMPLATE = (
    """Using the numbers {numbers}, create an equation that equals {target}.\n"
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.\n"
    "Show your reasoning in <think> </think> tags. And return the final equation in <answer> </answer> tags.\n"
    "Keep your reasoning concise."""
)

def init_generation_params() -> Dict[str, Any]:
    return {
        "temperature": 0.8,
        "top_p": 0.95,
        "do_sample": True,
        "max_new_tokens": 512,  # REQUIRED: cap at 512 for scoring fairness
        "repetition_penalty": 1.05,
    }
# EVOLVE-BLOCK-END


# ==========================
# Metric helpers (FROZEN)
# ==========================
def _extract_answer(solution_str: str) -> str | None:
    matches = re.findall(r"<answer>(.*?)</answer>", solution_str, flags=re.DOTALL)
    if not matches:
        return None
    return matches[-1].strip()


def _validate_numbers(equation_str: str, available_numbers: List[int]) -> bool:
    try:
        used_numbers = [int(n) for n in re.findall(r"\d+", equation_str)]
    except Exception:
        return False
    return Counter(used_numbers) == Counter(list(available_numbers))


# Safe expression evaluation utilities
def tokenize(expr: str):
    s = expr.replace(" ", "")
    if re.search(r"[^0-9\.\+\-\*\/\(\)]", s):
        return None
    tokens = []
    i = 0
    L = len(s)
    while i < L:
        ch = s[i]
        if ch in "+-" and (i == 0 or s[i - 1] in "(*+-/"):
            if i + 1 < L and (s[i + 1].isdigit() or s[i + 1] == "."):
                j = i + 1
                while j < L and (s[j].isdigit() or s[j] == "."):
                    j += 1
                tokens.append(s[i:j])
                i = j
            elif i + 1 < L and s[i + 1] == "(":
                tokens.append("0")
                tokens.append(ch)
                i += 1
            else:
                return None
        elif ch.isdigit() or ch == ".":
            j = i
            while j < L and (s[j].isdigit() or s[j] == "."):
                j += 1
            tokens.append(s[i:j])
            i = j
        elif ch in "+-*/()":
            tokens.append(ch)
            i += 1
        else:
            return None
    return tokens


def to_postfix(tokens: List[str]):
    prec = {"+": 1, "-": 1, "*": 2, "/": 2}
    output = []
    ops = []
    for t in tokens:
        if re.fullmatch(r"[+-]?\d+\.\d*|[+-]?\.\d+|[+-]?\d+", t):
            output.append(t)
        elif t in prec:
            while ops and ops[-1] in prec and prec[ops[-1]] >= prec[t]:
                output.append(ops.pop())
            ops.append(t)
        elif t == "(":
            ops.append(t)
        elif t == ")":
            while ops and ops[-1] != "(":
                output.append(ops.pop())
            if not ops or ops[-1] != "(":
                return None
            ops.pop()
        else:
            return None
    while ops:
        if ops[-1] in ("(", ")"):
            return None
        output.append(ops.pop())
    return output


def eval_postfix(postfix: List[str]):
    stack = []
    for t in postfix:
        if re.fullmatch(r"[+-]?\d+\.\d*|[+-]?\.\d+|[+-]?\d+", t):
            stack.append(float(t))
        elif t in {"+", "-", "*", "/"}:
            if len(stack) < 2:
                return None
            b = stack.pop()
            a = stack.pop()
            if t == "+":
                stack.append(a + b)
            elif t == "-":
                stack.append(a - b)
            elif t == "*":
                stack.append(a * b)
            elif t == "/":
                if b == 0:
                    return None
                stack.append(a / b)
        else:
            return None
    if len(stack) != 1:
        return None
    return stack[0]


def _evaluate_equation(equation_str: str) -> float | None:
    tokens = tokenize(equation_str)
    if tokens is None:
        return None
    postfix = to_postfix(tokens)
    if postfix is None:
        return None
    return eval_postfix(postfix)


def _build_prompts_and_answers(split, max_tokens: int, tokenizer) -> Tuple[List[str], List[Dict[str, Any]]]:
    prompts: List[str] = []
    answers: List[Dict[str, Any]] = []
    for ex in split:
        prompt_body = TEMPLATE.format(numbers=ex["nums"], target=ex["target"])
        prompt = tokenizer.apply_chat_template(
            [
                dict(role="system", content="You are a helpful assistant."),
                dict(role="user", content=prompt_body),
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompts.append(prompt)
        answers.append({"target": ex["target"], "numbers": ex["nums"]})
    return prompts, answers


def run_countdown_eval(seed: int = 0, model_id: str | None = None, subset_size: int = 64) -> Dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)

    model_name = (
        model_id
        or os.environ.get("GRPO_EVAL_MODEL_ID")
        or "Qwen/Qwen2.5-0.5B-Instruct"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device).eval()

    # Dataset
    test_split = load_dataset("justinphan3110/Countdown-Tasks-3to4", split="test")
    # Subset for fast iteration
    sample_indices = list(range(min(subset_size, len(test_split))))
    sampled = [test_split[i] for i in sample_indices]

    prompts, answers = _build_prompts_and_answers(sampled, 512, tokenizer)

    gen_kwargs = init_generation_params()

    responses: List[str] = []
    output_token_lengths: List[int] = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        responses.append(text)
        output_token_lengths.append(len(tokenizer.encode(text)))

    # Scoring (frozen logic)
    rewards: List[float] = []
    count_correct = 0
    count_partial = 0
    count_failed = 0
    for resp, gt in zip(responses, answers):
        ans = _extract_answer(resp)
        if ans is None:
            rewards.append(0.0)
            count_failed += 1
            continue
        if not _validate_numbers(ans, gt["numbers"]):
            rewards.append(0.1)
            count_partial += 1
            continue
        val = _evaluate_equation(ans)
        if val is None:
            rewards.append(0.1)
            count_partial += 1
            continue
        if abs(float(val) - float(gt["target"])) < 1e-3:
            rewards.append(1.0)
            count_correct += 1
        else:
            rewards.append(0.1)
            count_partial += 1

    accuracy = (count_correct / len(responses)) * 100.0 if responses else 0.0
    mean_reward = float(torch.tensor(rewards).mean().item()) if rewards else 0.0
    avg_output_tokens = (
        sum(output_token_lengths) / len(output_token_lengths)
        if output_token_lengths
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "mean_reward": mean_reward,
        "avg_output_tokens": float(avg_output_tokens),
        "count_correct": count_correct,
        "count_partial": count_partial,
        "count_failed": count_failed,
        "subset_size": len(responses),
    }


