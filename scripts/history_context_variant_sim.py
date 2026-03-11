#!/usr/bin/env python3
"""Simulate mori history-context variant selection.

Standard behavior modeled here:
1. Each history turn pair prebuilds four variants: full/slight/heavy/none.
2. The builder chooses among those variants by configured weights, instead of
   using the old README-style "fit full -> compress -> drop" fallback chain.
3. Selection is still budget-aware: only variants that fit the remaining
   history budget are considered selectable.

The runtime budget is token-based, but historical text sizes are easier to
sample in chars, so this simulator converts chars to tokens with a simple
approximation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


ELLIPSIS_CHARS = 3


@dataclass(frozen=True)
class PairShape:
    user_chars: int
    assistant_chars: int

    @property
    def total_chars(self) -> int:
        return self.user_chars + self.assistant_chars


@dataclass(frozen=True)
class VariantCost:
    chars: int
    tokens: int


@dataclass
class SimConfig:
    runs: int = 50_000
    seed: int = 42
    history_pairs: int = 12
    min_pair_chars: int = 120
    max_pair_chars: int = 1_200
    input_token_budget: int = 12_000
    reserved_non_history_tokens: int = 10_800
    history_budget_tokens: Optional[int] = None
    chars_per_token: float = 4.0
    recency_decay: float = 0.90
    weight_full: float = 1.00
    weight_slight: float = 0.72
    weight_heavy: float = 0.40
    weight_none: float = 0.00
    ratio_slight: float = 0.65
    ratio_heavy: float = 0.30
    selection_mode: str = "weighted"

    def effective_history_budget_tokens(self) -> int:
        if self.history_budget_tokens is not None:
            return max(0, int(self.history_budget_tokens))
        return max(0, int(self.input_token_budget) - int(self.reserved_non_history_tokens))


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def estimate_tokens(char_count: int, cfg: SimConfig) -> int:
    if char_count <= 0:
        return 0
    return max(1, int(math.ceil(char_count / max(0.1, cfg.chars_per_token))))


def split_pair_chars(total_chars: int, rng: random.Random) -> PairShape:
    user_ratio = rng.uniform(0.35, 0.55)
    user_chars = max(1, int(total_chars * user_ratio))
    assistant_chars = max(1, total_chars - user_chars)
    return PairShape(user_chars=user_chars, assistant_chars=assistant_chars)


def compress_side(original_chars: int, max_chars: int) -> int:
    if original_chars <= 0 or max_chars <= 0:
        return 0
    kept = min(original_chars, max_chars)
    if kept < original_chars:
        kept += ELLIPSIS_CHARS
    return kept


def build_variants(pair: PairShape, cfg: SimConfig) -> Dict[str, VariantCost]:
    full_chars = pair.total_chars

    def compressed_variant(ratio: float) -> VariantCost:
        ratio = clamp(ratio, 0.05, 0.95)
        budget = max(40, int(math.floor(pair.total_chars * ratio)))
        user_budget = max(20, int(math.floor(budget * 0.45)))
        assistant_budget = max(20, budget - user_budget)
        chars = compress_side(pair.user_chars, user_budget) + compress_side(pair.assistant_chars, assistant_budget)
        return VariantCost(chars=chars, tokens=estimate_tokens(chars, cfg))

    variants = {
        "full": VariantCost(chars=full_chars, tokens=estimate_tokens(full_chars, cfg)),
        "slight": compressed_variant(cfg.ratio_slight),
        "heavy": compressed_variant(cfg.ratio_heavy),
        "none": VariantCost(chars=0, tokens=0),
    }
    return variants


def ranked_weights(recency_factor: float, cfg: SimConfig) -> List[Tuple[str, float]]:
    return [
        ("full", max(0.0, cfg.weight_full * recency_factor)),
        ("slight", max(0.0, cfg.weight_slight * recency_factor)),
        ("heavy", max(0.0, cfg.weight_heavy * recency_factor)),
        ("none", max(0.0, cfg.weight_none)),
    ]


def pick_variant(
    variants: Dict[str, VariantCost],
    remaining_tokens: int,
    recency_factor: float,
    cfg: SimConfig,
    rng: random.Random,
) -> Tuple[str, str]:
    weighted = [
        (name, weight)
        for name, weight in ranked_weights(recency_factor, cfg)
        if variants[name].tokens <= remaining_tokens
    ]
    if not weighted:
        return "none", "forced_budget"

    if cfg.selection_mode == "ranked":
        weighted.sort(
            key=lambda item: (
                -item[1],
                {"full": 0, "slight": 1, "heavy": 2, "none": 3}.get(item[0], 99),
            )
        )
        return weighted[0][0], "ranked"

    total = sum(weight for _, weight in weighted)
    if total <= 1e-9:
        for name in ("full", "slight", "heavy", "none"):
            if variants[name].tokens <= remaining_tokens:
                return name, "zero_weight_fallback"
        return "none", "forced_budget"

    pick = rng.random() * total
    acc = 0.0
    for name, weight in weighted:
        acc += weight
        if pick <= acc:
            if name == "none":
                return name, "sampled_none"
            return name, "weighted"
    tail = weighted[-1][0]
    if tail == "none":
        return tail, "sampled_none"
    return tail, "weighted"


def run_once(cfg: SimConfig, rng: random.Random) -> Dict[str, float]:
    budget_tokens = cfg.effective_history_budget_tokens()
    used_tokens = 0
    used_chars = 0
    selected_counts = {"full": 0, "slight": 0, "heavy": 0, "none": 0}
    reason_counts = {
        "weighted": 0,
        "ranked": 0,
        "sampled_none": 0,
        "forced_budget": 0,
        "zero_weight_fallback": 0,
    }

    for idx in range(cfg.history_pairs, 0, -1):
        total_chars = rng.randint(cfg.min_pair_chars, cfg.max_pair_chars)
        pair = split_pair_chars(total_chars, rng)
        variants = build_variants(pair, cfg)

        recency_index = cfg.history_pairs - idx
        recency_factor = clamp(cfg.recency_decay, 0.01, 1.0) ** recency_index
        remaining_tokens = max(0, budget_tokens - used_tokens)
        choice, reason = pick_variant(variants, remaining_tokens, recency_factor, cfg, rng)

        selected_counts[choice] += 1
        reason_counts[reason] += 1
        used_tokens += variants[choice].tokens
        used_chars += variants[choice].chars

    kept_pairs = selected_counts["full"] + selected_counts["slight"] + selected_counts["heavy"]
    return {
        "used_tokens": used_tokens,
        "used_chars": used_chars,
        "kept_pairs": kept_pairs,
        "full": selected_counts["full"],
        "slight": selected_counts["slight"],
        "heavy": selected_counts["heavy"],
        "none": selected_counts["none"],
        "weighted": reason_counts["weighted"],
        "ranked": reason_counts["ranked"],
        "sampled_none": reason_counts["sampled_none"],
        "forced_budget": reason_counts["forced_budget"],
        "zero_weight_fallback": reason_counts["zero_weight_fallback"],
        "budget_tokens": budget_tokens,
    }


def run_sim(cfg: SimConfig) -> Dict[str, float]:
    rng = random.Random(cfg.seed)
    totals = {
        "used_tokens": 0,
        "used_chars": 0,
        "kept_pairs": 0,
        "full": 0,
        "slight": 0,
        "heavy": 0,
        "none": 0,
        "weighted": 0,
        "ranked": 0,
        "sampled_none": 0,
        "forced_budget": 0,
        "zero_weight_fallback": 0,
    }

    budget_tokens = cfg.effective_history_budget_tokens()
    for _ in range(cfg.runs):
        row = run_once(cfg, rng)
        for key in totals:
            totals[key] += int(row[key])

    denom_runs = float(cfg.runs)
    denom_pairs = float(cfg.runs * cfg.history_pairs)
    avg_used_tokens = totals["used_tokens"] / denom_runs
    avg_used_chars = totals["used_chars"] / denom_runs
    fill_ratio = 0.0
    if budget_tokens > 0:
        fill_ratio = avg_used_tokens / float(budget_tokens)

    return {
        "runs": cfg.runs,
        "seed": cfg.seed,
        "selection_mode": cfg.selection_mode,
        "history_pairs_per_run": cfg.history_pairs,
        "input_token_budget": cfg.input_token_budget,
        "reserved_non_history_tokens": cfg.reserved_non_history_tokens,
        "effective_history_budget_tokens": budget_tokens,
        "avg_history_tokens_used": avg_used_tokens,
        "avg_history_chars_used": avg_used_chars,
        "avg_history_budget_fill_ratio": fill_ratio,
        "avg_kept_pairs": totals["kept_pairs"] / denom_runs,
        "full_ratio": totals["full"] / denom_pairs,
        "slight_ratio": totals["slight"] / denom_pairs,
        "heavy_ratio": totals["heavy"] / denom_pairs,
        "none_ratio": totals["none"] / denom_pairs,
        "forced_budget_none_ratio": totals["forced_budget"] / denom_pairs,
        "sampled_none_ratio": totals["sampled_none"] / denom_pairs,
        "weighted_pick_ratio": totals["weighted"] / denom_pairs,
        "ranked_pick_ratio": totals["ranked"] / denom_pairs,
        "zero_weight_fallback_ratio": totals["zero_weight_fallback"] / denom_pairs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate mori history variant selection with weighted multi-variant retention"
    )
    parser.add_argument("--runs", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history-pairs", type=int, default=12)
    parser.add_argument("--min-pair-chars", type=int, default=120)
    parser.add_argument("--max-pair-chars", type=int, default=1_200)
    parser.add_argument("--input-token-budget", type=int, default=12_000)
    parser.add_argument("--reserved-non-history-tokens", type=int, default=10_800)
    parser.add_argument("--history-budget-tokens", type=int, default=None)
    parser.add_argument("--chars-per-token", type=float, default=4.0)
    parser.add_argument("--recency-decay", type=float, default=0.90)
    parser.add_argument("--weight-full", type=float, default=1.00)
    parser.add_argument("--weight-slight", type=float, default=0.72)
    parser.add_argument("--weight-heavy", type=float, default=0.40)
    parser.add_argument("--weight-none", type=float, default=0.00)
    parser.add_argument("--ratio-slight", type=float, default=0.65)
    parser.add_argument("--ratio-heavy", type=float, default=0.30)
    parser.add_argument(
        "--selection-mode",
        choices=("weighted", "ranked"),
        default="weighted",
        help="weighted = sample by variant weights; ranked = always pick max-score feasible variant",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final summary as JSON instead of plain text",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SimConfig:
    return SimConfig(
        runs=max(1, int(args.runs)),
        seed=int(args.seed),
        history_pairs=max(1, int(args.history_pairs)),
        min_pair_chars=max(1, int(args.min_pair_chars)),
        max_pair_chars=max(int(args.min_pair_chars), int(args.max_pair_chars)),
        input_token_budget=max(1, int(args.input_token_budget)),
        reserved_non_history_tokens=max(0, int(args.reserved_non_history_tokens)),
        history_budget_tokens=None if args.history_budget_tokens is None else max(0, int(args.history_budget_tokens)),
        chars_per_token=max(0.1, float(args.chars_per_token)),
        recency_decay=clamp(float(args.recency_decay), 0.01, 1.0),
        weight_full=max(0.0, float(args.weight_full)),
        weight_slight=max(0.0, float(args.weight_slight)),
        weight_heavy=max(0.0, float(args.weight_heavy)),
        weight_none=max(0.0, float(args.weight_none)),
        ratio_slight=clamp(float(args.ratio_slight), 0.05, 0.95),
        ratio_heavy=clamp(float(args.ratio_heavy), 0.05, 0.95),
        selection_mode=str(args.selection_mode),
    )


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    summary = run_sim(cfg)

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    print("=== mori History Variant Simulation ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
