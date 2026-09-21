"""Streaming, read-only audit of Einher JSONL corpus/archive files.

This is deliberately dependency-free and does not load the JSONL files into
memory.  It emits an auditable JSON snapshot used by the phase-close report.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


METRICS = (
    "n_trades", "win_rate", "sharpe_ratio", "total_return", "profit_factor",
    "max_drawdown", "avg_net_return", "t_statistic", "p_value", "dsr",
)


def unwrap(row: dict) -> dict:
    return row.get("einher", row) if isinstance(row, dict) else {}


def at(row: dict, key: str, default="?"):
    e = unwrap(row)
    if key in e:
        return e.get(key, default)
    return row.get(key, default)


def universe(row: dict) -> dict:
    u = at(row, "universe", {})
    return u if isinstance(u, dict) else {}


def tree_features_and_atoms(tree) -> list[tuple[str, str, object]]:
    out = []
    def walk(node):
        if isinstance(node, dict):
            if "feature_ref" in node:
                out.append((str(node["feature_ref"]), str(node.get("operator", "?")), node.get("value")))
            for value in node.values():
                if isinstance(value, (dict, list)):
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(tree)
    return out


def tree_shape(node):
    """Logical/feature/operator shape, independent of thresholds and child order."""
    if not isinstance(node, dict):
        return "?"
    if "feature_ref" in node:
        return ("A", str(node["feature_ref"]), str(node.get("operator", "?")))
    op = str(node.get("operator", node.get("logical_operator", node.get("type", "?")))).upper()
    children = []
    for key in ("left", "right", "children", "conditions"):
        value = node.get(key)
        if isinstance(value, list):
            children.extend(tree_shape(v) for v in value)
        elif isinstance(value, dict):
            children.append(tree_shape(value))
    if op in {"AND", "OR"}:
        children.sort(key=repr)
    return (op, tuple(children))


def family(feature: str) -> str:
    lower = feature.lower()
    if lower.startswith("factor_"):
        return "factors"
    for prefix, name in (
        ("pattern_", "price_action"), ("signal_", "market_structure"),
        ("volume", "volume_flow"), ("obv", "volume_flow"), ("chaikin", "volume_flow"),
        ("atr", "volatility"), ("vol", "volatility"), ("rsi", "momentum"),
        ("stoch", "momentum"), ("roc", "momentum"), ("momentum", "momentum"),
        ("ema", "trend"), ("sma", "trend"), ("adx", "trend"),
        ("skew", "risk"), ("kurt", "risk"),
    ):
        if lower.startswith(prefix):
            return name
    return "other"


def safe_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def numeric_summary(values):
    if not values:
        return None
    values.sort()
    return {
        "n": len(values), "min": values[0], "p25": values[int(.25 * (len(values) - 1))],
        "median": statistics.median(values), "mean": statistics.fmean(values),
        "p75": values[int(.75 * (len(values) - 1))], "p95": values[int(.95 * (len(values) - 1))],
        "max": values[-1],
    }


def analyze(path: Path, corpus_shapes=None, corpus_feature_sets=None):
    counts = Counter()
    features, families, operators, shapes, exact_strategies = (Counter() for _ in range(5))
    assets, timeframes, horizons, directions, sources, reasons = (Counter() for _ in range(6))
    metrics = defaultdict(list)
    n_atoms, n_features = [], []
    malformed = 0
    same_as_corpus = near_corpus = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            e = unwrap(row)
            counts["records"] += 1
            u = universe(row)
            assets[str(u.get("asset", at(row, "asset", "?")))] += 1
            timeframes[str(u.get("timeframe", at(row, "timeframe", "?")))] += 1
            horizons[str(u.get("horizon", at(row, "horizon", "?")))] += 1
            directions[str(at(row, "direction", "?"))] += 1
            source = at(row, "source", {})
            sources[str(source.get("model", source.get("generator", "?")) if isinstance(source, dict) else source)] += 1
            reason = row.get("rejection_reason", row.get("reason"))
            if reason is not None:
                reasons[str(reason)] += 1
            atoms = tree_features_and_atoms(e.get("condition_tree", {}))
            fset = frozenset(a[0] for a in atoms)
            shape = repr(tree_shape(e.get("condition_tree", {})))
            exact = repr(sorted((a[0], a[1], repr(a[2])) for a in atoms))
            shapes[shape] += 1
            exact_strategies[exact] += 1
            n_atoms.append(len(atoms)); n_features.append(len(fset))
            for f, op, _ in atoms:
                features[f] += 1; families[family(f)] += 1; operators[op] += 1
            md = e.get("metrics", {})
            if not isinstance(md, dict):
                md = {}
            for key in METRICS:
                value = md.get(key)
                if safe_number(value):
                    metrics[key].append(float(value))
            if corpus_shapes is not None:
                if shape in corpus_shapes:
                    same_as_corpus += 1
                if fset and any(len(fset & fs) / len(fset | fs) >= .8 for fs in corpus_feature_sets):
                    near_corpus += 1
    duplicate_exact = sum(v - 1 for v in exact_strategies.values() if v > 1)
    duplicate_shapes = sum(v - 1 for v in shapes.values() if v > 1)
    return {
        "path": str(path), "records": counts["records"], "malformed": malformed,
        "distinct_exact_strategies": len(exact_strategies), "duplicate_exact_records": duplicate_exact,
        "distinct_structure_shapes": len(shapes), "duplicate_structure_records": duplicate_shapes,
        "assets": assets, "timeframes": timeframes, "horizons": horizons, "directions": directions,
        "sources": sources, "rejection_reasons": reasons, "features": features, "families": families,
        "operators": operators, "metrics": {k: numeric_summary(v) for k, v in metrics.items()},
        "conditions_per_strategy": numeric_summary(n_atoms), "unique_features_per_strategy": numeric_summary(n_features),
        "near_corpus": {"same_structure": same_as_corpus, "feature_jaccard_ge_0_8": near_corpus} if corpus_shapes is not None else None,
        "_shapes": set(shapes), "_feature_sets": set(frozenset(eval_shape_features(s)) for s in []),
    }, shapes, set()


def eval_shape_features(_):
    return []


def plain(value):
    if isinstance(value, Counter):
        return dict(value.most_common())
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items() if not k.startswith("_")}
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    corpus, corpus_shapes, _ = analyze(args.corpus)
    # Re-scan corpus to retain feature sets; still streaming and small memory.
    corpus_sets = set()
    with args.corpus.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                e = unwrap(json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue
            corpus_sets.add(frozenset(a[0] for a in tree_features_and_atoms(e.get("condition_tree", {}))))
    archive, _, _ = analyze(args.archive, corpus_shapes, corpus_sets)
    args.output.write_text(json.dumps({"corpus": plain(corpus), "archive": plain(archive)}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
