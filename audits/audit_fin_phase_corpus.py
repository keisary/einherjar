# Audit fin de phase — analyse quantitative corpus/archive Einherjar
# Usage: /d/midas_v2/midas/Scripts/python.exe audits/audit_fin_phase_corpus.py
import json, math
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path("D:/midas_v2/einherjar/audits/audit_fin_phase_2026-09-11.json")

FILES = {
    "corpus": "D:/midas_v2/einherjar/outputs/corpus.jsonl",
    "archive": "D:/midas_v2/einherjar/outputs/archive.jsonl",
    "corpus_1d": "D:/midas_v2/einherjar/outputs/corpus_1d_full.jsonl",
    "archive_1d": "D:/midas_v2/einherjar/outputs/archive_1d_full.jsonl",
    "corpus_1d_old": "D:/midas_v2/einherjar/outputs/corpus_1d.jsonl",
    "archive_1d_old": "D:/midas_v2/einherjar/outputs/archive_1d.jsonl",
}

def load(p):
    recs, malformed = [], 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                malformed += 1
    return recs, malformed

def einher(r):
    return r.get("einher", r) if isinstance(r, dict) else None

def tree_sig(node, prec=6):
    if node is None:
        return ("N",)
    if "feature_ref" in node:
        v = node.get("value")
        try:
            v = round(float(v), prec) if v is not None else None
        except Exception:
            v = None
        return ("L", node.get("feature_ref"), node.get("operator"), v)
    return ("B", node.get("op"), tree_sig(node.get("left"), prec), tree_sig(node.get("right"), prec))

def feats(node):
    if node is None:
        return []
    if "feature_ref" in node:
        return [node.get("feature_ref")]
    return feats(node.get("left")) + feats(node.get("right"))

def ncond(node):
    if node is None:
        return 0
    if "feature_ref" in node:
        return 1
    return ncond(node.get("left")) + ncond(node.get("right"))

def ops(node):
    if node is None:
        return []
    if "feature_ref" in node:
        return [node.get("operator")]
    return ops(node.get("left")) + ops(node.get("right"))

def stats(vals):
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    n = len(s)
    def pct(q):
        i = min(n - 1, int(q * n))
        return s[i]
    return {
        "n": n, "min": s[0], "p10": pct(0.10), "p25": pct(0.25),
        "med": pct(0.50), "p75": pct(0.75), "p90": pct(0.90), "max": s[-1],
        "mean": sum(s) / n,
    }

METRICS = ["n_trades","win_rate","sharpe_ratio","avg_net_return","total_return",
           "alpha","t_statistic","p_value","max_drawdown","profit_factor",
           "avg_holding_bars","tp_hit_rate","n_tp","n_sl","n_timeout"]

def family_map():
    fm = {}
    d = json.load(open("D:/midas_v2/einherjar/config/features_list.json", encoding="utf-8"))
    for k in ("indicator_columns","quant_columns","pattern_columns","factor_columns"):
        for f in d.get(k, []):
            fm[f] = k.replace("_columns","")
    return fm

FM = family_map()
def fams(fs):
    return sorted({FM.get(f, "other") for f in fs})

def analyze(name, recs):
    r = {"file": name, "records": len(recs)}
    ids = Counter()
    sigs = Counter()
    fset_groups = defaultdict(list)
    ass = Counter(); cls = Counter(); tf = Counter(); hz = Counter(); direc = Counter()
    models = Counter(); scopes = Counter(); fam_use = Counter(); feat_use = Counter()
    ncond_hist = Counter(); op_use = Counter()
    metrics = defaultdict(list)
    multi = Counter()
    by_tf_hz = Counter()
    rows = []
    for rec in recs:
        e = einher(rec)
        if not isinstance(e, dict):
            continue
        try:
            i = e.get("id", "?")
            ids[i] += 1
            ct = e.get("condition_tree")
            sigs[tree_sig(ct)] += 1
            fs = feats(ct)
            key = tuple(sorted(set(fs)))
            fset_groups[key].append(i)
            for f in fs:
                feat_use[f] += 1
            for f in set(fs):
                fam_use[FM.get(f, "other")] += 1
            nc = ncond(ct)
            ncond_hist[nc] += 1
            for o in ops(ct):
                op_use[o] += 1
            uni = e.get("universe", {})
            a = uni.get("asset"); tfv = uni.get("timeframe"); hzv = uni.get("horizon")
            ass[a] += 1; tf[tfv] += 1; hz[hzv] += 1; direc[e.get("direction")] += 1
            cls[uni.get("asset_class")] += 1
            by_tf_hz[(tfv, hzv)] += 1
            src = e.get("source", {})
            m = src.get("model")
            models[str(m)] += 1
            scopes[e.get("scope")] += 1
            if a == "multi" or (isinstance(a, str) and a.startswith("multi")):
                multi[e.get("scope")] += 1
            met = e.get("metrics", {})
            for k in METRICS:
                v = met.get(k)
                if isinstance(v, (int, float)):
                    metrics[k].append(v)
            rows.append((i, a, tfv, hzv, e.get("direction"), nc,
                         met.get("n_trades"), met.get("win_rate"),
                         met.get("sharpe_ratio"), met.get("alpha"),
                         met.get("t_statistic"), met.get("p_value"),
                         met.get("total_return"), e.get("scope")))
        except Exception:
            pass
    r["distinct_ids"] = len(ids)
    dup_ids = {k: v for k, v in ids.items() if v > 1}
    r["records_with_dup_id"] = sum(v for v in dup_ids.values())
    r["n_dup_id_groups"] = len(dup_ids)
    r["exact_structural_sigs"] = len(sigs)
    dup_sig = {k: v for k, v in sigs.items() if v > 1}
    r["records_in_dup_sig_groups"] = sum(v for v in dup_sig.values())
    r["n_dup_sig_groups"] = len(dup_sigs := dup_sig) 
    r["distinct_feature_sets"] = len(fset_groups)
    big = sorted(((len(v), k) for k, v in fset_groups.items()), reverse=True)[:12]
    r["feature_set_reuse_top"] = [("+".join(k)[:200], n) for n, k in big]
    r["assets"] = dict(ass.most_common(40))
    r["n_assets"] = len(ass)
    r["asset_classes"] = dict(cls.most_common())
    r["timeframes"] = dict(tf.most_common())
    r["horizons"] = dict(hz.most_common())
    r["directions"] = dict(direc.most_common())
    r["models"] = dict(models.most_common())
    r["scopes"] = dict(scopes.most_common())
    r["tf_x_horizon"] = [{"tf": k[0], "horizon": k[1], "n": v}
                         for k, v in by_tf_hz.most_common(40)]
    r["n_conditions_hist"] = dict(sorted(ncond_hist.items()))
    r["operators"] = dict(op_use.most_common())
    r["feature_families"] = dict(fam_use.most_common())
    r["top_features"] = dict(feat_use.most_common(30))
    r["n_distinct_features"] = len(feat_use)
    r["metrics"] = {k: stats(v) for k, v in metrics.items() if v}
    # flags
    sharpe = metrics.get("sharpe_ratio", [])
    r["n_sharpe_gt_15"] = sum(1 for v in sharpe if v > 15)
    r["n_sharpe_gt_5"] = sum(1 for v in sharpe if v > 5)
    wr = metrics.get("win_rate", [])
    r["n_win_rate_1"] = sum(1 for v in wr if v >= 1.0)
    nt = metrics.get("n_trades", [])
    r["n_lt5_trades"] = sum(1 for v in nt if v < 5)
    alpha = metrics.get("alpha", [])
    r["n_alpha_neg"] = sum(1 for v in alpha if v < 0)
    return r

def rej_reasons(recs):
    c = Counter()
    for rec in recs:
        rr = rec.get("rejection_reason") or rec.get("reason") or "?"
        # Normaliser le motif principal
        if "BH" in rr or "bonferroni" in rr.lower() or "multi-tests" in rr:
            key = "BH/multi-test"
        elif "trades" in rr.lower():
            key = "n_trades"
        elif "profit" in rr.lower() or "sharpe" in rr.lower():
            key = "profit/shape"
        elif "corr" in rr.lower():
            key = "corr"
        elif "time" in rr.lower():
            key = "temps"
        elif "holdout" in rr.lower():
            key = "holdout"
        elif "divers" in rr.lower() or "dupl" in rr.lower() or "jumeau" in rr.lower() or "twin" in rr.lower():
            key = "diversite/duplicata"
        else:
            key = "autre"
        c[(key, rr[:120])] += 1
    return c

def main():
    out = {}
    for name, p in FILES.items():
        recs, mal = load(p)
        out[name] = {"path": p, "malformed": mal}
        if recs:
            out[name]["analysis"] = analyze(name, recs)
            if "archive" in name:
                rc = rej_reasons(recs)
                out[name]["rejection_reasons"] = [
                    {"group": g, "detail": d, "count": n}
                    for (g, d), n in rc.most_common(60)
                ]
    out["_files_sizes"] = {name: Path(p).stat().st_size for name, p in FILES.items()}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("WROTE", OUT)

if __name__ == "__main__":
    main()