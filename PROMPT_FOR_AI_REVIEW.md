# Prompt for external AI review — Einherjar xgb_einhers

Copy-paste the block below into another AI assistant (Claude, GPT, Gemini, etc.) to get a critical review of the project.

---

## PROMPT START

You are a critical code reviewer and quantitative finance expert. I am building a trading strategy discovery system called **Einherjar / xgb_einhers**. I want your honest, **brutal** assessment — no politeness, no hype. If the project is overhyped or has serious methodological flaws, I want to know.

### What this system does

It takes historical OHLCV + 213 technical features (RSI, MACD, ATR, realized vol, kurtosis, skewness, risk factors, patterns, market structure) from MIDAS V3 compiled data, trains an XGBoost regressor on signed forward returns (Y_ret), extracts decision tree paths, converts each path into a trading rule (called an "Einher"), backtests it on OHLCV with ATR-based SL/TP, and admits it to a JSONL corpus if it passes criteria (min trades, sharpe, win rate, profit factor, max DD, family diversity, holdout trades).

### What's been built (since 2026-08-17)

**Code** (~3000 lines, 12 modules, all in `src/einherjar/research/xgb_einhers/`):
- `data_loader.py` — loads X (213 features), Y_dir/Y_ret/Y_hor, OHLCV, aligns by timestamp
- `label_engineer.py` — supervised target
- `model.py` — GBDT (xgboost primary, sklearn fallback) with `GBDTConfig.regularized()` (max_depth=3, min_child_weight=50, reg_alpha=1, reg_lambda=5)
- `path_extractor.py` — extracts tree paths as conditions
- `condition_tree.py` — AST of conditions (AND-only)
- `einher_builder.py` — builds Einher from path
- `backtester.py` — NEW backtester (replaces a buggy one), intrabar simulation, SL-first convention
- `admission.py` — multi-criteria: n_trades, sharpe, win_rate, profit_factor, max_DD, family diversity ≥ 2, min_holdout_trades
- `einher_io.py` — JSONL serialization
- `runner.py` — CLI supporting `--assets BTCUSD,ETHUSD,... --regularized --apply-dedup --drop-sparse --min-holdout-trades N`
- `feature_dedup.py` — drops correlated features (|r| > 0.85)
- `feature_filter.py` — drops sparse binary patterns (pct_True < 0.5%)
- `multi_asset_loader.py` — concatenates multiple assets for training

**Tests**: 81 tests, 0 failures, in `src/einherjar/research/tests/test_xgb_einhers/`

**Discovered Einhers (BTCUSD 1h, multi-horizon)**:
| Horizon | n | val sharpe | val WR | val/holdout ratio |
|---|---|---|---|---|
| 6h | 19 | 12.05 | 55.8% | **0.96** |
| 12h | 19 | 10.63 | 58.3% | 0.82 |
| 1d | 17 | 9.20 | 64.8% | 0.86 |
| 2d | 14 | 8.70 | 70.8% | 0.83 |

**Cross-asset test** (14 Einhers 2d → 4 other cryptos): **100% passing** (n_trades≥5, win_rate≥40%):
- ETH: WR 74%, sharpe 28, ~1191 trades/Einher
- LTC: WR 71%, sharpe 23, ~1045 trades/Einher
- ADA: WR 67%, sharpe 14, ~552 trades/Einher
- BCH: WR 73%, sharpe 18, ~574 trades/Einher

**Known issues I found and (claim to have) fixed**:
1. Sprint 2.1.4 — **Initial system was 100% overfit** (1/7 Einhers survived holdout). Fixed via regularization + multi-asset.
2. Sprint 2.2.4 — Found that 90+ patterns had < 0.5% True rate (useless for splits). Filtered them out.
3. Sprint 2.5.1 — **Critical bug**: val_metrics were actually full-dataset metrics (backtest was on entire X_aligned, not just the val split). Fixed by backtesting on [60%, 80%] of X_aligned. After fix, val/holdout ratio went from 0.30 → 0.88.
4. Sprint 2.6.1 — Found 100% of BTC Einhers generalize to ETH/LTC/ADA/BCH (cross-asset universality).

### Artifacts you can read

- Plan: `D:/midas_v2/Einherjar/audits/PLAN_XGBOOST_EINHER_AZ.md`
- Reports: `D:/midas_v2/Einherjar/audits/SPRINT_2_*.md` (6 reports)
- Code: `D:/midas_v2/Einherjar/src/einherjar/research/xgb_einhers/*.py`
- Tests: `D:/midas_v2/Einherjar/src/einherjar/research/tests/test_xgb_einhers/*.py`
- Outputs: `D:/midas_v2/Einherjar/outputs/einhers_*.jsonl`, `outputs/holdout_report_*.json`, `outputs/cross_asset_report_*.json`, `outputs/multi_horizon_report.json`
- Run command:
  ```bash
  $env:PYTHONPATH='src'
  & "D:/midas_v2/midas/Scripts/python.exe" -m einherjar.research.xgb_einhers.runner run \
    --asset BTCUSD --timeframe 1h --horizon 2d \
    --regularized --apply-dedup --drop-sparse --min-holdout-trades 5 \
    --output outputs/einhers.jsonl
  ```

### Specific questions I want you to answer

1. **Methodological soundness**: Is the overall approach (XGBoost → tree paths → trading rules) sound? What are the biggest methodological risks I might be missing?

2. **Look-ahead bias**: I split data 60/20/20 with embargo, train on first 60%, validate on next 20%, test on last 20% (holdout). After fixing the val=full bug (Sprint 2.5.1), I backtest on the val segment [60%, 80%] of X_aligned (not X_valid, but the OHLCV-aligned version). **Is there still any look-ahead risk in how I align X with OHLCV or in how I backtest?**

3. **Overfit vs real signal**: Sprint 2.6.1 shows 100% cross-asset passing on 4 different cryptos with win_rate 70%+. **Is this too good to be true?** What are the most likely explanations for spurious cross-asset universality?

4. **Statistical significance**: Each Einher has 5–50 trades on the holdout. **Is n=5 statistically significant enough to claim a real edge?** What minimum sample size would you require?

5. **Survivorship / selection bias**: I generate ~30 candidate paths per run, admit 14. With 4 horizons × 5 seeds × multiple asset combinations, I'm testing many hypotheses. **What's the multiple-comparison correction I should apply?**

6. **The features**: I use 213 features (mostly technical indicators). 86% of model importance goes to `risk` (8 features) and `statistical` (15 features) families. The 90+ candlestick patterns have < 0.5% positive rate and are essentially zero. **Does this suggest my "feature engineering" is actually just doing a glorified statistical arbitrage on volatility regimes?**

7. **Live trading readiness**: What would you need to see before deploying this to real money? What's missing?

8. **The honest verdict**: Is this a serious quantitative system, a research prototype, or a well-packaged form of overfitting? **Be direct.**

### What I want from you

- A point-by-point critique of the methodology.
- Identification of any specific code issues if you spot them in the files.
- A "red flags" section listing everything that worries you.
- An honest assessment of whether the cross-asset / multi-horizon results are plausible or suspicious.
- Specific improvements I should prioritize, ranked by expected impact.
- If you think this is fundamentally flawed, say so clearly and explain why.

Don't be polite. I'd rather hear "this is overfit garbage" now than lose money later.

## PROMPT END
