# CLAUDE
Got the repo. Let me extract and read the key modules — data loading/alignment, backtester, and admission — since those are where look-ahead bias and the cross-asset "too good to be true" results would actually live.J'ai été chercher le repo réel (`keisary/einherjar`) et lu le code source des modules concernés au lieu de me fier uniquement à ton prompt. Verdict direct : **ce système a un bug de métrique qui explique à lui seul une bonne partie des résultats "trop beaux pour être vrais"**, et il manque la correction multi-tests que tu appliques pourtant ailleurs (discovery_engine.py). Détail ci-dessous, point par point.

## Le red flag n°1 — ton "Sharpe" n'est pas un Sharpe

Dans `backtester.py::compute_metrics` :

```python
std = float(np.std(rets, ddof=1)) if n > 1 else 0.0
sharpe = float(avg_net / std * np.sqrt(max(n, 1)))
```

Le commentaire dit *"Sharpe annualisé (approx : sqrt(n_trades) sur la base trades/an)"* — mais ce n'est **pas** une annualisation, c'est la formule d'une **statistique de Student (t-stat)** appliquée à la moyenne des rendements. `sqrt(n_trades)` n'a rien à voir avec `sqrt(périodes par an)`. Conséquence directe et vérifiable :

- BTC 2d : ~14-19 Einhers, sharpe ~8-12
- ETH cross-asset : ~1191 trades/Einher, sharpe 28
- LTC : ~1045 trades/Einher, sharpe 23
- ADA/BCH : ~550-570 trades, sharpe 14-18

Le ratio entre ces "sharpes" suit presque exactement `sqrt(n_trades_ETH / n_trades_BTC) ≈ sqrt(1191/19) ≈ 7.9`. Autrement dit : **plus un Einher génère de trades sur un actif liquide, plus son "sharpe" explose mécaniquement**, indépendamment de la qualité réelle du edge. C'est exactement le pattern que tu observes (les actifs avec le plus de trades ont les scores les plus hauts), et ça répond directement à tes questions 3, 6 et 8 : non, ce n'est pas de l'universalité cross-asset miraculeuse, c'est un artefact de formule qui gonfle le score avec le volume de signaux.

Il faut corriger en deux temps :
1. Un vrai Sharpe annualisé = `(moyenne_rendement / std_rendement) * sqrt(trades_par_an)`, où `trades_par_an` est estimé depuis la fréquence réelle des signaux, pas depuis `n`.
2. Séparer "significativité statistique du edge" (t-stat, avec n) de "qualité risque/rendement" (Sharpe, avec fréquence). Ce sont deux métriques différentes que ton code confond.

## Le red flag n°2 — pas de correction multi-tests dans ce module

`admission.py` filtre sur `n_trades`, `sharpe`, `win_rate`, `profit_factor`, `max_drawdown`, diversité de familles, et (optionnellement) `min_holdout_trades`. **Aucune trace de Benjamini-Hochberg ou d'un ajustement quelconque du seuil selon le nombre d'hypothèses testées** — contrairement à ce que tu as fait dans `discovery_engine.py` pour un autre module. Tu génères ~30 candidats par run × 4 horizons × plusieurs assets × plusieurs seeds : c'est un nombre d'hypothèses conséquent, et avec `min_sharpe=0.3` (déjà un seuil bas, et rendu encore plus permissif par le bug ci-dessus), tu vas laisser passer des faux positifs par pur hasard multiple-testing.

## Sur tes 8 questions

**Q1 — Solidité méthodologique.** L'architecture (arbre → règles → backtest → admission) est saine dans son principe. Le point d'entrée à `OPEN[t+1]` après un signal calculé sur la bougie `t` close est correct et évite le look-ahead intrabar évident. Le vrai risque n'est pas l'architecture, c'est la métrique de sélection (voir red flag n°1) et l'absence de garde-fou statistique (red flag n°2).

**Q2 — Look-ahead.** Le join `align_xy_with_ohlcv` est un inner join propre sur timestamp, rien de suspect. Par contre l'embargo est **fixe à 50 bougies** (`temporal_split`, `data_loader.py`) quel que soit l'horizon de la target. Pour un horizon 2d en 1h (48 bougies), c'est *juste* suffisant — mais c'est du pile ou face, pas une marge de sécurité. Si tu ajoutes un horizon plus long (3d, 1w), l'embargo actuel ne suivra plus et tu auras un vrai leakage entre train et val (le label d'un point proche de la frontière train utilise des bougies futures qui tombent dans la fenêtre val). Corrige en fixant `embargo_bars = max(embargo_min, horizon_bars)`.

**Q3/Q6/Q8 — Cross-asset "trop beau" / vrai signal vs bruit.** Comme démontré ci-dessus, c'est en grande partie un artefact de métrique. Une fois corrigée, il faut relancer *exactement* le même pipeline et revoir les chiffres — je m'attends à une chute significative des "sharpes" annoncés, surtout sur les actifs à fort volume de trades. Sur le fait que 86% de l'importance vienne des familles `risk`/`statistical` (volatilité) : oui, c'est cohérent avec un système qui capture surtout un régime de volatilité (mean-reversion ou momentum conditionné par ATR/vol réalisée), pas des patterns de chandeliers. Ce n'est pas forcément mauvais — la prime de risque de volatilité est un edge réel et documenté en finance — mais ce n'est pas ce que "213 features de pattern recognition" laisse penser. Sois honnête dans ta documentation : c'est un système de régime de volatilité, pas un système multi-facteurs.

**Q4 — Significativité (n holdout).** 5 à 50 trades, c'est **insuffisant** pour conclure à un edge réel, surtout avec la formule de Sharpe buguée qui gonfle artificiellement la confiance. Avec des rendements de trading bruités, tu veux au minimum 100-200 trades holdout indépendants pour qu'un test de moyenne (t-test, ou mieux : bootstrap sur les rendements) ait une puissance correcte. En dessous de 30, ne fais confiance à rien — le `min_trades=30` en admission est un minimum absolu, pas un seuil de confiance.

**Q5 — Correction multi-comparaisons.** Applique Benjamini-Hochberg (comme dans `discovery_engine.py`) sur l'ensemble des candidats testés à travers horizons × assets × seeds dans un même run de recherche, avant l'étape d'admission finale. Sans ça, ton taux de faux positifs augmente avec chaque nouvelle combinaison que tu testes.

**Q7 — Prêt pour le live ?** Non, pas en l'état. Avant d'y penser : (1) corrige le Sharpe, (2) ajoute BH, (3) refais tourner holdout avec au moins 100+ trades par Einher retenu, (4) walk-forward out-of-sample sur une période complètement nouvelle (pas juste holdout statique — un vrai test séquentiel), (5) paper trading en live pendant plusieurs semaines avant tout capital réel.

## Priorités classées par impact

1. **Corriger la formule de Sharpe** (backtester.py) — sans ça, toutes tes métriques de sélection sont biaisées.
2. **Ajouter BH ou équivalent** dans admission.py.
3. **Embargo proportionnel à l'horizon**, pas fixe à 50.
4. **Relever le seuil `min_holdout_trades`** à 100+ pour toute conclusion de "edge validé".
5. Relancer tous les runs sprint 2.5/2.6 avec ces corrections et comparer — c'est le vrai test de vérité.

Tu n'es pas dans l'erreur sur la structure générale du pipeline (elle est plutôt bien pensée, avec embargo, holdout séparé, diversité de familles). L'erreur est précise et localisée : une ligne de code dans le calcul du Sharpe qui invalide la comparabilité de tes scores entre Einhers ayant des fréquences de trade différentes — et c'est probablement la cause principale de l'"universalité cross-asset" qui t'a semblé suspecte à raison.

# GEMINI
### The Executive Verdict: Brutal Summary

**Verdict: Research Prototype suffering from structural metric inflation and severe beta-leakage.**

Your system is not "ready for production," and the metrics you are observing (**Val Sharpe 8.7–12.0, Cross-Asset Sharpe 14–28, Win Rate 70%+**) are **empirically impossible** in liquid crypto markets. 

To put this in perspective: Jim Simons’ Renaissance Medallion Fund achieved a net Sharpe of ~3.0–4.0 with billions in infrastructure and world-class execution. If a 1-hour XGBoost path-extraction model truly produced a Sharpe of 28 across 1,191 trades on ETH, you would own the entire crypto market cap within 18 months.

The fact that you see these numbers does **not** prove you found an invincible alpha; it proves that **there are structural calculation errors, execution timing leaks, or target/time-series leakage in your pipeline.**

---

### Red Flags: The Immediate Warning Signs

1. **Sharpe Ratios of 14 to 28 on Cross-Asset Tests**:
   This is the single biggest statistical red flag. Sharpe ratios above 4.0 on hourly crypto data are 99.9% of the time caused by **calculating Sharpe on raw trade returns instead of a continuous time-series equity curve**, or **mis-annualizing sparse trade returns** (e.g., multiplying per-trade Sharpe by $\sqrt{8760}$ as if trades happen every single hour of the year without idle time).
2. **Exploding Trade Counts (14 to 1,191)**:
   On BTC holdout, each Einher executes 5–50 trades. When tested on ETH/LTC, the trade count mysteriously explodes to **1,045–1,191 trades per Einher**. This indicates your cross-asset test was either run on un-sliced, un-embargoed full history, or the rules are firing indiscriminately on different baseline volatility regimes.
3. **Statistical Insignificance ($n=5$ holdout trades)**:
   Admitting rules based on $\ge 5$ trades on holdout has zero statistical power. A coin flipped 5 times has an 18.75% chance of landing heads $\ge 4$ times ($80\%$ win rate) purely by chance. When screening dozens of candidate paths, multiple rules will pass this threshold by pure variance.
4. **86% Feature Importance on Risk & Volatility**:
   The model is not predicting direction from technical signals; it is isolating broad volatility regimes (e.g., bull market expansions vs. low-vol consolidation) and acting as a directional beta proxy.

---

### Point-by-Point Answers to Your 8 Questions

```
                                  PIPELINE RISK MAP
                                  
  [ 213 MIDAS Features ] ──► [ XGBoost (Y_ret) ] ──► [ Path Extraction ]
          │                         │                        │
  ⚠️ Leakage Check:          ⚠️ Target Leakage:       ⚠️ Selection Bias:
   Rolling Lookback?          Purge & Embargo?         30 paths x 4 horiz x 5 seeds
   Future Data in Norm?       Overlapping Y_ret?       = 600 tests (DSR needed)
                                                             │
  [ JSONL Corpus ] ◄─────── [ Admission Gate ] ◄──── [ Backtester (1h OHLCV) ]
                                    │                        │
                             ⚠️ Stat Significance:   ⚠️ Execution Reality:
                              n=5 trades is noise      - Next-bar Open vs Close?
                              Sharpe 14-28 inflated    - High/Low bar ambiguity
                                                       - Slippage & fees (0.05-0.10%)
```

---

#### 1. Methodological Soundness (XGBoost $\to$ Tree Paths $\to$ Trading Rules)

**Is the approach sound?**
In theory, extracting decision paths from tree ensembles (RuleFit / Decision List extraction) is a known technique to convert non-linear models into interpretable, rule-based systems. **However, using it this way introduces severe methodological traps:**

* **Loss of Ensemble Regularization**: XGBoost achieves its generalization by averaging hundreds of shallow trees with shrinkage (learning rate $\eta \approx 0.01\text{--}0.1$). When you isolate **one single path** from **one tree**, you strip away the entire ensemble voting mechanism and the variance reduction of bagging/boosting. You are left with an unregularized, brittle rectangular hyper-box in 213-dimensional space.
* **AND-only AST Pitfall**: Decision trees partition continuous space into orthogonal step functions ($x_1 > c_1 \land x_2 \le c_2$). Financial markets rarely have rigid orthogonal boundaries. A rule like $\text{ATR} > 1.452 \land \text{RSI} < 42.1$ is hyper-sensitive to threshold drift as market volatility regimes shift over time.

---

#### 2. Look-Ahead Bias & Alignment Hazards

Even after fixing the Sprint 2.5.1 bug (evaluating val on full data), several high-probability look-ahead leaks remain:

1. **Feature Engineering Pre-Computation**:
   * Were the 213 MIDAS V3 features computed **before** the 60/20/20 train/val/holdout split?
   * If any feature uses centering, rolling normalization ($z$-score), EMA/SMA with warm-up periods, quantile cuts, or ATR calculations that spanned across the split boundaries without strict causal windowing, you have feature leakage.
2. **Target Horizon Overlap (Purging & Embargoing)**:
   * You train on forward returns $Y\_ret$ with horizons up to **2 days (48 hours)**.
   * If your train set ends at index $T$, the label for bar $T-47$ uses prices all the way up to bar $T+1$.
   * If you do not **purge** the last 48 bars of the training set and **embargo** the first 48 bars of the validation set (as formalized by Marcos López de Prado in *Advances in Financial Machine Learning* [^1]), the model learns labels that contain prices from the validation period.
3. **Backtest Intrabar Execution Timing**:
   * If your condition evaluates on bar $t$'s metrics (which require bar $t$'s `Close`, `High`, `Low`, `Volume`), **at what exact timestamp and price does the backtester enter?**
   * *The Classic Bug*: Evaluating conditions using bar $t$ data, but executing the entry at bar $t$'s `Open` or `Close` while allowing bar $t$'s `High`/`Low` to trigger TP/SL within the same bar.
   * *Correct Execution*: If a condition is met at bar $t$'s `Close`, the earliest possible entry is bar $t+1$'s `Open`, and TP/SL can only be triggered by prices from bar $t+1$ onward.

---

#### 3. Overfit vs. Real Signal (The Sharpe 14–28 Reality Check)

**Is this too good to be true? Yes, 100%.**

There are three primary reasons why your cross-asset test produced 70%+ win rates and astronomical Sharpe ratios:

1. **The Trade-Based Sharpe Ratio Fallacy**:
   If your backtester calculates Sharpe using trade PnLs instead of continuous bar-by-bar portfolio returns:
   $$\text{Sharpe}_{\text{trade}} = \frac{\text{mean}(R_{\text{trade}})}{\text{std}(R_{\text{trade}})} \times \sqrt{N_{\text{annual\_trades}}}$$
   If you have a tight TP/SL ratio where 7 out of 10 trades hit TP, the standard deviation of trade returns will be small, and multiplying by $\sqrt{8760}$ or $\sqrt{1000}$ explodes the Sharpe to 20+. 
   *True Sharpe must be calculated on the continuous hourly return of the portfolio (including all the zero-return hours when in cash).*
2. **Single-Factor Market Beta Disguised as Multi-Asset Universality**:
   BTC, ETH, LTC, ADA, and BCH have cross-asset return correlations between $0.80$ and $0.95$. They are not 5 independent bets; they are 5 leveraged expressions of the **same underlying crypto market factor**. If your Einher learned to buy when broad market momentum is positive and volatility is non-explosive, it will show high win rates across all 5 coins during general market uptrends.
3. **No Transaction Costs, Slippage, or Funding Rates**:
   On 1,191 trades per asset, typical exchange taker fees (0.05% per side = 0.10% round-trip) plus slippage (0.02%–0.05%) will cost **1.2% to 1.5% of total portfolio equity per round-trip**. With 1,000 trades, friction alone will consume **120%–150% of the account balance**.

---

#### 4. Statistical Significance of Sample Size ($n=5$ to $50$)

**$n=5$ is completely statistically meaningless.**

Let us evaluate the standard error ($\text{SE}$) of a win rate $p = 0.70$:
$$\text{SE}(p) = \sqrt{\frac{p(1-p)}{n}}$$

| Sample Size ($n$) | Standard Error | 95% Confidence Interval for 70% Win Rate | Statistically Significant Edge? |
|---|---|---|---|
| **5** | $\pm 20.5\%$ | $[29.8\%, 100.0\%]$ | **No (Indistinguishable from pure noise)** |
| **20** | $\pm 10.2\%$ | $[49.9\%, 90.1\%]$ | **No (Includes 50% break-even)** |
| **50** | $\pm 6.5\%$ | $[57.3\%, 82.7\%]$ | **Weak / Marginal** |
| **200+** | $\pm 3.2\%$ | $[63.6\%, 76.4\%]$ | **Minimum threshold for production confidence** |

*Rule of Thumb*: In quantitative finance, you need at least **$n \ge 100$ independent, non-overlapping trades** on completely untouched out-of-sample data before a Sharpe $> 1.5$ can be distinguished from random walk noise at the $\alpha = 0.05$ significance level.

---

#### 5. Multiple Testing & Selection Bias

You generate ~30 candidate paths per run $\times$ 4 horizons $\times 5$ seeds $\approx$ **600 tested hypotheses**.

When you test 600 hypotheses at standard $\alpha = 0.05$, you are mathematically guaranteed to find:
$$600 \times 0.05 = 30 \text{ false discoveries (flukes with high Sharpe and high WR)}$$

You then filter down and pick the top 14 candidates. **This is textbook selection bias (data mining / $p$-hacking).**

##### Required Corrections:
1. **Deflated Sharpe Ratio (DSR)**:
   You must compute the Deflated Sharpe Ratio (Bailey & López de Prado, 2014) [^2], which discounts your observed Sharpe $S$ based on the number of trials $N$ and the variance of the tested strategies:
   $$E[\max_{N}(S)] \approx (1 - \gamma) Z^{-1}\left(1 - \frac{1}{N}\right) + \gamma Z^{-1}\left(1 - \frac{1}{N e}\right)$$
   For $N = 600$ tests, the expected maximum Sharpe under pure white noise is **$S_{\text{null}} \approx 3.2\text{--}3.5$**. Any observed Sharpe below that threshold after 600 trials has a high probability of being a false discovery.
2. **Family-Wise Error Rate (FWER) / False Discovery Rate (FDR)**:
   Apply the **Benjamini-Hochberg procedure** across all extracted candidate paths to cap the FDR at $5\%$.

---

#### 6. Feature Dynamics: Volatility Regime vs. True Alpha

**Does this suggest your system is doing a glorified volatility regime filter?**

**Yes, unequivocally.**

* Candlestick patterns have near-zero importance and $< 0.5\%$ frequency because single-bar candle patterns in 1h crypto are non-stationary noise.
* 86% of feature importance resting in `risk` and `statistical` (realized vol, ATR, kurtosis, skewness) proves that the XGBoost trees are partitioning the data into:
  $$\text{Regime} = \begin{cases} \text{High Volatility / Trend Expansion} & \implies \text{Long Trades Triggered} \\ \text{Low Volatility / Choppy Mean-Reversion} & \implies \text{No Trade} \end{cases}$$

This means you have built a **volatility regime gate**, not an alpha predictor. When volatility is in a specific band during a historical crypto bull cycle, buying and setting ATR-based TP/SL naturally generates a high win rate because the market had an upward drift during those exact volatility bands.

---

#### 7. Live Trading Readiness: What Is Missing?

If you connect this system to a live exchange tomorrow, it will almost certainly degrade rapidly due to missing execution realities:

```
                               LIVE EXECUTION GAP
                               
   Backtest Assumption                         Live Market Reality
  ─────────────────────                       ─────────────────────
  Fill at Close/Open         ──► Slippage:    Market orders cross spread (-2 to -5 bps)
  Zero Cost Execution        ──► Fees:        Taker fee (5-10 bps per side = 10-20 bps round trip)
  Instantaneous Fills        ──► Latency:     Signal lag on 1h bar close (100ms - 2s)
  Symmetric Execution        ──► Skewed Liq:  Stop-losses suffer worse fills than take-profits
  Endless Capacity           ──► Market Imp:  Large sizes move thin order books
  Spot Hold = Margin Hold    ──► Funding:     Perpetual swaps charge 8h funding (can bleed 20-50% APR)
```

1. **Transaction Fees & Slippage**: Deduct at least **0.08%–0.12% per round-trip trade** (taker fee + spread + adverse execution). Re-run your backtest; watch how many 6h/12h Einhers turn negative.
2. **Perpetual Swap Funding Rates**: If holding longs for 2 days during bull runs, funding rates on Binance/Bybit can easily cost 0.03%–0.10% per day.
3. **Execution Latency & Price Slippage on Stop Losses**: In fast-moving crypto markets, stop-losses are executed via market orders and almost never fill at the exact ATR stop price. You must model 2–3 ticks of adverse slippage on all SL triggers.

---

#### 8. The Honest Verdict

**Category: Research Prototype with severe structural over-optimism.**

* **The Good**: The modular architecture (AST condition trees, JSONL serialization, multi-asset loader, GBDT fallback) is clean, structured, and well-engineered. You clearly have strong software development capabilities.
* **The Fatal Flaw**: The quantitative evaluation framework is broken. The reported Sharpe ratios (8.7 to 28) are clear mathematical symptoms of trade-level Sharpe miscalculation, intrabar execution look-ahead, or un-purged target leakage.
* **The Reality**: If you deployed this live today, you would likely suffer substantial drawdowns from exchange fees, regime shifts, and curve-fitting decay.

---

### Ranked Action Items for Immediate Remediation

| Priority | Action Item | Expected Impact | How to Implement |
|---|---|---|---|
| **P0 (Critical)** | **Fix Sharpe Calculation** | Fixes inflated numbers | Replace per-trade Sharpe with continuous hourly equity-curve Sharpe: $R_t = \frac{Equity_t - Equity_{t-1}}{Equity_{t-1}}$. Annualize with $\sqrt{8760}$ on **continuous portfolio returns**, not trade returns. |
| **P0 (Critical)** | **Audit Intrabar Execution Timing** | Eliminates look-ahead leak | Ensure condition met at Bar $t$ Close strictly executes at **Bar $t+1$ Open**. No High/Low checks on Bar $t$. |
| **P1 (High)** | **Implement Purged & Embargoed Cross-Validation** | Prevents horizon leakage | For 2-day ($48\text{h}$) horizons, purge 48 bars before split boundaries and embargo 48 bars after. |
| **P1 (High)** | **Add Real-World Frictions** | Filters unviable alphas | Deduct **10 bps (0.10%) round-trip fee** + **3 bps slippage** on every trade in backtester. |
| **P2 (Medium)** | **Raise Admission Trade Threshold** | Eliminates low-$n$ noise | Require a minimum of **$n \ge 100$ holdout trades** before admitting an Einher. Reject any rule with $n < 50$. |
| **P2 (Medium)** | **Apply Multiple Hypothesis Discount (DSR)** | Filters data-mining flukes | Calculate the **Deflated Sharpe Ratio (DSR)** for all extracted paths across all runs and reject rules with $\text{DSR} < 0.95$. |
| **P3 (Low)** | **Drop Binary Pattern Features** | Reduces noise & dimension | Permanently strip all 90+ candlestick pattern features from the input matrix to reduce model dimensionality and training time. |

---

### Summary Takeaway

Stop spending time celebrating Sharpe 28 cross-asset "discoveries." Treat any Sharpe $> 3.0$ as a bug until proven otherwise with continuous-time, fee-deducted, purged out-of-sample backtests. Once you fix the Sharpe formula and add transaction costs, you will likely see your metrics drop to realistic levels (Sharpe 0.8–1.6, Win Rate 51%–55%). That is where real quantitative research begins.

---

### References
[^1]: López de Prado, M. (2018). *Advances in Financial Machine Learning*. John Wiley & Sons. (Chapters 2–4: Purged & Embargoed Cross-Validation).
[^2]: Bailey, D. H., & López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality*. Journal of Portfolio Management, 40(5), 94-107.

# GPT
Verdict brutal : l’idée XGBoost → règles interprétables est recevable comme outil de découverte. Le système actuel n’est pas validé scientifiquement, encore moins prêt pour paper trading. Les Sharpe 8–35, les taux de réussite 70–80 % et le « 100 % cross-asset » sont surtout des signaux d’alarme sur le protocole, pas une preuve d’alpha.

Les problèmes bloquants que le code confirme :

- Le backtester ouvre une position pour chaque signal, y compris si une position est déjà ouverte. Les trades de 6 h à 2 j se chevauchent massivement, mais le Sharpe traite leurs rendements comme indépendants (`mean/std * sqrt(n_trades)`). La somme des retours, le drawdown et le nombre de trades ne correspondent donc pas à un portefeuille exécutable. C’est probablement la source principale des Sharpe absurdes. [backtester.py](/D:/midas_v2/einherjar/src/einherjar/research/xgb_einhers/backtester.py:158)

- Le « cross-asset » backteste chaque règle sur toute l’histoire ETH/LTC/ADA/BCH, pas uniquement sur une période postérieure et strictement OOS. Il ne démontre donc ni généralisation temporelle ni indépendance par rapport à la recherche. Pire : le critère `passing` ne vérifie que `n_trades >= 5` et `win_rate >= 40 %`; le Sharpe et le profit factor affichés ne sont pas des conditions d’admission. [test_cross_asset.py](/D:/midas_v2/einherjar/src/einherjar/research/tests/test_xgb_einhers/test_cross_asset.py:98)

- Le correctif « val = [60 %, 80 %] » n’utilise pas les mêmes indices que l’entraînement. L’entraînement découpe `X_valid` après exclusion des labels invalides et embargo ; le backtest découpe séparément `X_aligned` à 60–80 %, sans embargo. Le rapport reconnaît une approximation de 5 %, mais ce n’est pas acceptable pour revendiquer une validation. [runner.py](/D:/midas_v2/einherjar/src/einherjar/research/xgb_einhers/runner.py:229) [Sprint 2.5](/D:/midas_v2/einherjar/audits/SPRINT_2_5_RAPPORT_2026-08-17.md)

- Le holdout n’est plus un holdout : il est utilisé pour filtrer les candidats (`min_holdout_trades`), relu par les tests/rapports, puis utilisé pour appuyer les décisions ultérieures. Une fois consulté et exploité dans le processus de sélection, il ne peut plus servir de test final.

- Les résultats « multi-actif » sont méthodologiquement invalides dans le runner : le modèle est entraîné sur la concaténation d’actifs, mais le backtest retombe sur l’historique complet du seul actif primaire. Le code l’indique explicitement comme un fallback full-dataset. [runner.py](/D:/midas_v2/einherjar/src/einherjar/research/xgb_einhers/runner.py:303)

- Le système annoncé comme « ATR-based SL/TP » ne l’est pas : `compute_atr` existe mais n’est pas utilisé. Les règles XGBoost emploient toujours TP 2,5 % / SL 1,5 % fixes. [backtester.py](/D:/midas_v2/einherjar/src/einherjar/research/xgb_einhers/backtester.py:259)

- Les exécutions sont optimistes : pas de gestion réaliste des gaps à l’ouverture, de slippage, de spread variable, de latence, de liquidité, de funding/borrow pour les shorts, ni de limite de capital/exposition. Un stop touché est rempli exactement au niveau du stop, même si le marché le traverse.

- La régularisation est utile mais insuffisante. En plus, `early_stopping_rounds` est défini dans la configuration mais jamais transmis à `model.fit`; le jeu de validation ne pilote donc pas d’arrêt anticipé. [model.py](/D:/midas_v2/einherjar/src/einherjar/research/xgb_einhers/model.py:154)

Réponses directes :

1. L’approche est bonne pour générer des hypothèses interprétables, pas pour établir une stratégie. Extraire beaucoup de chemins de nombreux arbres crée une énorme famille de tests corrélés.

2. Oui, il reste un risque sérieux de contamination et surtout une invalidité de découpage : les indices modèle, labels valides, OHLCV aligné et backtest ne sont pas reliés par une unique table/index temporel canonique.

3. « 100 % cross-asset » est trop beau pour être crédible ici. L’explication la plus probable est le backtest de signaux corrélés, sur périodes historiques intégrales et avec une barre de passage très basse — pas une universalité de marché.

4. Cinq trades n’ont aucune valeur probante. Même 50 trades peuvent être insuffisants si les positions se chevauchent. Il faut des observations de portefeuille non chevauchantes, ou une correction explicite de la dépendance. Il n’existe pas de seuil universel ; exige au minimum plusieurs régimes et suffisamment de temps OOS pour mesurer un Sharpe avec intervalle de confiance, pas seulement cinq réalisations.

5. Il faut journaliser chaque essai — chemins, seeds, horizons, filtres, versions de données, réglages manuels — puis appliquer une validation imbriquée avec CPCV/purge/embargo, PBO et Deflated Sharpe Ratio. Ces méthodes ciblent précisément le biais de sélection et les distributions de rendements non normales. [Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551), [PBO / CSCV](https://www.quantresearch.org/Publications.htm)

6. Oui : le constat suggère un détecteur de régimes de volatilité/risque, pas une découverte de patterns. Ce n’est pas mauvais, mais il faut le formuler et le tester comme tel, contre des baselines simples : volatilité, trend, buy-and-hold, breakout, et une stratégie conditionnée uniquement sur ces huit features.

Priorités, dans cet ordre :

1. Réécrire le backtest comme portefeuille : une règle claire sur le chevauchement, capital, exposition, ordre d’exécution, coûts/slippage/funding et courbe d’equity horodatée.
2. Construire une table canonique par actif avec timestamp, X, labels et OHLCV ; produire tous les splits par indices de cette table, purgeant l’horizon réel.
3. Geler un vrai test final jamais ouvert ; faire la sélection uniquement dans des folds internes purgés.
4. Refaire le cross-asset sur segments strictement postérieurs, jamais sur l’historique complet, et comparer au même portefeuille exécutable.
5. Calculer PBO/CPCV et DSR sur la matrice complète des candidats, pas uniquement sur les survivants.
6. Ajouter des baselines et des tests de placebo : labels permutés temporellement, règles perturbées, coûts multipliés, slippage adverse.
7. Seulement après cela : paper trading très limité, avec monitoring et kill-switch.

L’étiquette honnête aujourd’hui est : prototype de recherche prometteur, avec des défauts d’évaluation qui invalident ses revendications de performance. Ne déployez pas de capital réel sur ces résultats.