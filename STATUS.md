# STATUS — État réel du moteur de recherche Einherjar

> Vue d'ensemble rapide de l'état d'implémentation. Pour les détails, voir
> `src/einherjar/research/ALGORITHME_RESEARCH.md` et `ONTOLOGY.md`.
>
> **Statuts** : ✅ implémenté et testé · 🟡 partiel · ⏸ reporté · 🚧 en cours

**Dernière mise à jour** : 2026-08-03 (BNF chantier Phase 1-4 + comparateur multi-obj + pilotage)

---

## Socle commun (P0/P1/P2)

| # | Item | Statut | Notes |
|---|---|---|---|
| P0 #1 | SL/TP relatifs (multiples d'ATR) | ✅ | `CalibratedParams.compute_sl_tp_at_entry`, recalculés à chaque trade |
| P0 #2 | Calibration sur vrais trades (pas prix 1.0) | ✅ | `train_calibrate` : passe provisoire 1.5×ATR, MFE/MAE réels |
| P0 #3 | CLI sur données réelles | ✅ | `npy_real_loader` + `_load_real_data` (alignment OHLCV/features) |
| P0 #4 | refine/admit/holdout réels | ✅ | `handle_*` consomment la sélection, plus de stubs |
| P0 #5 | PBO + drawdown réels | ✅ | `evaluate_pbo` (CPCV López de Prado, K=6+embargo) + `max_drawdown_from_returns` (equity_curve par trade) |
| P0 #6 | Cross-asset ≥ 2 actifs | ✅ | `evaluate_cross_asset` exige min_n_assets=2 (opt-in allow_single_asset) |
| P0 #7 | Transformations : pas de fallback silencieux | ✅ | `EvaluationError` si `transformation != None` |
| P0 #10 | Générateurs complets (NSGA-II/Memetic/TypedGP/Beam/Random) | ✅ | Branche `p10-moteurs-reels` mergée sur main (commits b62c941/ce687e8/b2fad31/0fa52dc) |
| P1 #1 | Seuils relatifs (quantiles) | ✅ | `data/threshold_calibration.py` + `_sample_threshold_for` |
| P1 #2 | Raffinement déprécié | ✅ | `BeamRefiner` déprécié (DeprecationWarning), migrer vers générateurs |
| P1 #3 | Séparation stricte génération/val/holdout | ✅ | Tests `test_separation_stricte_p1_3.py` + ledger atomique |
| P1 #4 | Holdout persistant (ledger atomique) | ✅ | `holdout/ledger.py` (append-only, fsync, anti-réentrance) |
| P1 #5 | Diversité par corrélation (ret_series) | ✅ | `_max_pearson` aligne par début commun, `ret_series` stocké dans corpus/archive |
| P1 #6 | Quotas diversité | ✅ | `evaluate_quotas` (family_max, type_max, direction_min) + extensible |
| P1 #7 | Chargement corpus persistant | ✅ | `corpus/store.py` (CorpusEntry + CorpusStore.append/load/summary) |
| P1 #8 | Archive candidat complet | ✅ | `ArchiveEntry` enrichi (ret_series, fingerprint canonique, etc.) |
| P1 #9 | Versionnage données (schema, hash, période, timezone) | ✅ | `data/versioning.py` enrichi (content_sha256, start/end_ts) |
| P1 #10 | Contrôle bloquant données | ✅ | `data/validation.py` (NaN, index monotone, gaps, anti-fuite) |
| P2 #1 | Sharpe annualisé dynamique (par timeframe) | ✅ | `periods_per_year_for_timeframe` (1m→525600, 1h→8760, 1d→365, etc.) |
| P2 #2 | Contrat 218 features strict | ✅ | `_check_taxonomy_218` (loader.py) enforcé |
| P2 #3 | Chemins config + dépendances + test démarrage | ✅ | `pyproject.toml` OK + `test_demarrage_p2_3.py` |
| P2 #4 | Statut documents à jour | ✅ | Ce fichier `STATUS.md` (reflexion centralisee) |

---

## Moteurs (P10, chantier dédié)

| Moteur | Statut | Notes |
|---|---|---|
| RandomSearchGenerator | ✅ | Random search sous contraintes (typage, profondeur, ratios). Seuils calibrés (P1 #1). |
| BeamSearchGenerator | ✅ | Vraie expansion par niveaux + scoring intermédiaire + élitisme. Requiert engine. |
| TypedGPGenerator | ✅ | STGP complet (Koza 1992 + Montana 1995) : grow+full init, sélection tournoi k=3, crossover sous-arbre type-preserving, mutation sous-arbre, élitisme. Requiert engine. |
| MemeticGenerator | ✅ | EA TypedGP + phase LSO (hill climbing réel via engine). Requiert engine. |
| NSGA2Generator | ✅ | Deb 2002 complet : non-dominance sort + crowding + SBX + 8 contraintes dures + 4 objectifs. Médiane multi-actifs (P1 #6). Requiert engine. |
| GrammaticalEvolutionGenerator | ✅ | **RÉACTIVÉ** (BNF Phase 4) : chromosome 8 bits × 12 gènes, décodage via `BNFCodec` (Ryan 1998), produit des Hypothèses valides. Tire au hasard parmi 218 features + bloc relations OHLCV. Branche `bnf-ge-integration`. |

---

## Raffinement, Admission, Holdout, Comparateur, Pilotage

| Bloc | Statut | Notes |
|---|---|---|
| Raffinement | 🟡 (déprécié) | `BeamRefiner` déprécié (P1 #2). Migration vers "générer N nouveaux candidats via les générateurs + pipeline complet". |
| Comparateur multi-objectif | ✅ | `GeneratorComparator` partage l'engine avec les générateurs. Score composite = 0.40·sharpe + 0.30·admission + 0.15·diversity + 0.15·coherence (normalisation min-max entre moteurs, redistribution des poids si coherence=0). Branche `comparator-multiobj`. |
| Admission (7 critères S-3.4) | ✅ | `evaluate_all_criteria` (DSR, PBO, bootstrap CI, n_trades, cross-asset, max_dd). |
| Admission (multi-actifs strict) | ✅ | P0 #6 : médiane par actif/fold (pas moyenne), `min_n_assets=2`. |
| Holdout persistant | ✅ | `HoldoutEvaluator` + `HoldoutLedger` (anti-réentrance post-redémarrage, atomique). |
| Pilotage (rapport par moteur) | ✅ | Module `pilotage.py` : `PilotageReport` (volume/perf/diversité/admissions/rejets par moteur + synthèse globale + winner). Branche `pilotage-report`. |

---

## BNF (chantier séparé, en dernier)

| # | Phase | Statut | Notes |
|---|---|---|---|
| Phase 1 | Terminaux BNF (218 features) | ✅ | 100% (Lots 0-4e, 218/218 features couvertes). 4 helpers (`_default_atomic_grammar`, `_oscillator_grammar`, `_unit_bounded_grammar`, `_correlation_grammar`, `_signal_grammar`) + 1 bloc relations OHLCV. |
| Phase 2 | Anti-tautologies (`compute_ohlcv_range_quantiles` pour `q_range_pX`) | ⏸ | **DIFFÉRÉ** à la demande user — peut attendre. |
| Phase 3 | Orientation sémantique des patterns (108) | ✅ | Module `bnf_semantic.py` : `SemanticOrientation` enum (BULLISH/BEARISH/NEUTRAL) + heuristique de classification (cas exacts prioritaires puis suffixes). `GrammaticalEvolutionGenerator` ajoute `meta.semantic_orientation`. Branche `bnf-phase-3-semantic`. |
| Phase 4 | Parser BNF → Condition/ConditionNode + intégration GE | ✅ | Module `bnf_parser.py` : parser BNF textuel + décodeur GE (Ryan 1998, codon % nb_productions, wraparound) + mapper AST→Condition (4 cas : quantile/discret/composé/featureref). `GrammaticalEvolutionGenerator` réactivé (plus de `NotImplementedError`). 30 tests bout-en-bout. Branches `bnf-phase-4-parser` + `bnf-ge-integration`. |

---

## Tests

- **169 tests verts** au total (`python -m unittest discover -s src/einherjar/research/tests -p 'test_*.py'`)
- Répartition : 84 socle + 30 parser BNF + 39 sémantique + 15 pilotage + 1 adapt
- Couverture par bloc : moteur, admission, corpus, holdout, validation, diversité, raffinement, démarrage, BNF (parser, sémantique), pilotage
- Ruff clean (line length 100, conventions Google)

---

## Branches mergées sur main (par sprint)

- `p10-moteurs-reels` : 5 générateurs vrais (Random, Beam, TypedGP, Memetic, NSGA-II)
- `sprint-p1-coherence-persistance` : P1 #2-#8 + P2 #1-#4
- `bnf-ohlcv` : 5 features OHLCV + bloc relations (Lot 0)
- `bnf-lot1` à `bnf-lot4e` : 213 features restantes en 11 sous-lots
- `bnf-phase-4-parser` : parser BNF + mapper
- `bnf-ge-integration` : réactivation `GrammaticalEvolutionGenerator`
- `bnf-phase-3-semantic` : orientation sémantique 108 patterns
- `comparator-multiobj` : score composite 4 axes
- `pilotage-report` : rapport structuré par moteur

`main` : **socle complet + BNF chantier Phase 1/3/4 + comparateur multi-obj + pilotage**, pipeline 7 étapes fonctionnel, 6 générateurs.
