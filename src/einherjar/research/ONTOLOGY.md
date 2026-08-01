# Ontologie d'Einherjar

> Document de référence — Contrat conceptuel du système.
> Toute décision d'architecture ou d'implémentation doit s'y conformer.
> Toute modification de ce document nécessite une revue explicite.

---

## Préambule — Philosophie

Einherjar est un **laboratoire de découverte de stratégies de trading**. Son produit est un **corpus d'Einhers indépendants** : des unités de trading autonomes, validées, caractérisées et réutilisables, chacune capable d'exploiter une amplitude de marché spécifique avec un avantage statistique démontré.

L'objectif n'est pas de trouver la meilleure stratégie. C'est de **découvrir un grand nombre de stratégies indépendantes et correctes**, dont la combinaison diversifiée fait croître un capital réel de manière robuste.

Le système fonctionne comme un chercheur : il observe, formule des hypothèses, les expérimente, les valide, conserve ce qui résiste. Il ne manipule jamais le marché directement — il manipule des représentations.

---

## Concepts fondamentaux

Le système manipule **8 concepts**. Chacun a un rôle unique, des attributs obligatoires, un cycle de vie, et des relations explicites avec les autres.

### 1. Données (Data)

**Définition** — Mesure brute d'un marché sur une bougie. Le substrat physique du système.

**Attributs obligatoires**
- `asset` : identifiant de l'actif (ex: `BTCUSD`)
- `timeframe` : granularité temporelle (ex: `1h`)
- `timestamp` : instant de clôture de la bougie
- `open`, `high`, `low`, `close`, `volume` : valeurs OHLCV

**Cycle de vie**
- `ingérée` → `certifiée` (cohérence temporelle, absence de trous) → `consommée`

**Relations** : aucune. Sert exclusivement de substrat au calcul des features.

**N'est pas** : un signal, une décision, une stratégie, une interprétation.

---

### 2. Feature

**Définition** — Représentation calculée d'une propriété observable du marché à un instant donné. Une feature informe, ne décide pas.

**Attributs obligatoires**
- `nom` : référence canonique dans la taxonomie (ex: `rsi_14`)
- `type` ∈ {`atomic`, `quantitative`, `pattern`, `signal`, `factor`}
- `famille` : nature économique (momentum, trend, volatility, volume_flow, market_regime, statistical, risk, microstructure, market_structure, price_action, cross_asset, other)
- `domaine_valeur` ∈ {`float`, `booléen`, `catégoriel`}

**Attributs dérivés**
- `relations` : liste de pointeurs vers features liées
  - `derives_from` : la feature est une transformation d'une autre
  - `aggregates` : la feature agrège plusieurs autres features
  - `equivalent` : la feature porte la même information qu'une autre en régime normal
- `verified` : flag de fiabilité de la formule (`code`, `name_only`, `unverified`, …)

**Cycle de vie**
- `définie` (dans la taxonomie) → `calculée` (sur les données) → `validée` (par spot-check) → `utilisée` (dans les conditions)

**Relations** : aucune vers le haut du système. Relations latérales (entre features) uniquement.

**N'est pas** : une condition, un signal de trading, une stratégie.

**Référence** : 218 features utilisables, définies dans `feature_taxonomy_corrected.json`. 28 features exclues (19 fantômes non vérifiés, 8 meta-factors à collinéarité composée, 1 alias pur).

---

### 3. Contexte de marché (Market Context)

**Définition** — Vecteur complet des features à un instant `(asset, timeframe, timestamp)`. C'est ce que le moteur "voit" à un moment donné. Une condition porte sur un contexte, pas sur des features isolées.

**Attributs**
- `asset`, `timeframe`, `timestamp` : identifiants
- `snapshot` : dictionnaire `{feature_name → valeur}` pour toutes les features applicables à ce contexte

**Cycle de vie**
- `construit` (à chaque bougie close) → `évalué` (par les conditions) → `archivé` (si pertinent, pour debug et traçabilité)

**Relations** : contient des features. Est évalué par les conditions.

**N'est pas** : un signal, un Einher, un trade.

**Note opérationnelle** : conceptuellement important, mais en pratique c'est une **ligne du DataFrame enrichi**. On ne le matérialise pas en objet séparé sauf pour la documentation.

---

### 4. Condition

**Définition** — Contrainte booléenne portant sur un contexte de marché. Décrit un état du marché, ne prédit rien.

**Forme** — `feature + transformation + opérateur + valeur`, combinable par AND/OR/NOT/XOR.

**Attributs**
- `feature_ref` : référence à une feature
- `transformation` : optionnelle (cf. langage de recherche)
  - Pour continues : `identity`, `percentile(window)`, `zscore(window)`, `slope(window)`, `delta(window)`, `crossover(other)`, `crossunder(other)`
  - Pour patterns : `presence`, `absence`, `within_N(window)`, `streak(window)`
  - Pour signals : `state == X`, `transition(from, to)`, `in_state_for(N)`
  - Pour factors : transformations continues
- `opérateur` ∈ {`<`, `>`, `<=`, `>=`, `==`, `!=`, `in`}
- `valeur` : constante typée (numérique, percentile `P0..P100`, ou catégorielle)
- `relations_logiques` : pour les conditions composées

**Cycle de vie**
- `générée` (par le moteur ou un humain) → `évaluée` (sur un contexte) → résultat `True` ou `False`

**Relations** : utilise des features. Est contenue dans une ou plusieurs hypothèses.

**N'est pas** : une hypothèse (ne prédit rien), un signal, un trade, un Einher.

**Coût** : quasi-nul. Une condition est une expression évaluée instantanément.

---

### 5. Amplitude (Amplitude)

**Définition** — Mouvement de marché qu'un Einher cherche à capturer, exprimé en unités de prix ou en multiple d'ATR. Indépendante des features qui servent à la prédire. C'est la **mesure de succès** d'un Einher.

**Attributs**
- `valeur` : numérique
- `unité` ∈ {`prix_absolu`, `multiple_ATR`}
- `direction_implicite` ∈ {`+`, `-`} (signe selon la direction de l'Einher)

**Cycle de vie**
- `définie` (dans une hypothèse) → `traduite` (en TP/SL dans l'Einher) → `mesurée` (en live par les trades réalisés)

**Relations** : définie dans une hypothèse. Traduite en TP/SL dans l'Einher.

**Pourquoi concept propre** : sans amplitude cible, on ne sait pas ce qu'on cherche. Deux Einhers de même direction mais d'amplitudes différentes ne sont pas comparables. C'est aussi ce qui permet la cohérence entre hypothèse et trade réel.

---

### 6. Hypothèse (Hypothesis)

**Définition** — Affirmation testable selon laquelle, dans un contexte de marché donné, un mouvement d'amplitude spécifiée se produira dans une direction donnée. C'est l'**unité de recherche** du moteur.

**Forme** — `(condition_tree, amplitude_cible, direction)`.

**Attributs**
- `id` : identifiant unique
- `condition_tree` : arbre de conditions combinées par AND/OR/NOT/XOR
- `amplitude_cible` : référence à l'amplitude prédite
- `direction` ∈ {`long`, `short`}
- `universe` : `assets` (liste ou `*`), `timeframes` (liste ou `*`)
- `mesures_brutes` (calculées sur l'historique) :
  - `n_occurrences` : nombre de fois où la condition_tree a été vraie
  - `mfe_moyen` : Maximum Favorable Excursion moyen observé
  - `mae_moyen` : Maximum Adverse Excursion moyen observé
  - `distribution_rendements` : distribution des rendements sur N bougies après signal
  - `winrate_brut` : fréquence d'atteinte de l'amplitude cible
  - `temps_moyen_vers_amplitude` : durée typique pour atteindre l'amplitude

**Cycle de vie**
1. `construite` (par le générateur)
2. `testée` sur l'historique → calcul des mesures brutes
3. **Soit** scorée positivement → `candidate` à la validation
4. **Soit** scorée négativement → `rejetée`, archivée pour mémoire

**Relations** : contient des conditions (≥1). Référence une amplitude. Mène à un Einher si elle passe la validation finale.

**N'est pas** : un Einher (manque la validation statistique formelle, le SL/TP figé, les métriques de portefeuille). N'est pas une condition isolée (a une dimension prédictive).

**Coût** : modéré. Tester une hypothèse = itérer sur l'historique, calculer MFE/MAE/winrate sur chaque occurrence.

---

### 7. Einher

**Définition** — Hypothèse validée et entièrement spécifiée, définissant une stratégie de trading autonome capable d'exploiter une amplitude de marché avec un avantage statistique démontré.

**Attributs obligatoires**
- `id` : identifiant unique
- `hypothèse_origine` : référence à l'hypothèse dont il est issu
- `direction` ∈ {`long`, `short`}
- `condition_tree` : héritée de l'hypothèse
- `universe` : `assets`, `timeframes`
- `sl` : stop-loss (prix absolu ou multiple ATR)
- `tp` : take-profit (prix absolu ou multiple ATR)
- `amplitude_cible` : héritée
- `métriques_validation` :
  - Statistiques de base : `win_rate`, `sharpe`, `expectancy`, `profit_factor`, `sortino`
  - Métriques portefeuille : `cagr`, `max_drawdown`, `mar_ratio`, `ulcer_index`, `risk_of_ruin`
  - Robustesse : stabilité temporelle, généralisation cross-asset
- `domaine_validité` : régime(s) de marché où il performe, contraintes connues
- `statut` ∈ {`candidat`, `validé`, `actif`, `dégradé`, `archivé`}

**Cycle de vie**
1. Issu d'une hypothèse validée (`admis`)
2. Statut `validé` après passage de la validation finale (métrique portefeuille)
3. Statut `actif` : exécuté en live
4. Si dégradation persistante → `dégradé` puis `archivé`
5. Si reconvergence après dégradation → retour à `actif`

**Relations** : encapsule une hypothèse validée. Appartient à un corpus. Capture une amplitude.

**N'est pas** : un signal isolé, un trade unique, une hypothèse non validée.

**Garanties** : complet, exécutable, reproductible, généralisable, validé hors-échantillon.

---

### 8. Archive (Jalon de rejets)

**Définition** — Mémoire historique du système contenant l'ensemble des hypothèses et Einhers rejetés ou déclassés au cours du processus. Conservée séparément du corpus actif pour deux raisons : (1) éviter de polluer l'espace de décision, (2) préserver une **mémoire scientifique reproductible** : ce qui a été testé, sur quelles données, avec quels paramètres, et pourquoi ça n'a pas marché. Une règle rejetée doit pouvoir être **réévaluée** sur un nouveau jeu de données / une nouvelle version, sans réécrire l'historique.

**Attributs obligatoires**
- `id` : identifiant unique de l'élément archivé
- `type_élément` ∈ {`hypothèse`, `einher`}
- `raison_rejet` : cause documentée et normalisée (cf. catalogue S-3.6)
- `date_rejet` : horodatage UTC

**Attributs de contexte (reproductibilité)**
- `data_version` : identifiant du dataset utilisé (hash ou tag du bundle OHLCV)
- `seed` : graine RNG du run qui a émis ce rejet
- `splits` : bornes temporelles explicites `{train: [t0, t1), val: [t1, t2), holdout: [t2, t3)}` + `purge_window` (nb bougies purgées en bordure de label) + `embargo_bougies` (nb bougies exclues entre splits)
- `costs_simulated` : `{spread_pct, commission_pct, slippage_pct}` appliqués lors de l'évaluation
- `sl_tp_source` : provenance des niveaux SL/TP (`from_train` uniquement, jamais recalibrés)

**Attributs métriques (snapshot complet)**
- `mesures_brutes_train` : `MesuresBrutes` sur le train (inclut MAE_p75, MFE_p50 utilisés pour figer SL/TP)
- `mesures_brutes_val` : `MesuresBrutes` sur le val
- `metriques_portefeuille_val` : `{cagr, max_drawdown, mar_ratio, ulcer_index, risk_of_ruin, profit_factor, n_trades}`
- `bootstrap_ci_val` : `{sharpe_ci_low, sharpe_ci_high, ret_ci_low, ret_ci_high}` par block bootstrap
- `deflated_sharpe_ratio` : DSR corrigé pour le nombre d'essais indépendants
- `probability_of_backtest_overfitting` : PBO estimé par CPCV léger
- `descriptors_comportementaux` : `{signal_overlap_vs_corpus, ret_corr_vs_corpus, distribution_by_regime, distribution_by_horizon}`

**Attributs d'empreinte (fingerprint canonique)**
- `fingerprint_structurel` : hash de `condition_tree + direction + universe + amplitude_cible + sl + tp` (anti-doublon exact, déterministe)
- `fingerprint_comportemental` : hash des `descriptors_comportementaux` arrondis à une grille stable (permet de détecter des règles structurellement différentes mais économiquement équivalentes, dans la limite de la même époque de données)
- `fingerprint` : concaténation `fingerprint_structurel + ":" + fingerprint_comportemental`

**Cycle de vie**
- `créée` (à chaque rejet) → `consultable` (lecture seule, jamais modifiée) → `jamais_supprimée` (conservée indéfiniment, mais **réévaluable** sur nouveau `data_version`)

**Relations** : reçoit des rejets depuis les hypothèses (échec de validation) et les Einhers (dégradation ou archivage). Est distincte du corpus actif mais partage la même structure d'empreinte.

**N'est pas** : un cimetière. C'est une **base de connaissances négative** : ce qui a été essayé et n'a pas fonctionné, consultable pour éviter de re-tester inutilement, et pour comprendre les frontières du système.

**Usage** :
- Empêcher la re-soumission d'un doublon exact sur le même `data_version` (par `fingerprint_structurel`)
- Empêcher la re-soumission d'un équivalent comportemental sur le même `data_version` (par `fingerprint_comportemental`)
- Analyser les causes de rejet (améliorer la génération)
- Auditer le système (savoir ce qui a été tenté, quand, sur quelles données, avec quels paramètres)
- **Réévaluer** une hypothèse rejetée sur un nouveau `data_version` sans la considérer comme un doublon (anti-leak : on n'efface jamais l'historique)

---

### 9. Corpus

**Définition** — Mémoire opérationnelle du système contenant l'ensemble des Einhers actuellement considérés comme exploitables. Vivant, versionné, diversifié par construction.

**Attributs**
- `version` : identifiant de version (`corpus_v1`, `corpus_v2`, …)
- `einhers` : liste des Einhers actifs
- `contraintes_diversité` :
  - Quotas par famille (ex: max 40% d'une famille)
  - Quotas par type (ex: max 60% de patterns)
  - Quotas par direction (équilibre long/short)
  - Quotas par horizon (court, moyen, long terme)
- `statistiques_globales` : métriques agrégées du corpus
- `date_dernière_mise_à_jour`

**Cycle de vie**
- `initialisé` (vide) → `peuplé` (premier batch d'Einhers validés) → `maintenu` (entrées/sorties/re-validations) → `versionné` (à chaque évolution majeure)

**Relations** : contient des Einhers. Est alimenté par la validation, purgé par la dégradation. Est consommé par le moteur d'exécution. Est distinct de l'Archive (qui reçoit les rejets).

**N'est pas** : un dump JSON figé, un catalogue exhaustif, un cimetière de tentatives. C'est un objet vivant qui respire.

---

## Diagramme des relations

```
                    ┌─────────┐
                    │ Données │
                    └────┬────┘
                         │ produit
                         ▼
                    ┌─────────┐         ┌──────────────────┐
                    │ Features│◄────────│ Contexte marché  │
                    └────┬────┘ contient└────────┬─────────┘
                         │                      │
                         │ utilise              │ évalue
                         ▼                      ▼
                    ┌──────────────────────────────┐
                    │         Condition            │
                    └────────────┬─────────────────┘
                                 │ compose
                                 ▼
                    ┌──────────────────┐    référence
                    │    Hypothèse     │──────────────┐
                    └────────┬─────────┘              │
                             │                       │
                  ┌──────────┴──────────┐            │
                  │                     │            │
             rejets (non validé)    valide         │
                  │                     │            │
                  ▼                     ▼            ▼
            ┌──────────┐         ┌─────────────┐  ┌────────────┐
            │ Archive  │         │   Einher    │◄─│ Amplitude  │
            └──────────┘         └──────┬──────┘  └────────────┘
              reçoit les               capture
              rejets                     │
                                        │ contient
                                        ▼
                                 ┌─────────────┐
                                 │   Corpus    │
                                 └─────────────┘
```

**Lecture** : les données produisent les features. À un instant, les features forment un contexte. Une condition est une contrainte sur ce contexte. Une hypothèse est composée de conditions et référence une amplitude. Une hypothèse qui échoue à la validation est archivée. Une hypothèse validée devient un Einher qui capture l'amplitude. Le corpus contient les Einhers actifs. L'Archive conserve les rejets.

---

## Sémantique d'évaluation

Comment chaque concept est réellement évalué, testé, validé.

### S-1. Évaluation d'une condition

**Entrée** : une condition (atomique ou composée) + un contexte (ligne du DataFrame à un instant donné).
**Sortie** : un booléen strict.

#### Algorithme

```
eval(condition, context) → bool:
  if condition est composée:
    gauche = eval(condition.gauche, context)
    droite = eval(condition.droite, context)
    return appliquer(condition.relation_logique, gauche, droite)
  
  if condition est atomique:
    value = context[condition.feature_ref]   # peut être NaN
    if value is NaN or not finite: return False
    if condition.transformation existe:
      value = apply(value, condition.transformation, context.history)
      if erreur: return False
    return compare(value, condition.opérateur, condition.valeur)
```

#### Règles dures

- **Valeur manquante ou non-finie** → `False`. Une condition ne se déclenche jamais sur de la donnée absente. Pas de NaN-propagation, pas de tri-state.
- **Transformation impossible** (paramètre hors borne, fenêtre trop courte) → `False` + log warning.
- **Le contexte doit être strictement contemporain** : `eval` ne consulte que la ligne courante + l'historique strictement antérieur (jamais futur).
- **Pas d'effet de bord** : `eval` est pure, ne modifie rien.
- **Comparaisons flottantes** : `==` et `!=` utilisent `isclose` avec tolérance (`atol=1e-9`).

#### Table de compatibilité type / transformation

| Type de feature | Transformations autorisées |
|---|---|
| `atomic` (float) | identity, percentile, zscore, slope, delta, crossover, crossunder |
| `quantitative` (float) | idem atomic |
| `pattern` (bool) | presence, absence, within_N, streak |
| `signal` (catégoriel) | state_eq, transition, in_state_for |
| `factor` (float [0,1]) | identity, percentile, zscore, slope, delta |

Toute autre combinaison = erreur de typage à la compilation, pas à l'exécution.

---

### S-2. Test d'une hypothèse

**Entrée** : une hypothèse (condition_tree + amplitude + direction + universe) + un historique OHLCV enrichi + un jeu temporel (train | val | holdout).
**Sortie** : un objet `MesuresBrutes` calculé **uniquement sur le jeu demandé**.

#### Algorithme (vectorisé)

```
test(hypothèse, historique, period='train') → MesuresBrutes:
  
  N = fenêtre d'observation, FIXE par hypothèse
      (calculée UNE SEULE FOIS sur le train, voir S-2.1)
  K = cooldown d'observation, défaut 5 (paramétrable par hypothèse)
  sl_price, tp_price = niveaux d'exécution figés depuis le train (jamais recalibrés)
  
  results = []
  for (asset, tf) in hypothèse.universe:
    df = historique[asset, tf]
    signals_mask = eval_vectorisé(hypothèse.condition_tree, df)
    signal_indices = where(signals_mask)[0]
    
    last_valid = len(df) - N
    filtered = filter(signal_indices, lambda i: i < last_valid)
    spaced = appliquer_cooldown(filtered, K)
    
    for idx in spaced:
      # Entrée à l'OPEN de la bougie t+1, pas au close
      entry_price = df.open[idx + 1]
      
      # Simulation intrabar de TP et SL sur la fenêtre [t+1, t+N]
      future_high = df.high[idx+1 : idx+N+1].to_numpy()
      future_low  = df.low[idx+1  : idx+N+1].to_numpy()
      future_open = df.open[idx+1 : idx+N+1].to_numpy()
      future_close= df.close[idx+1: idx+N+1].to_numpy()
      
      # Convention de priorité : SL touché avant TP sur la même bougie
      # (hypothèse conservatrice, en pratique l'ordre réel est indéterministe)
      if hypothèse.direction == long:
        exit_price, exit_reason, mfe, mae = _simulate_long(
          entry_price, sl_price, tp_price, future_high, future_low, future_open, future_close)
      else:  # short
        exit_price, exit_reason, mfe, mae = _simulate_short(
          entry_price, sl_price, tp_price, future_high, future_low, future_open, future_close)
      
      ret_pct = (exit_price - entry_price) / entry_price
      if hypothèse.direction == short:
        ret_pct = -ret_pct
      
      results.append({
        'mfe_pct':        mfe / entry_price,
        'mae_pct':        mae / entry_price,
        'ret_pct_brut':   ret_pct,
        'exit_reason':     exit_reason,    # 'tp' | 'sl' | 'timeout'
        'n_bougies_held': _bars_to_exit,
      })
  
  return aggregate(results)


_simulate_long(entry, sl, tp, highs, lows, opens, closes):
  mfe = max(highs) - entry
  mae = entry - min(lows)
  for i in range(len(highs)):
    # L'ordre intrabar est indéterministe — convention : SL d'abord (conservateur)
    if lows[i] <= sl:
      return sl, 'sl', mfe, entry - lows[i]
    if highs[i] >= tp:
      return tp, 'tp', max(highs[:i+1]) - entry, entry - min(lows[:i+1])
  return closes[-1], 'timeout', mfe, mae
```

#### S-2.1. Fenêtre d'observation `N` (figée depuis le train)

`N` est **calculée une seule fois**, sur le **train set uniquement**, jamais recalculée sur la validation ou le holdout. Reproductibilité garantie.

- Si `amplitude.unité == prix_absolu` : `N = clamp(ceil(amplitude.valeur / atr_p50), min_N, max_N)`, où `atr_p50` = médiane de l'ATR(14) sur le **train** (percentile 50, plus robuste que la moyenne face aux outliers et aux changements de régime). `min_N` et `max_N` sont des garde-fous (ex: `min_N = 3`, `max_N = 50`).
- Si `amplitude.unité == multiple_ATR` : `N = clamp(round(amplitude.valeur * K_atr), min_N, max_N)` où `K_atr` est un facteur de conversion bougie/ATR dépendant du timeframe.

**Pourquoi le percentile plutôt que la moyenne** : l'ATR moyen global est sensible aux régimes de volatilité extrêmes (un seul krach fait exploser la moyenne). Le percentile 50 (médiane) est robuste.

**Pourquoi clamp** : sans bornes, on peut obtenir N=2 (bruit pur) ou N=500 (pas un trade). Les garde-fous forcent un trade "raisonnable".

#### S-2.2. Cooldown d'observation `K`

Si deux signaux sont à moins de `K` bougies, seul le premier est conservé. Défaut `K = 5`. Écrrasable par hypothèse.

**Justification** : sans cooldown, on surestime la fréquence de signal et on crée de l'auto-corrélation dans les MesuresBrutes.

#### Mesures brutes en sortie

```
MesuresBrutes:
  n_signals: int
  n_tp_hit: int
  n_sl_hit: int
  n_timeout: int
  
  mfe_mean_pct: float
  mae_mean_pct: float
  mfe_p50, mfe_p75, mfe_p90: float
  mae_p50, mae_p75, mae_p90: float
  
  ret_mean_pct: float               # brut, sans frais
  ret_std_pct: float
  sharpe_brut: float               # ret_mean / ret_std * sqrt(N_periodes_par_an)
  
  # Distingue maintenant TP et SL (pas de 'ret > 0')
  tp_hit_rate: float               # n_tp_hit / n_signals
  sl_hit_rate: float               # n_sl_hit / n_signals
  timeout_rate: float             # n_timeout / n_signals
  
  avg_holding_period: float        # n_bougies moyen
  time_to_amplitude_mean: float    # bougies moyennes pour atteindre TP (sur les tp_hit)
  
  # Statistiques de blocage (block bootstrap)
  bootstrap_sharpe_ci_low: float   # borne basse du CI 95% par block bootstrap
  bootstrap_sharpe_ci_high: float
  
  per_asset_stats: dict[(asset, tf) → MesuresBrutes]
```

#### Règles dures

- **Entrée à l'OPEN** de la bougie t+1, jamais au close (la bougie du signal est close au moment où on l'observe).
- **Simulation intrabar** : TP et SL sont testés sur high/low de chaque bougie, pas seulement sur close.
- **Convention de priorité** : si TP et SL sont touchés dans la même bougie, on compte SL (hypothèse conservatrice). En réalité l'ordre est indéterministe, on documente cette convention.
- **SL/TP figés depuis le train** : `sl_price` et `tp_price` sont calculés une seule fois à partir des MesuresBrutes du train set, puis **jamais recalibrés** dans les phases val/holdout/live.
- **Pas de frais à ce stade** : les MesuresBrutes sont **brutes** (frais = 0). Les frais sont appliqués au stade de la validation finale.
- **Pas de position sizing** : rendements en %, pas en dollars.
- **Bougies de fin d'historique** : exclues si `idx + N >= len(df)`.
- **Block bootstrap** : les intervalles de confiance utilisent des blocs temporels (pas d'échantillonnage i.i.d.) pour respecter l'autocorrélation des rendements.

---

### S-3. Validation et admission d'un Einher

**Entrée** : une hypothèse testée (MesuresBrutes) + un dataset historique complet.
**Sortie** : verdict `admis` ou `rejeté`. Si admis, un `Einher` complet.

> **Note de portée** : le moteur d'évaluation hors-échantillon détaillé dans S-2 (entrée OPEN, simulation intrabar TP/SL, block bootstrap) est exécuté **avant** toute décision de rejet. Les étapes ci-dessous consomment ses `MesuresBrutes` et **n'inventent aucun nouveau calcul** sur les données val/holdout.

#### S-3.1 — Jeux temporels verrouillés (train / val / holdout)

**Trois jeux disjoints, temporellement ordonnés** :

```
[t0 ───────────── t1 ───────────── t2 ───────────── t3)
   ▲                ▲                ▲                ▲
   train (60%)      val (20%)        holdout (20%)    fin

Période            Bornes             Rôle
─────────────────────────────────────────────────────────
train              [t0, t1)           calibration : N, SL, TP, poids fitness
val                [t1, t2)           tuning hyperparamètres du générateur
holdout            [t2, t3)           ÉVALUATION FINALE — JAMAIS touché pendant le dev
```

**Purging** : à chaque borne `t1` et `t2`, on exclut les bougies dont le label d'amplitude déborde sur le jeu suivant. Pour une amplitude avec horizon `N`, on purge `N` bougies après chaque split côté jeu suivant, pour garantir qu'aucun trade ouvert en fin de train n'est résolu avec une bougie du val (et idem val→holdout).

**Embargo** : on exclut en outre `embargo_bougies` bougies supplémentaires (défaut : 1 bougie) après chaque frontière de split, comme marge de sécurité contre les micro-fuites de features (ex : un rolling mean qui intègre la dernière bougie du jeu précédent).

**Lock** : une fois les bornes `t0, t1, t2, t3` et `embargo` fixés, ils sont **gelés** pour toute la durée de développement. Tout re-run utilise les mêmes bornes. Le holdout n'est **jamais** consulté pendant le développement.

**Règles dures** :
- Le holdout est consommé **une seule fois**, à la toute fin, pour produire l'évaluation finale.
- Tout contact accidentel du holdout pendant le dev (chargement, agrégation, log) invalide le run.
- Les splits sont définis **par actif et par timeframe**, pas globalement (le temps est relatif à chaque série).

#### S-3.2 — Calibration sur le train (exécutée une seule fois)

Sur le train **uniquement** :

1. **Calcul de `N`** : `clamp(ceil(amplitude / atr_p50), min_N, max_N)` où `atr_p50` est la médiane de l'ATR(14) sur le train (cf. S-2.1).
2. **Calcul de `sl_price` et `tp_price`** :
   - `tp = entry ± MFE_p50` (médiane du Maximum Favorable Excursion observé sur le train)
   - `sl = entry ± MAE_p75` (percentile 75 du Maximum Adverse Excursion observé sur le train)
3. **Calcul des poids de fitness** (si métrique composite retenue) ou des seuils initiaux des critères d'admission.

**Ces trois sorties sont figées** : `N`, `sl_price`, `tp_price`, poids. Elles ne sont **jamais** recalculées sur val ou holdout.

#### S-3.3 — Test sur le val (tuning, pas admission)

Sur le val, on ré-évalue la même hypothèse avec `N`, `sl_price`, `tp_price`, coûts **figés depuis S-3.2**. Aucun paramètre n'est modifié en fonction du résultat.

Sortie : `MesuresBrutes_val` + `MétriquesPortefeuille_val` (cf. S-3.5) + descripteurs comportementaux (cf. S-3.7).

C'est sur le val que se font :
- Le tuning des hyperparamètres du **générateur** (taille de population, taux de mutation, profondeur max, etc.).
- L'estimation de la distribution de Sharpe (par block bootstrap) qui nourrira le DSR et le PBO.
- La sélection finale des Einhers candidats à l'admission.

Le val ne **décide pas** l'admission seul : il qualifie ou disqualifie.

#### S-3.4 — Critères d'admission (substitué aux 9 critères naïfs)

Une hypothèse est admissible si **tous** les critères suivants sont satisfaits (ET logique strict) :

| Critère | Seuil | Justification | Calcul |
|---|---|---|---|
| **DSR (Deflated Sharpe Ratio)** | > seuil cible (à recalibrer empiriquement, V1 : > 0.95 si n_essais_indep > 1) | Corrige le Sharpe pour le nombre d'essais indépendants et la non-normalité des rendements | DSR(Sharpe_val, n_essais_indep, skew, kurtosis) selon Bailey & López de Prado |
| **PBO (Probability of Backtest Overfitting)** | < 0.20 (à recalibrer) | Probabilité que la configuration sélectionnée soit overfittée, estimée par CPCV léger | Combinatorial Purged CV avec K=6 groupes, N=6 chemins, sur le val uniquement |
| **Block bootstrap CI sur Sharpe** | `sharpe_ci_low > 0` (IC 95%, blocs de longueur = `1.5 × N_max`) | L'intervalle de confiance bas du Sharpe est strictement positif | Block bootstrap sur la série de rendements val |
| **Block bootstrap CI sur ret_total** | `ret_total_ci_low > 0` (IC 95%) | Idem sur le retour total net | Idem |
| **`n_trades` total** | ≥ `n_trades_min` (à recalibrer, V1 : 30) | Significativité statistique minimale | Comptage direct |
| **`consistency_cross_asset`** | ≥ `frac_assets_positifs` (V1 : 0.70) des actifs du `universe` doivent être positifs sur val | Généralisation minimale non triviale | Par-asset, puis agrégation |
| **`max_drawdown_val`** | < `dd_max` (V1 : 0.25) | Risque de portefeuille maîtrisé | Sur l'equity curve val |
| **Diversité comportementale vs corpus courant** | `signal_overlap` < `overlap_max` ET `ret_corr` < `corr_max` (V1 : 0.30 et 0.50) | L'invariant I-8 ne s'évalue pas seulement sur la structure mais aussi sur le comportement | Descripteurs comportementaux (S-3.7) vs corpus |

**Seuils** : valeurs de V1, **à recalibrer empiriquement** après les 50 premiers Einhers testés. Pas figés avant données réelles. Le protocole de recalibrage est lui-même une décision à documenter.

**Comparaison avec l'ancienne liste** : on retire `gap_train_to_test` (trompeur — voir critique de l'IA tierce : la comparaison naïve train/test ignore la volatilité des estimates), `hit_tp_rate_test` (proxy bruité — la qualité d'un trade est mieux capturée par le ratio Sharpe/PnL net que par un simple compteur de hits), et `cagr_test` comme seuil absolu (couvert par le bootstrap CI sur ret_total).

#### S-3.5 — Métriques portefeuille (calculées sur le val)

```
MétriquesPortefeuille:
  cagr_val:             (1 + total_return_val)^(1/years) - 1
  max_drawdown_val:     max(peak_to_trough) sur equity curve val
  mar_ratio_val:        cagr_val / max_drawdown_val
  ulcer_index_val:      sqrt(mean(drawdown²))
  risk_of_ruin_val:     P(DD > 50%) estimée par bootstrap
  avg_holding_period:   durée moyenne des trades
  n_trades_per_year:    fréquence d'exécution
  profit_factor_val:    sum(gains) / |sum(pertes)|
```

Ces métriques sont **toujours accompagnées de leur IC bootstrap** (cf. S-2 : `bootstrap_sharpe_ci_low/high`, idem sur CAGR et MDD). Une métrique sans IC est inutilisable en présence d'autocorrélation.

#### S-3.6 — Création de l'Einher (étape D corrigée)

Si toutes les marches de S-3.4 passent :

- L'Einher hérite de l'hypothèse (condition_tree, direction, universe, amplitude_cible)
- Le **SL** = `sl_price` **figé depuis S-3.2** (calculé sur le train)
- Le **TP** = `tp_price` **figé depuis S-3.2** (calculé sur le train)
- Les **métriques de validation publiées** sont celles du val, accompagnées de leur IC bootstrap
- Statut initial : `validé`, puis `actif` au premier passage dans le scheduler live
- Le holdout n'est **pas** consulté à ce stade

**Catalogue des raisons de rejet** (normalisé pour l'Archive) :
`DSR_FAIL`, `PBO_FAIL`, `BOOTSTRAP_CI_FAIL`, `N_TRADES_FAIL`, `CROSS_ASSET_FAIL`, `DD_FAIL`, `DIVERSITY_FAIL`, `ALREADY_IN_ARCHIVE`, `SEMANTIC_CHANGED`, `OTHER`.

#### S-3.7 — Descripteurs comportementaux (diversité I-8 corrigée)

L'invariant I-8 (diversité du corpus) impose des quotas structurels (famille, type, direction), mais deux Einhers structurellement différents peuvent produire les mêmes dates de signal et les mêmes pertes. La diversité **de portefeuille** n'est garantie que par des descripteurs **comportementaux**, exportés par chaque hypothèse et croisés avec le corpus courant.

Descripteurs obligatoires :

```
descriptors_comportementaux:
  signal_dates:              liste des timestamps de signal
  signal_overlap_vs_corpus:  Jaccard moyen des `signal_dates` avec chaque Einher du corpus
  ret_series:                série des rendements nets val, alignée sur le calendrier val
  ret_corr_vs_corpus:        matrice de corrélation des `ret_series` avec ceux du corpus
  distribution_by_regime:    Sharpe / winrate / expectancy par régime de marché (bull / bear / range)
  distribution_by_horizon:   mêmes stats par horizon de sortie (TP / SL / timeout)
  holding_period_hist:       histogramme de la durée des trades
  exit_reason_breakdown:     {tp: %, sl: %, timeout: %}
```

**Usage** : le moteur d'admission croise ces descripteurs avec le corpus courant et applique les seuils de S-3.4 (`signal_overlap` < `overlap_max`, `ret_corr` < `corr_max`). La Portfolio Layer (hors périmètre du moteur de découverte) reçoit ensuite ces descripteurs comme information d'allocation ; le moteur de découverte ne décide ni sizing ni allocation.

#### S-3.8 — Évaluation finale sur le holdout (sacrée)

Une seule fois, **après** que tous les hyperparamètres du générateur, les poids de fitness, les seuils d'admission, et les splits train/val sont gelés :

```
holdout_eval(einher, holdout):
  # N, sl_price, tp_price, coûts, fingerprint : tous figés depuis train/val
  mesures = test(einher.hypothèse, holdout, sl=figé, tp=figé, N=figé)
  publier(métriques_holdout + IC_bootstrap_holdout + descripteurs_comportementaux_holdout)
  comparer aux métriques val : dégradation attendue et bornée
  archiver(tout) # dans l'Archive avec data_version + seed + splits figés
```

**Règles dures** :
- Le holdout n'est consulté qu'**une seule fois** dans toute la vie d'un Einher.
- Le résultat est **publié tel quel**, sans recalibrage.
- Toute différence importante entre val et holdout (`degradation_ratio > seuil`) déclenche un flag mais ne modifie pas l'admission (le holdout n'a pas le droit de "réparer" une admission contestable).

#### S-3.9 — Gestion des rejets

Toute hypothèse qui échoue à un critère est **archivée** (concept #8 Archive) avec :
- Snapshot complet : `data_version`, `seed`, `splits`, `purge_window`, `embargo_bougies`, `costs_simulated`, `sl_tp_source = "from_train"`
- `MesuresBrutes` train ET val
- `MétriquesPortefeuille` val + IC bootstrap
- DSR, PBO
- Descripteurs comportementaux
- Fingerprint canonique (structurel + comportemental)
- Raison du rejet (catalogue S-3.6) et date

Les Einhers qui se dégradent en live sont également archivés avec leur historique de performance. **Aucune donnée n'est supprimée** : l'Archive est append-only et **réévaluable** sur nouveau `data_version`.

#### S-3.10 — Règles dures

- **Aucune fuite temporelle** : aucun paramètre (N, SL, TP, poids, seuils) n'est jamais calculé sur val ou holdout puis utilisé sur le train. La calibration est **uniquement** sur le train.
- **Holdout sacré** : un seul accès pendant toute la durée de vie d'un Einher, à la toute fin.
- **Seuils stricts ET** : un seul critère non satisfait = rejet.
- **Pas d'exception manuelle** : pas de bypass possible des seuils.
- **Re-soumission interdite** sur le même `data_version` : un `fingerprint_structurel` ou `fingerprint_comportemental` déjà présent dans l'Archive bloque la re-soumission. Sur un **nouveau** `data_version`, la réévaluation est permise (anti-leak : on n'efface jamais l'historique).
- **SL/TP figés depuis le train** : `sl_price` et `tp_price` ne sont **jamais** recalibrés sur val ou holdout, ni en live.

---

## Invariants du système

Ces propriétés ne doivent **jamais** être violées. Si l'une d'elles est brisée, le système est compromis.

### I-1 — Un Einher est complet
Une stratégie incomplète n'est pas un Einher. Tout Einher doit posséder : entrée, sortie, SL, TP, conditions, direction, métriques. Sans exception.

### I-2 — Un Einher est reproductible
À données identiques, le système doit toujours produire exactement le même comportement. Deux exécutions du moteur sur le même dataset produisent les mêmes Einhers (même fingerprint).

### I-3 — Un Einher est généralisable
Une règle qui ne marche que sur un actif ou une période n'est pas un Einher. La validation exige au minimum N actifs de la classe (cohérence cross-asset), la survie à un split train/val/holdout temporellement strict, et le passage des seuils DSR + PBO + block bootstrap CI. Walk-forward seul ne suffit plus : il faut une preuve statistique de non-surapprentissage. Cette contrainte est non négociable.

### I-4 — Un Einher a un edge net de frais prouvé
Pas de Sharpe isolé, pas de backtest flatteur. L'edge est mesuré après application réaliste des coûts (spread, commission, slippage) et doit rester positif sur la période de test out-of-sample.

### I-5 — Aucune fuite temporelle
La recherche n'utilise jamais le futur. Walk-forward strict, pas de peek. Une condition qui "marche" en utilisant l'information des bougies futures est un bug, pas une découverte.

### I-6 — Une seule signification par information
Pas de doublons dans l'espace de recherche. `RSI`, `RSI_norm`, `RSI_signal`, `Factor_Momentum` ne sont pas quatre informations indépendantes — ce sont quatre représentations de la même information, liées par des relations typées. Les relations entre features sont imposées et exploitées par le moteur.

### I-7 — Types non confondus
Les patterns ne se manipulent pas comme des variables continues. Les factors ne se manipulent pas comme des observations atomiques. Chaque type a sa sémantique et ses opérations autorisées. La grammaire l'interdit au niveau syntaxique.

### I-8 — Diversité structurelle du corpus
Le corpus n'est jamais dominé par une seule famille, un seul type, ou une seule direction. La diversité est une **contrainte imposée** au niveau du corpus, pas une propriété espérée au niveau des Einhers individuels.

### I-9 — Métriques objectives
La qualité d'un Einher ne dépend jamais d'une intuition ou d'un jugement qualitatif. Elle est exclusivement mesurée par des métriques quantitatives, reproductibles, calculées sur des données hors-échantillon.

### I-10 — Deux étapes de validation
Le scoring rapide (pendant la recherche) et la validation finale (avant admission au corpus) sont des étapes **strictement séparées** avec des exigences différentes. Le score rapide est optimiste par construction (rôle = ne pas rater de candidats). La validation finale est stricte (rôle = admettre ou rejeter). Mélanger les deux = s'auto-valider.

---

## Exclusions explicites

Les concepts suivants sont **hors périmètre** du moteur de découverte. Ils sont la responsabilité d'autres couches du système.

| Concept exclu | Justification | Couche responsable |
|---|---|---|
| Sizing / taille de position | Décision de risque, pas de signal | Risk Manager |
| Levier utilisé | Décision de risque | Risk Manager |
| Cooldown / fréquence | Gestion opérationnelle | Scheduler |
| Capital allocation | Décision portefeuille | Portfolio Layer |
| Order type (market/limit/OCO) | Décision d'exécution | Execution Layer |
| Frais par venue | Dépend du broker | Execution Layer |
| Calendrier / heures de marché | Dépend de l'asset | Scheduler |
| PnL réalisé | Mesure post-trade | Execution / Portfolio |
| Drawdown en temps réel | Mesure runtime | Risk Manager |
| Corrélation inter-actifs | Décision portefeuille | Portfolio Layer |

Le moteur de découverte est un sous-système **pur**, sans contamination par les autres couches.

---

## Propriétés structurelles de l'ontologie

1. **Unidirectionnalité** : l'information va du bas vers le haut uniquement (Données → Features → Condition → Hypothèse → Einher → Corpus). Aucun concept ne remonte vers ses composants.

2. **Pas de saut** : on ne peut pas passer d'une feature directement à un Einher. Il faut traverser les niveaux : Feature → Condition → Hypothèse → Einher. Chaque transition ajoute de l'information.

3. **Coût progressif** : générer une condition = gratuit. Tester une hypothèse = modéré. Valider un Einher = lourd. La complexité du calcul croît avec le niveau, ce qui force à être économe en haut.

4. **Diversité imposée au corpus, pas aux Einhers** : la diversité est une propriété de l'ensemble. Les Einhers individuels peuvent être similaires ; c'est le corpus qui doit être diversifié.

5. **Aucun concept ne porte d'information sur le futur** sauf l'amplitude (qui est une cible, pas une prédiction chiffrée). Conditions, features, contexte = présent uniquement.

---

## Glossaire alphabétique

| Concept | Définition express |
|---|---|
| **Amplitude** | Mouvement de marché qu'un Einher cherche à capturer. Cible, pas prédiction chiffrée. |
| **Archive** | Mémoire historique append-only des rejets (hypothèses non validées, Einhers dégradés). Distincte du corpus. |
| **Condition** | Contrainte booléenne sur un contexte. Décrit un état, ne prédit rien. |
| **Contexte de marché** | Vecteur complet des features à un instant `(asset, timeframe, timestamp)`. |
| **Corpus** | Mémoire opérationnelle vivante, versionnée, diversifiée, des Einhers validés. |
| **Données** | Mesure brute OHLCV d'une bougie. Substrat physique du système. |
| **Einher** | Stratégie de trading complète, validée, autonome, exécutable. Unité de production de PnL. |
| **Feature** | Représentation calculée d'une propriété du marché. Informe, ne décide pas. |
| **Hypothèse** | Affirmation testable : "si condition(s) vraie(s), alors mouvement d'amplitude X dans direction Y". Unité de recherche. |

---

## Statut du document

**État** : **Non figé — V2.1 en cours**. Ontologie + sémantique d'évaluation définies, mais la section S-3 a été substantiellement révisée suite à la critique d'une IA tierce (2026-08-01). Voir `ALGORITHME_RESEARCH.md` section "Critique IA tierce" pour la liste des points acceptés.

**Composants figés** :
- 9 concepts (Données, Feature, Contexte, Condition, Amplitude, Hypothèse, Archive, Einher, Corpus)
- 10 invariants (I-1 à I-10)
- 5 propriétés structurelles
- 4 sections de sémantique (S-1 à S-3, dont S-3 refondu en V2.1)

**Composants révisés en V2.1 (2026-08-01)** :
- **S-2** : entrée à l'OPEN de t+1, simulation intrabar TP/SL, N figée depuis train avec ATR percentile 50, block bootstrap dans MesuresBrutes, convention SL avant TP sur même bougie.
- **S-3** : walk-forward 70/30 → split train/val/holdout 60/20/20 avec purging/embargo ; 9 critères naïfs → DSR + PBO (CPCV léger) + block bootstrap CI + cross-asset + n_trades + max_dd + diversité comportementale ; SL/TP figés depuis train uniquement (correction du look-ahead bias) ; holdout sacré, consulté une seule fois ; descripteurs comportementaux obligatoires pour la diversité réelle du corpus.
- **Section 8 (Archive)** : ajout `data_version`, `seed`, `splits` (avec `purge_window`, `embargo_bougies`), `costs_simulated`, `sl_tp_source`, snapshot métriques complet (DSR/PBO/IC bootstrap), descripteurs comportementaux, fingerprint canonique (structurel + comportemental), catalogue de raisons de rejet, garantie de **réévaluabilité** sur nouveau `data_version`.

**Points à recalibrer empiriquement** (V1+, pas bloquants) :
- Seuils d'admission S-3.4 (DSR cible, PBO max, n_trades_min, dd_max, overlap_max, corr_max) après 50+ Einhers testés
- Valeur du cooldown d'observation `K` (défaut 5)
- Longueur de bloc pour le block bootstrap (défaut : `1.5 × N_max`)
- Paramètres du CPCV léger (K=6, N=6 par défaut)

**Points NON figés, en attente de décision empirique** :
- Choix du générateur (random / GE / GP typé / beam) — voir `ALGORITHME_RESEARCH.md` § 10.2
- Pondération de la métrique composite (si on en garde une)
- Architecture détaillée de la grammaire BNF (conceptuelle, pas de code encore)

**Date** : 2026-08-01 (V2.1)

**Auteurs** : Mavis + collaboration utilisateur

**Validé par** : utilisateur

**Toute modification** doit faire l'objet d'une revue explicite et être tracée dans le changelog.
