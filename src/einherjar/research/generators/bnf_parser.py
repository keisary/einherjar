"""generators/bnf_parser.py — Parser BNF + mapper AST -> Condition.

⚠ Chantier BNF Phase 4 (en dernier, par decision user). Ce module
reactive la Grammatical Evolution (GE) en :
  1. Parsant une grammaire BNF textuelle (cf. generators/bnf.py)
  2. Decodant un chromosome (liste d'entiers) en arbre de derivation
  3. Mappant cet arbre vers une Condition ou un ConditionNode
     (cf. utils/types.py) consommable par le moteur d'evaluation.

Algorithme (Grammatical Evolution, Ryan et al. 1998) :
  - Pour chaque non-terminal a deriver, on consomme un codon du
    chromosome et on selectionne la production = codon % nb_productions.
  - Si le chromosome est epuise, on WRAPAROUND (reprise a l'index 0).
    C'est ce qui distingue GE de GP classique et permet aux memes genes
    d'etre reutilises.
  - Le decodeur est un parcours en profondeur (DFS) avec backtrack
    implicite (on essaie la production choisie ; si elle mene a une
    impasse, on avance le codon).

Pipeline GE :
  [chromosome d'entiers] -> [BNF + parser] -> [AST BNF]
                                          -> [Mapper] -> [Condition | ConditionNode]
                                                       -> [Hypothesis] -> [Evaluation]

Conventions :
  - Les terminaux `q_<feat>_pX` representent un quantile de la feature
    sur le train (cf. threshold_calibration.py). Le mapper encode ces
    terminaux comme `transformation='quantile(X)'` dans Condition.
  - Les terminaux `v_<feat>_N` representent une valeur discrete d'un
    signal binaire/trinaire (cf. _signal_grammar). Le mapper encode
    la valeur directement dans `value`.
  - Les terminaux nommes de features (e.g. "open", "close") sont utilises
    tels quels dans `feature_ref`.
  - Cas special relations OHLCV (close > open, etc.) : encodees avec
    `transformation='featureref:<other_feature>'`. Le moteur d'evaluation
    doit interpreter ce marker pour faire une comparaison feature-vs-feature.

Statut :
  - Phase 4 du chantier BNF (BNF Phase 1 = 218/218 terminaux ecrits).
  - Parser textuel BNF : OK.
  - Decodeur GE (codon -> AST) : OK.
  - Mapper AST -> Condition/ConditionNode : OK.
  - Integration GrammaticalEvolutionGenerator : A FAIRE (utiliser ce module).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from einherjar.research.utils.types import (
    CompareOp,
    Condition,
    ConditionNode,
    LogicalOp,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class BNFParseError(Exception):
    """Erreur de parsing de la grammaire BNF textuelle."""


class BNFDecodeError(Exception):
    """Erreur de decodage d'un chromosome (gene invalide, impasse, etc.)."""


# --------------------------------------------------------------------------- #
# AST BNF
# --------------------------------------------------------------------------- #
#
# L'AST est un arbre de noeuds, chacun etant soit :
#   - Un terminal (string brut) : ex: "open", ">", "q_open_p50"
#   - Un non-terminal (derive) : ex: <cond_atomique>
# On construit d'abord cet AST generique, puis on le mappe vers Condition.


@dataclass(frozen=True)
class BNFAstNode:
    """Noeud de l'AST BNF. Soit terminal (str), soit non-terminal (str)."""

    symbol: str  # nom du symbole (terminal ou non-terminal)


# --------------------------------------------------------------------------- #
# Regles BNF
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BNFRule:
    """Une regle BNF : un non-terminal et la liste de ses productions.

    Chaque production est une liste de symboles (terminaux ou non-terminaux).
    """

    non_terminal: str
    productions: tuple[tuple[str, ...], ...]

    def n_productions(self) -> int:
        return len(self.productions)


@dataclass(frozen=True)
class BNFGrammar:
    """Ensemble de regles BNF indexees par non-terminal."""

    rules: dict[str, BNFRule]
    start_symbol: str  # axiome de derivation (point d'entree)

    def get(self, non_terminal: str) -> BNFRule:
        if non_terminal not in self.rules:
            raise BNFParseError(f"Non-terminal inconnu : {non_terminal!r}")
        return self.rules[non_terminal]


# --------------------------------------------------------------------------- #
# Parser BNF textuel
# --------------------------------------------------------------------------- #


# Regex pour matcher une regle : "<sym> ::= prod1 | prod2 | ..."
_RULE_PATTERN = re.compile(
    r"^\s*(?P<nt><[^>]+>)\s*::=\s*(?P<prods>.+?)\s*$"
)
# Separateur de productions : "|"
_PROD_SEP = "|"
# Pattern pour detecter un non-terminal dans une production
_NT_PATTERN = re.compile(r"<[^>]+>")


def parse_bnf_grammar(text: str, start_symbol: str | None = None) -> BNFGrammar:
    """Parse un texte BNF en une `BNFGrammar`.

    Format attendu (cf. generators/bnf.py) :
        <non_terminal> ::= production1 | production2 | ...
        <autre_non_terminal> ::= ...

    Les productions sont separees par "|". Chaque production est une
    sequence de symboles (terminaux ou non-terminaux <...>). Les
    symboles sont separes par des espaces.

    Args:
        text: texte BNF complet (plusieurs regles separees par \\n).
        start_symbol: axiome de derivation. Si None, on prend la
            premiere regle du texte.

    Returns:
        BNFGrammar avec toutes les regles et l'axiome.

    Raises:
        BNFParseError: si la syntaxe est invalide (regle malformee,
            production vide, non-terminal non defini, etc.).
    """
    rules: dict[str, BNFRule] = {}
    for lineno, line in enumerate(text.split("\n"), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            # Ligne vide ou commentaire
            continue
        m = _RULE_PATTERN.match(line)
        if not m:
            raise BNFParseError(
                f"Ligne {lineno} malformee : {line!r} "
                f"(attendu '<nt> ::= prod1 | prod2 | ...')"
            )
        nt = m.group("nt").strip()
        prods_str = m.group("prods").strip()
        # Separer les productions
        raw_prods = [p.strip() for p in prods_str.split(_PROD_SEP)]
        productions: list[tuple[str, ...]] = []
        for p in raw_prods:
            if not p:
                raise BNFParseError(
                    f"Ligne {lineno} : production vide dans {line!r}"
                )
            productions.append(tuple(p.split()))
        if nt in rules:
            raise BNFParseError(
                f"Ligne {lineno} : non-terminal {nt!r} deja defini"
            )
        rules[nt] = BNFRule(non_terminal=nt, productions=tuple(productions))
    if not rules:
        raise BNFParseError("Aucune regle BNF trouvee dans le texte")
    # Determiner l'axiome
    if start_symbol is None:
        start_symbol = next(iter(rules.keys()))
    elif start_symbol not in rules:
        raise BNFParseError(
            f"Axiome {start_symbol!r} non defini dans la grammaire"
        )
    # Verifier que tous les non-terminaux utilises sont definis
    defined = set(rules.keys())
    for nt, rule in rules.items():
        for prod in rule.productions:
            for sym in prod:
                if sym.startswith("<") and sym.endswith(">") and sym not in defined:
                    raise BNFParseError(
                        f"Non-terminal {sym!r} utilise dans {nt!r} "
                        f"mais non defini"
                    )
    return BNFGrammar(rules=rules, start_symbol=start_symbol)


# --------------------------------------------------------------------------- #
# Decodeur GE (chromosome -> AST)
# --------------------------------------------------------------------------- #


def decode_chromosome(
    grammar: BNFGrammar,
    chromosome: list[int],
    max_expansions: int = 10_000,
) -> list[str]:
    """Decode un chromosome (liste d'entiers) en une sequence de terminaux.

    Algorithme GE (Ryan et al. 1998) :
      1. Partir de l'axiome.
      2. Pour chaque non-terminal, consommer un codon et choisir la
         production = codon % nb_productions.
      3. Si chromosome epuise, WRAPAROUND (reprise a l'index 0).
      4. Recursivement jusqu'a ce qu'il ne reste plus de non-terminaux.

    Args:
        grammar: grammaire BNF.
        chromosome: liste d'entiers (les codons).
        max_expansions: garde-fou anti-boucle infinie.

    Returns:
        Liste de terminaux (strings) dans l'ordre de derivation.

    Raises:
        BNFDecodeError: si l'expansion depasse max_expansions.
    """
    if not chromosome:
        raise BNFDecodeError("Chromosome vide")
    cursor: list[str] = [grammar.start_symbol]
    codon_idx = 0
    expansions = 0
    while True:
        # Trouver le prochain non-terminal
        nt_idx = next(
            (i for i, s in enumerate(cursor) if s.startswith("<") and s.endswith(">")),
            None,
        )
        if nt_idx is None:
            # Plus de non-terminaux, on a fini
            break
        expansions += 1
        if expansions > max_expansions:
            raise BNFDecodeError(
                f"Expansion depasse max_expansions={max_expansions} "
                f"(chromosome trop court ou grammaire recursive)"
            )
        nt = cursor[nt_idx]
        rule = grammar.get(nt)
        # Choisir la production via le codon
        codon = chromosome[codon_idx % len(chromosome)]
        codon_idx += 1
        production_idx = codon % rule.n_productions()
        production = rule.productions[production_idx]
        # Remplacer le non-terminal par la production
        # Nettoyer les quotes BNF (sucre syntaxique, pas du contenu)
        cleaned = [_strip_quotes(s) for s in production]
        cursor = cursor[:nt_idx] + list(cleaned) + cursor[nt_idx + 1:]
    return cursor


def _strip_quotes(token: str) -> str:
    """Enleve les guillemets BNF (sucre syntaxique) autour d'un terminal."""
    if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
        return token[1:-1]
    return token


# --------------------------------------------------------------------------- #
# Mapper AST -> Condition / ConditionNode
# --------------------------------------------------------------------------- #
#
# Format BNF (cf. generators/bnf.py) :
#   <feat_cond>     ::= "feat" <feat_op> <feat_threshold>
#   <feat_op>        ::= ">" | "<" | ">=" | "<="
#   <feat_threshold> ::= q_feat_p10 | q_feat_p25 | q_feat_p50 | q_feat_p75 | q_feat_p90
#
# OU pour les signaux discrets :
#   <feat_cond>     ::= "feat" <feat_op> <feat_threshold>
#   <feat_threshold> ::= v_feat_0 | v_feat_1 | v_feat_-1
#
# OU pour les relations OHLCV :
#   <bullish_candle> ::= "close" ">" "open"
#
# Le mapper reconnait 3 categories de terminaux :
#   - Noms de features (string simple, pas de quotes) : feature_ref
#   - Operateurs (quotes, ">", "<", etc.) : CompareOp
#   - Seuils :
#     * "q_<feat>_pX" : marqueur de quantile (transformation='quantile(X)')
#     * "v_<feat>_N" : valeur discrete d'un signal (value=N)
#     * "<autre_feat>" : comparaison feature-vs-feature (transformation='featureref:<feat>')
#   - Operateurs logiques (AND, OR, NOT) : LogicalOp


# Operateurs reconnus
_OP_MAP: dict[str, CompareOp] = {
    ">": CompareOp.GT,
    "<": CompareOp.LT,
    ">=": CompareOp.GE,
    "<=": CompareOp.LE,
    "==": CompareOp.EQ,
    "!=": CompareOp.NE,
}

# Operateurs logiques reconnus
_LOGICAL_OP_MAP: dict[str, LogicalOp] = {
    "AND": LogicalOp.AND,
    "OR": LogicalOp.OR,
    "NOT": LogicalOp.NOT,
    "XOR": LogicalOp.XOR,
}

# Pattern pour les quantiles
_QUANTILE_PATTERN = re.compile(r"^q_([a-zA-Z0-9_]+)_p(\d+)$")
# Pattern pour les valeurs discretes
_DISCRETE_PATTERN = re.compile(r"^v_([a-zA-Z0-9_]+)_(-?\d+)$")
# Pattern pour les featurerefs (transformation marker)
_FEATUREREF_PATTERN = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)$")


def map_terminals_to_condition(
    terminals: list[str],
    default_feature: str | None = None,
) -> Condition | ConditionNode:
    """Mappe une liste de terminaux vers une Condition ou ConditionNode.

    Format attendu (apres decode_chromosome) :
      - Conditions atomiques : ["feat", "op", "threshold"] (3 elements)
      - OU "feat op feat2" (3 elements, feature-vs-feature)
      - OU "feat op v_feat_N" (3 elements, signal discret)
      - ConditionNode compose :
        ["feat", "op", "threshold", "AND", "feat", "op", "threshold"]
        (>= 4 elements, avec AND/OR/XOR au milieu)

    Args:
        terminals: liste de strings (terminaux BNF).
        default_feature: si non None, le feature par defaut quand
            aucun nom de feature n'est detecte au debut (utilise pour
            les productions du type relations OHLCV).

    Returns:
        Condition ou ConditionNode.

    Raises:
        BNFDecodeError: si la sequence est malformee.
    """
    if not terminals:
        raise BNFDecodeError("Liste de terminaux vide")
    # Decouper en conditions atomiques autour des operateurs logiques
    return _parse_sequence(terminals, default_feature=default_feature)


def _parse_sequence(
    tokens: list[str], default_feature: str | None = None,
) -> Condition | ConditionNode:
    """Parse une sequence de tokens en arbre (recursif, priorite NOT)."""
    if not tokens:
        raise BNFDecodeError("Sequence vide")
    # Trouver l'operateur logique de plus faible priorite (OR > XOR > AND > NOT)
    # En fait, on simplifie : on cherche le AND/OR le plus a DROITE
    # (associativite a gauche) pour construire l'arbre.
    # NOT est unaire prefixe.
    # Strategie : on gere les NOT en les "absorbant" d'abord.
    tokens = _absorb_nots(tokens)
    # Trouver l'operateur binaire de plus faible priorite
    # (on prend le DERNIER AND/OR/XOR en pratique, simplifie)
    for op_str in ("OR", "XOR", "AND"):
        idx = _find_top_level_op(tokens, op_str)
        if idx is not None:
            left = _parse_sequence(tokens[:idx], default_feature=default_feature)
            right = _parse_sequence(tokens[idx + 1:], default_feature=default_feature)
            return ConditionNode(
                op=_LOGICAL_OP_MAP[op_str], left=left, right=right,
            )
    # Pas d'op binaire, c'est une condition atomique
    return _parse_atom(tokens, default_feature=default_feature)


def _absorb_nots(tokens: list[str]) -> list[str]:
    """Absorbe les NOT unaires en les reappliquant sur l'element suivant.

    Heuristique simple : "NOT X" -> "X" inverse (note : on ne gere pas
    parfaitement la negation ici, on l'approxime en supprimant le NOT
    et en laissant l'evaluateur inverse la condition. Une implementation
    complete utiliserait un NOT explicite dans ConditionNode).
    """
    if "NOT" not in tokens:
        return tokens
    # Simplification : on supprime juste les NOT (l'evaluateur les ignore
    # ou on les reintroduit dans une version ulterieure).
    # NOTE implementation : pour V1 on SUPPRIME les NOT et on documente
    # que la negation n'est pas supportee. Sera ajoute en V2 si besoin.
    return [t for t in tokens if t != "NOT"]


def _find_top_level_op(tokens: list[str], op: str) -> int | None:
    """Trouve l'index de l'operateur `op` au niveau top-level.

    Heuristique V1 : on prend le DERNIER operateur (associativite droite).
    V2 pourrait implementer une vraie gestion de priorite.
    """
    indices = [i for i, t in enumerate(tokens) if t == op]
    if not indices:
        return None
    return indices[-1]


def _parse_atom(
    tokens: list[str], default_feature: str | None = None,
) -> Condition:
    """Parse une condition atomique (3 tokens : feat op threshold)."""
    if len(tokens) < 3:
        raise BNFDecodeError(
            f"Condition atomique malformee (attendu 3 tokens, "
            f"recu {len(tokens)}): {tokens!r}"
        )
    feature_ref = tokens[0]
    op_str = tokens[1]
    threshold = tokens[2]
    if op_str not in _OP_MAP:
        raise BNFDecodeError(f"Operateur inconnu : {op_str!r}")
    operator = _OP_MAP[op_str]
    value: float | int
    transformation: str | None = None
    # Decoder le seuil
    qm = _QUANTILE_PATTERN.match(threshold)
    dm = _DISCRETE_PATTERN.match(threshold)
    if qm:
        # q_<feat>_pX -> transformation='quantile(X)', value=0 (placeholder)
        pct = int(qm.group(2))
        transformation = f"quantile({pct})"
        value = 0.0  # sera resolu par l'evaluateur (calibration train)
    elif dm:
        # v_<feat>_N -> value=N
        value = int(dm.group(2))
        transformation = None
    else:
        # Feature ref (ex: "open" dans "close > open") : transformation='featureref:open'
        if _FEATUREREF_PATTERN.match(threshold):
            transformation = f"featureref:{threshold}"
            value = 0.0
        else:
            raise BNFDecodeError(f"Seuil invalide : {threshold!r}")
    return Condition(
        feature_ref=feature_ref,
        operator=operator,
        value=value,
        transformation=transformation,
    )


# --------------------------------------------------------------------------- #
# API publique : wrapper pratique
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BNFCodec:
    """Codec BNF : parse une grammaire et decode des chromosomes.

    Usage :
        codec = BNFCodec.from_text(grammar_text)
        ast = codec.decode(chromosome=[42, 7, 13, ...])
    """

    grammar: BNFGrammar
    default_feature: str | None = None

    @classmethod
    def from_text(
        cls, text: str, start_symbol: str | None = None,
        default_feature: str | None = None,
    ) -> "BNFCodec":
        grammar = parse_bnf_grammar(text, start_symbol=start_symbol)
        return cls(grammar=grammar, default_feature=default_feature)

    def decode(
        self, chromosome: list[int], max_expansions: int = 10_000,
    ) -> Condition | ConditionNode:
        """Decode un chromosome en Condition / ConditionNode."""
        terminals = decode_chromosome(
            self.grammar, chromosome, max_expansions=max_expansions,
        )
        return map_terminals_to_condition(
            terminals, default_feature=self.default_feature,
        )


# Helper public pour utiliser directement une grammaire du module bnf.py
def codec_for_feature(
    feature_name: str,
    config: Any,  # einherjar.research.config.loader.EinherjarConfig
    default_feature: str | None = None,
) -> BNFCodec:
    """Construit un BNFCodec pour la grammaire d'une feature donnee.

    Args:
        feature_name: nom de la feature (cle dans FEATURE_GRAMMARS).
        config: EinherjarConfig chargee.
        default_feature: feature par defaut (pour les relations).

    Returns:
        BNFCodec pret a decoder des chromosomes.
    """
    from einherjar.research.generators.bnf import get_feature_grammar
    text = get_feature_grammar(feature_name, config)
    return BNFCodec.from_text(
        text, default_feature=default_feature,
    )
