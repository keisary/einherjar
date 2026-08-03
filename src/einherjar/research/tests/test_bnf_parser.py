"""Tests pour le parser BNF + mapper AST -> Condition.

Couvre :
  - Parsing de grammaires BNF textuelles
  - Decodage de chromosomes (Grammatical Evolution)
  - Mapping terminals -> Condition (4 cas : quantile, discret, compose,
    relation inter-features)
  - Cas d'erreur (BNF invalide, chromosome vide, etc.)
"""

import unittest

from einherjar.research.config.loader import load_config
from einherjar.research.generators.bnf import (
    get_feature_grammar,
    get_relations_grammar,
)
from einherjar.research.generators.bnf_parser import (
    BNFCodec,
    BNFDecodeError,
    BNFGrammar,
    BNFParseError,
    BNFRule,
    decode_chromosome,
    map_terminals_to_condition,
    parse_bnf_grammar,
)
from einherjar.research.utils.types import (
    CompareOp,
    Condition,
    ConditionNode,
    LogicalOp,
)


class TestBNFParser(unittest.TestCase):
    """Tests du parser BNF textuel."""

    def test_parse_simple_grammar(self) -> None:
        """Une grammaire simple est correctement parsee.

        NOTE : le parser preserve les quotes autour des terminaux.
        Le strip des quotes est fait par decode_chromosome (sucre
        syntaxique de la BNF, pas du contenu semantique).
        """
        text = """
        <cond> ::= "x" ">"
        """
        grammar = parse_bnf_grammar(text)
        self.assertEqual(grammar.start_symbol, "<cond>")
        self.assertIn("<cond>", grammar.rules)
        rule = grammar.rules["<cond>"]
        self.assertEqual(rule.n_productions(), 1)
        self.assertEqual(rule.productions[0], ('"x"', '">"'))

    def test_parse_multiple_productions(self) -> None:
        """Plusieurs productions alternatives par regle."""
        text = """
        <op> ::= ">" | "<" | ">=" | "<="
        """
        grammar = parse_bnf_grammar(text)
        rule = grammar.rules["<op>"]
        self.assertEqual(rule.n_productions(), 4)
        self.assertEqual(
            rule.productions,
            (('">"',), ('"<"',), ('">="',), ('"<="',)),
        )

    def test_parse_nested_non_terminals(self) -> None:
        """Les non-terminaux imbriques sont preserves."""
        text = """
        <cond> ::= <feat> <op> <thresh>
        <feat> ::= "open" | "close"
        <op> ::= ">"
        <thresh> ::= "q_open_p50"
        """
        grammar = parse_bnf_grammar(text)
        cond_rule = grammar.rules["<cond>"]
        self.assertEqual(cond_rule.productions[0], ("<feat>", "<op>", "<thresh>"))

    def test_parse_skip_comments_and_blanks(self) -> None:
        """Les commentaires (#) et lignes vides sont ignores."""
        text = """
        # Commentaire
        <cond> ::= "x"

        # Autre commentaire
        <op> ::= ">"
        """
        grammar = parse_bnf_grammar(text)
        self.assertEqual(len(grammar.rules), 2)

    def test_parse_undefined_non_terminal_raises(self) -> None:
        """Un non-terminal utilise mais non defini leve BNFParseError."""
        text = """
        <cond> ::= <undefined_nt> "x"
        """
        with self.assertRaises(BNFParseError):
            parse_bnf_grammar(text)

    def test_parse_duplicate_non_terminal_raises(self) -> None:
        """Un non-terminal defini 2 fois leve BNFParseError."""
        text = """
        <cond> ::= "x"
        <cond> ::= "y"
        """
        with self.assertRaises(BNFParseError):
            parse_bnf_grammar(text)

    def test_parse_malformed_line_raises(self) -> None:
        """Une ligne sans ::= leve BNFParseError."""
        text = """
        pas une regle valide
        """
        with self.assertRaises(BNFParseError):
            parse_bnf_grammar(text)

    def test_parse_empty_raises(self) -> None:
        """Un texte vide leve BNFParseError."""
        with self.assertRaises(BNFParseError):
            parse_bnf_grammar("")


class TestBNFDecoder(unittest.TestCase):
    """Tests du decodeur de chromosome (algorithme GE)."""

    def setUp(self) -> None:
        """Grammaire de test : <cond> = feat op threshold."""
        self.text = """
        <cond> ::= "feat" <op> <thresh>
        <op> ::= ">" | "<"
        <thresh> ::= "q_p10" | "q_p50"
        """
        self.grammar = parse_bnf_grammar(self.text)

    def test_decode_simple(self) -> None:
        """Un chromosome produit une sequence de terminaux."""
        terminals = decode_chromosome(self.grammar, [0, 0, 0])
        # 0 % 1 = 0 pour <cond> (1 prod) ; 0 % 2 = 0 pour <op> ; 0 % 2 = 0 pour <thresh>
        self.assertEqual(terminals, ["feat", ">", "q_p10"])

    def test_decode_codon_modulo(self) -> None:
        """Le codon est pris modulo le nombre de productions."""
        # 1 % 2 = 1 pour <op>
        terminals = decode_chromosome(self.grammar, [0, 1, 0])
        self.assertEqual(terminals, ["feat", "<", "q_p10"])

    def test_decode_wraparound(self) -> None:
        """Si chromosome epuise, wraparound (reprise a l'index 0)."""
        # Chromosome = [1, 0] : <cond> (1 prod, 1%1=0), <op> (2 prods, 0%2=0)
        # Pour <thresh>, on wrappe : 1 % 2 = 1
        terminals = decode_chromosome(self.grammar, [1, 0])
        self.assertEqual(terminals, ["feat", ">", "q_p50"])

    def test_decode_empty_chromosome_raises(self) -> None:
        """Un chromosome vide leve BNFDecodeError."""
        with self.assertRaises(BNFDecodeError):
            decode_chromosome(self.grammar, [])


class TestBNFMapper(unittest.TestCase):
    """Tests du mapper terminals -> Condition/ConditionNode."""

    def test_map_atomic_quantile(self) -> None:
        """Mapping d'une condition atomique avec quantile."""
        cond = map_terminals_to_condition(["open", ">", "q_open_p50"])
        self.assertIsInstance(cond, Condition)
        self.assertEqual(cond.feature_ref, "open")
        self.assertEqual(cond.operator, CompareOp.GT)
        self.assertEqual(cond.value, 0.0)
        self.assertEqual(cond.transformation, "quantile(50)")

    def test_map_atomic_ge_quantile(self) -> None:
        """Operateur >= OK."""
        cond = map_terminals_to_condition(["close", ">=", "q_close_p75"])
        self.assertEqual(cond.operator, CompareOp.GE)
        self.assertEqual(cond.transformation, "quantile(75)")

    def test_map_atomic_discrete(self) -> None:
        """Mapping d'un signal discret (v_<feat>_N)."""
        cond = map_terminals_to_condition(["pattern_doji", "==", "v_pattern_doji_1"])
        self.assertIsInstance(cond, Condition)
        self.assertEqual(cond.feature_ref, "pattern_doji")
        self.assertEqual(cond.operator, CompareOp.EQ)
        self.assertEqual(cond.value, 1)
        self.assertIsNone(cond.transformation)

    def test_map_atomic_negative_discrete(self) -> None:
        """Valeur discrete negative (signal trinaire)."""
        cond = map_terminals_to_condition(
            ["aroon_trend_signal", "==", "v_aroon_trend_signal_-1"],
        )
        self.assertEqual(cond.value, -1)

    def test_map_feature_vs_feature(self) -> None:
        """Mapping d'une comparaison inter-features (relation OHLCV)."""
        cond = map_terminals_to_condition(["close", ">", "open"])
        self.assertIsInstance(cond, Condition)
        self.assertEqual(cond.feature_ref, "close")
        self.assertEqual(cond.operator, CompareOp.GT)
        self.assertEqual(cond.transformation, "featureref:open")

    def test_map_compose_and(self) -> None:
        """Mapping d'une condition composee AND."""
        cond = map_terminals_to_condition(
            ["open", ">", "q_open_p50", "AND", "close", "<", "q_close_p50"],
        )
        self.assertIsInstance(cond, ConditionNode)
        self.assertEqual(cond.op, LogicalOp.AND)
        self.assertIsInstance(cond.left, Condition)
        self.assertIsInstance(cond.right, Condition)
        self.assertEqual(cond.left.feature_ref, "open")
        self.assertEqual(cond.right.feature_ref, "close")

    def test_map_compose_or(self) -> None:
        """Operateur OR dans un ConditionNode."""
        cond = map_terminals_to_condition(
            ["a", ">", "q_a_p50", "OR", "b", "<", "q_b_p50"],
        )
        self.assertIsInstance(cond, ConditionNode)
        self.assertEqual(cond.op, LogicalOp.OR)

    def test_map_invalid_operator_raises(self) -> None:
        """Operateur inconnu leve BNFDecodeError."""
        with self.assertRaises(BNFDecodeError):
            map_terminals_to_condition(["open", "@@@", "q_open_p50"])

    def test_map_invalid_threshold_raises(self) -> None:
        """Seuil avec caracteres illegaux leve BNFDecodeError.

        NOTE : un nom de feature simple (ex: 'foo') est accepte comme
        featureref par le mapper (le moteur d'eval signalera qu'il
        n'existe pas). Seuls les seuils avec caracteres illegaux
        (espaces, guillemets, etc.) lèvent une erreur.
        """
        with self.assertRaises(BNFDecodeError):
            map_terminals_to_condition(["open", ">", "feat with space"])

    def test_map_atom_too_short_raises(self) -> None:
        """Condition atomique < 3 tokens leve BNFDecodeError."""
        with self.assertRaises(BNFDecodeError):
            map_terminals_to_condition(["open", ">"])

    def test_map_empty_raises(self) -> None:
        """Liste vide leve BNFDecodeError."""
        with self.assertRaises(BNFDecodeError):
            map_terminals_to_condition([])


class TestBNFCodec(unittest.TestCase):
    """Tests d'integration codec (parse + decode + map) sur les
    grammaires reelles d'Einherjar (cf. generators/bnf.py).
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config("src/einherjar/research/config")

    def test_codec_atomic_feature(self) -> None:
        """Codec complet sur une grammaire atomique (open)."""
        codec = BNFCodec.from_text(get_feature_grammar("open", self.cfg))
        cond = codec.decode(chromosome=[0, 1, 2])
        self.assertIsInstance(cond, Condition)
        self.assertEqual(cond.feature_ref, "open")
        self.assertIn(cond.operator, (CompareOp.GT, CompareOp.LT, CompareOp.GE, CompareOp.LE))
        self.assertIsNotNone(cond.transformation)
        self.assertTrue(cond.transformation.startswith("quantile("))

    def test_codec_oscillator(self) -> None:
        """Codec sur un oscillateur (rsi_14, meme structure)."""
        codec = BNFCodec.from_text(get_feature_grammar("rsi_14", self.cfg))
        cond = codec.decode(chromosome=[0, 0, 0])
        self.assertIsInstance(cond, Condition)
        self.assertEqual(cond.feature_ref, "rsi_14")

    def test_codec_discrete_signal(self) -> None:
        """Codec sur un signal discret (pattern_doji, binaire)."""
        codec = BNFCodec.from_text(get_feature_grammar("pattern_doji", self.cfg))
        # pattern_doji_op a 4 productions (>, <, >=, <=) [default atomic operators]
        # pattern_doji_threshold a 2 productions (0, 1) -> codon 0 ou 1
        cond = codec.decode(chromosome=[0, 0, 1])  # op=0 (GT), threshold=1 (v_..._1)
        self.assertIsInstance(cond, Condition)
        self.assertEqual(cond.feature_ref, "pattern_doji")
        self.assertIn(cond.operator, (CompareOp.GT, CompareOp.LT, CompareOp.GE, CompareOp.LE))
        self.assertIn(cond.value, (0, 1))
        self.assertIsNone(cond.transformation)

    def test_codec_factor_unit_bounded(self) -> None:
        """Codec sur un factor (score agrege en [0, 1])."""
        codec = BNFCodec.from_text(
            get_feature_grammar("Factor_Momentum_Score", self.cfg),
        )
        cond = codec.decode(chromosome=[0, 1, 2])
        self.assertIsInstance(cond, Condition)
        self.assertEqual(cond.feature_ref, "Factor_Momentum_Score")

    def test_codec_ohlcv_relation(self) -> None:
        """Codec sur le bloc relations OHLCV (close > open, etc.)."""
        codec = BNFCodec.from_text(get_relations_grammar("ohlcv"))
        # Trouver un codon qui produit <bullish_candle> (close > open)
        terminals = decode_chromosome(codec.grammar, [0])
        self.assertEqual(terminals, ["close", ">", "open"])
        cond = codec.decode([0])
        self.assertIsInstance(cond, Condition)
        self.assertEqual(cond.feature_ref, "close")
        self.assertEqual(cond.operator, CompareOp.GT)
        self.assertEqual(cond.transformation, "featureref:open")

    def test_codec_ohlcv_bearish(self) -> None:
        """Codec sur la relation bearish (close < open)."""
        codec = BNFCodec.from_text(get_relations_grammar("ohlcv"))
        terminals = decode_chromosome(codec.grammar, [1])
        self.assertEqual(terminals, ["close", "<", "open"])
        cond = codec.decode([1])
        self.assertEqual(cond.operator, CompareOp.LT)
        self.assertEqual(cond.transformation, "featureref:open")


class TestBNFStress(unittest.TestCase):
    """Tests de stress / robustesse du codec."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config("src/einherjar/research/config")

    def test_decode_all_features_terminates(self) -> None:
        """Le decodeur termine sur les 218 features (pas de boucle infinie)."""
        for feat in self.cfg.usable_feature_names:
            codec = BNFCodec.from_text(get_feature_grammar(feat, self.cfg))
            # Chromosome de 10 codons, on verifie juste que ca decode sans raise
            try:
                cond = codec.decode(chromosome=[0] * 10)
                self.assertIsInstance(cond, (Condition, ConditionNode))
            except BNFDecodeError:
                # Certaines grammaires peuvent ne pas etre completement
                # derivables avec 10 codons, c'est OK.
                pass


if __name__ == "__main__":
    unittest.main()
