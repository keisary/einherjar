import sys
sys.path.insert(0, r"D:/midas_v2/einherjar/src/einherjar/research")

# Vérifier les signatures
print("=== ValidationEvaluator ===")
from validation.evaluator import ValidationEvaluator
import inspect
print(inspect.signature(ValidationEvaluator.__init__))

print("\n=== ExecutionEngine ===")
from execution.executor import ExecutionEngine
print(inspect.signature(ExecutionEngine.__init__))

print("\n=== PortfolioSelector ===")
from portfolio.selector import PortfolioSelector
print(inspect.signature(PortfolioSelector.__init__))

print("\n=== PortfolioAllocator ===")
from portfolio.allocator import PortfolioAllocator
print(inspect.signature(PortfolioAllocator.__init__))

print("\n=== PortfolioReporter ===")
from portfolio.portfolio_report import PortfolioReporter
print(inspect.signature(PortfolioReporter.__init__))

print("\n=== Explorer ===")
from discovery.explorer import Explorer
print(inspect.signature(Explorer.__init__))

print("\n=== DiscoveryGenerator ===")
from discovery.generator import DiscoveryGenerator
print(inspect.signature(DiscoveryGenerator.__init__))
