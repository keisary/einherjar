import io, os, re, sys
from datetime import datetime

# Diagnostics : tracer les logs de pipeline pour détecter les arrêts anormaux
# (crash silencieux) et les évaluations incomplètes dans les 4 runs TF.

def analyse_log(path: str) -> None:
    if not os.path.exists(path):
        print(f"  [ABSENT] {path}")
        return
    with io.open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    lines = text.splitlines()
    n_ok = len(re.findall(r"test_on\(val\) OK", text))
    n_echec = len(re.findall(r"Échec évaluation", text)) + len(re.findall(r"Aucun signal", text))
    n_step = re.findall(r"\[STEP (\d)\]", text)
    steps = list(dict.fromkeys(n_step))
    last = lines[-1][:140] if lines else "(vide)"
    n_test_start = len(re.findall(r"test_on\(val\) :", text))
    print(f"  {os.path.basename(path)}: steps={steps} evalOK={n_ok} echec={n_echec} "
          f"test_on_start={n_test_start} lignes={len(lines)}")
    print(f"    dernier: {last}")
    if n_test_start > n_ok + n_echec:
        print("    !! EVAL NON TERMINEE EN FIN DE LOG (crash potentiel)")

for d in sorted(os.listdir("outputs"), reverse=True):
    if not d.startswith("run_"):
        continue
    p = os.path.join("outputs", d, "run.log")
    analyse_log(p)