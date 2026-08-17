"""Run the 40-case adversarial suite against the trained SmishGuard pipeline."""
import sys, json
from collections import Counter, defaultdict
sys.path.insert(0, ".")

import joblib
from app.smishing.model import classify
from tests.adversarial_corpus import CASES

PIPE = joblib.load("ml/smishing_model.joblib")
THRESHOLD = 0.5

rows = []
for c in CASES:
    r = classify(PIPE, c["text"])
    risk = r["risk"]
    flagged = risk >= THRESHOLD
    should_flag = c["label"] == "smish"
    rows.append(dict(
        id=c["id"], label=c["label"], family=c["family"],
        pred=r["label"], risk=risk, conf=r["confidence"],
        flagged=flagged,
        correct=(flagged == should_flag),
        signals=[s for s in r["risk_signals"] if s["present"]],
        top=r["top_tokens"],
        text=c["text"], note=c["note"],
    ))

# ---- summary -------------------------------------------------------
tp = sum(1 for r in rows if r["label"] == "smish" and r["flagged"])
fn = sum(1 for r in rows if r["label"] == "smish" and not r["flagged"])
fp = sum(1 for r in rows if r["label"] == "genuine" and r["flagged"])
tn = sum(1 for r in rows if r["label"] == "genuine" and not r["flagged"])
prec = tp / (tp + fp) if tp + fp else 0
rec = tp / (tp + fn) if tp + fn else 0
f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0

print(f"\n=== RESULTS @ risk>={THRESHOLD} ===")
print(f"TP {tp}  FN {fn}  FP {fp}  TN {tn}")
print(f"recall (catch rate) {rec:.3f}   precision {prec:.3f}   F1 {f1:.3f}")
print(f"accuracy {(tp+tn)/len(rows):.3f}")

print("\n--- MISSED SMISHING (false negatives) ---")
for r in sorted([r for r in rows if r["label"] == "smish" and not r["flagged"]],
                key=lambda r: r["risk"]):
    print(f"  {r['id']} risk={r['risk']:.3f} pred={r['pred']:<22} {r['family']}")

print("\n--- FALSE ALARMS ON GENUINE ---")
for r in sorted([r for r in rows if r["label"] == "genuine" and r["flagged"]],
                key=lambda r: -r["risk"]):
    print(f"  {r['id']} risk={r['risk']:.3f} pred={r['pred']:<22} {r['family']}")

print("\n--- CAUGHT SMISHING ---")
for r in sorted([r for r in rows if r["label"] == "smish" and r["flagged"]],
                key=lambda r: -r["risk"]):
    print(f"  {r['id']} risk={r['risk']:.3f} pred={r['pred']:<22} {r['family']}")

print("\n--- CORRECTLY CLEARED GENUINE ---")
for r in sorted([r for r in rows if r["label"] == "genuine" and not r["flagged"]],
                key=lambda r: -r["risk"]):
    print(f"  {r['id']} risk={r['risk']:.3f} {r['family']}")

# ---- threshold sweep ----------------------------------------------
print("\n=== THRESHOLD SWEEP ===")
print(f"{'thr':>5} {'recall':>7} {'prec':>7} {'F1':>7} {'FP':>4} {'FN':>4}")
for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    _tp = sum(1 for r in rows if r["label"] == "smish" and r["risk"] >= t)
    _fp = sum(1 for r in rows if r["label"] == "genuine" and r["risk"] >= t)
    _fn = 20 - _tp
    _p = _tp / (_tp + _fp) if _tp + _fp else 0
    _r = _tp / 20
    _f = 2 * _p * _r / (_p + _r) if _p + _r else 0
    print(f"{t:>5.1f} {_r:>7.3f} {_p:>7.3f} {_f:>7.3f} {_fp:>4} {_fn:>4}")

# ---- risk distribution --------------------------------------------
print("\n=== RISK SPREAD ===")
for lab in ("smish", "genuine"):
    rs = sorted(r["risk"] for r in rows if r["label"] == lab)
    print(f"  {lab:<8} min={rs[0]:.3f} med={rs[len(rs)//2]:.3f} max={rs[-1]:.3f}")

# ---- signal firing rates ------------------------------------------
print("\n=== ENGINEERED SIGNAL FIRING ===")
fire = defaultdict(lambda: [0, 0])
for r in rows:
    i = 0 if r["label"] == "smish" else 1
    for s in r["signals"]:
        fire[s["label"]][i] += 1
print(f"{'signal':<58} {'smish/20':>9} {'genuine/20':>11}")
for k, (a, b) in sorted(fire.items(), key=lambda kv: -kv[1][0]):
    print(f"{k:<58} {a:>9} {b:>11}")
never = [v for v in __import__("app.smishing.model", fromlist=["x"]).FEATURE_LABELS.values()
         if v not in fire]
print("\nsignals that NEVER fired on any of the 40 cases:")
for n in never:
    print("  -", n)

json.dump(rows, open("results.json", "w"), indent=1)
print("\nwrote results.json")
