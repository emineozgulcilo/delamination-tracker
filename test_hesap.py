import numpy as np, sys
sys.path.insert(0, r"C:\Users\Süleyman\Desktop\delamınasyon")
from astm_d5528 import method_MBT, method_CC, method_MCC

# GFRP DCB deney verisi (P [N], delta [mm], a [mm])
P     = np.array([35.8,34.2,31.5,29.6,27.8,26.0,24.5,23.1,21.9,20.7,19.6])
delta = np.array([ 5.6, 7.1, 8.6,10.1,11.8,13.7,15.6,17.7,19.9,22.1,24.4])
a_mm  = np.array([50,55,60,65,70,75,80,85,90,95,100], dtype=float)

# SI donusumu
b       = 20e-3     # m
h       = 2e-3      # m
a_m     = a_mm * 1e-3
delta_m = delta * 1e-3

sep = "=" * 55
print(sep)
print("ASTM D5528 GFRP DCB Test - Dogrulama")
print("Numune: b=20mm, 2h=4mm, a0=50mm")
print(sep)

methods = [
    ("MBT", lambda: method_MBT(P, delta_m, a_m, b), "Delta (mm)"),
    ("CC",  lambda: method_CC(P, delta_m, a_m, b),  "n"),
    ("MCC", lambda: method_MCC(P, delta_m, a_m, b, h), "A1"),
]

for name, fn, extra in methods:
    res = fn()
    GIc_Jm2 = res[0]   # already in J/m2 (SI inputs)
    param    = res[1]
    r2       = res[4]
    if name == "MBT":
        param *= 1000  # m -> mm for readability
    avg = np.mean(GIc_Jm2)
    std = np.std(GIc_Jm2, ddof=1)
    cv  = 100*std/avg
    print(f"\n{name}:")
    print(f"  {extra:12s} = {param:.4f}")
    print(f"  R2           = {r2:.4f}")
    print(f"  GIc ort.     = {avg:.1f} J/m2")
    print(f"  GIc std      = {std:.1f} J/m2")
    print(f"  CV           = {cv:.1f}%")
    vals = [f"{v:.1f}" for v in GIc_Jm2]
    print(f"  Degerler     = {vals}")

print(f"\n{sep}")
print("Literatur (GFRP/epoksi): 300-400 J/m2")
print(sep)
