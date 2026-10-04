"""Konstanten der SPT-Demo: beide Vehikel (Neutral, Werkstatt/Logistik), Regler, Messreihen-Seeds."""

N_MIN, N_MAX, DEFAULT_N, N_STEP = 2, 60, 20, 1
SEED_MAX = 999999
DEFAULT_SEED = 35
SWEEP_SEEDS = tuple(range(100000, 100005))
SWEEP_CHAINS = 3                 # weitere Zufalls-Ketten je fester Instanz in Sweeps

# Bearbeitungszeiten
P_MIN, P_MAX = 1, 100

# Fälligkeiten (TF/RDD-Schema, Potts & Van Wassenhove-artig - für Stück 1 unbenutzt, für Stück 2 (EDD)
# vorbereitet; Formel vor Stück 2 per WebSearch zu verifizieren)
DEFAULT_TF, DEFAULT_RDD = 0.4, 0.6

# Gewichte (für gewichtete Folgestücke vorbereitet)
WEIGHT_VALUES = (1, 2, 4)
WEIGHT_PROBS = (0.5, 0.3, 0.2)

# Brute-Force-Vollaufzählung nur bis zu dieser Größe live in der App (9! = 362880, noch < 1 s)
BRUTE_FORCE_MAX_N = 9
BRUTE_FORCE_SWEEP_N = (2, 3, 4, 5, 6, 7, 8, 9)

# --- Vehikel B "Werkstatt/Logistik" ---------------------------------------------------------------------------
N_FAMILIES_MIN, N_FAMILIES_MAX, DEFAULT_N_FAMILIES = 2, 6, 3
SETUP_TIME_MIN, SETUP_TIME_MAX, DEFAULT_SETUP_TIME = 0, 60, 15
SHIFT_LENGTH_MIN, SHIFT_LENGTH_MAX, DEFAULT_SHIFT_LENGTH = 120, 960, 480

VEHICLE_LABELS = {"neutral": "Neutral", "logistik": "Werkstatt/Logistik"}
DEFAULT_VEHICLE = "neutral"


def _preset(n=DEFAULT_N, vehicle=DEFAULT_VEHICLE, setup_time=DEFAULT_SETUP_TIME, n_families=DEFAULT_N_FAMILIES):
    return {"n": n, "seed": DEFAULT_SEED, "chain_seed": 0, "vehicle": vehicle, "setup_time": setup_time, "n_families": n_families}


PRESETS = {
    "Standardfall (Voreinstellung)": _preset(),
    "Kleine Instanz (Brute-Force sichtbar)": _preset(n=BRUTE_FORCE_MAX_N),
    "Große Instanz (Skalierung)": _preset(n=N_MAX),
    "Werkstatt/Logistik-Vehikel": _preset(vehicle="logistik"),
    "Hohe Rüstlast (Werkstatt)": _preset(vehicle="logistik", setup_time=SETUP_TIME_MAX),
}
# Mittel über die fünf festen Sweep-Instanzen (Seeds 100000-100004, je drei Ketten-Seeds); Abstand = Prozent über SPT
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "20 Aufträge: Die längste-zuerst-Reihenfolge (LPT) liegt im Mittel 97.6 % und eine zufällige Reihenfolge 49.7 % über SPT - und SPT ist für diese Zielfunktion beweisbar die beste überhaupt.",
    "Kleine Instanz (Brute-Force sichtbar)": f"{BRUTE_FORCE_MAX_N} Aufträge: hier läuft die Vollaufzählung aller {BRUTE_FORCE_MAX_N}! Reihenfolgen live mit - SPT trifft auf jeder getesteten Instanz exakt das Minimum.",
    "Große Instanz (Skalierung)": f"{N_MAX} Aufträge: SPT bleibt weiterhin beweisbar optimal und braucht nur eine Sortierung (O(n log n)) - eine Vollaufzählung wäre bei dieser Größe aussichtslos.",
    "Werkstatt/Logistik-Vehikel": "Dieselben Aufträge, aber in Familien mit Rüstzeit beim Wechsel - SPT kennt diese Rüstzeiten nicht und bleibt dadurch nicht mehr beweisbar optimal.",
    "Hohe Rüstlast (Werkstatt)": f"Rüstzeit {SETUP_TIME_MAX} Minuten je Familienwechsel: SPT (ignoriert Rüstzeiten) liegt messbar über dem echten Optimum - der Abstand wächst mit der Rüstzeit.",
}
