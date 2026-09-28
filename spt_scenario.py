"""Vehikel A "Neutral" der Scheduling-Theorie-Linie: n Aufträge mit Bearbeitungszeit, Fälligkeit und Gewicht auf
EINER Maschine. Bearbeitungszeit ist für dieses Stück (SPT, 1||ΣCⱼ) die einzig verwendete Größe; Fälligkeit und
Gewicht sind für die Folgestücke (EDD, Moore-Hodgson, WSPT, gewichtete Verspätung) vorbereitet, die dasselbe
Vehikel wortgleich weiterverwenden - wie jede andere Konzepte-Linie dieser Website.

Fälligkeiten nach dem literaturüblichen TF/RDD-Schema (Tardiness-Faktor, Range der Fälligkeiten; Potts & Van
Wassenhove-artig): die mittlere Fälligkeit liegt bei `P * (1 - TF)` (P = Summe aller Bearbeitungszeiten, die
Fertigstellungszeit des letzten Auftrags EGAL in welcher Reihenfolge), gestreut über `RDD * P`. Für dieses Stück
unbenutzt - vor Stück 2 (EDD) per WebSearch exakt zu verifizieren, nicht blind aus dem Gedächtnis übernehmen."""

from dataclasses import dataclass

import numpy as np

import spt_constants as C


@dataclass(frozen=True)
class Instance:
    n: int
    p: np.ndarray        # Bearbeitungszeiten
    d: np.ndarray        # Fälligkeiten (für Folgestücke)
    w: np.ndarray        # Gewichte (für Folgestücke)
    seed: int
    tf: float
    rdd: float


def generate(n, seed, tf=C.DEFAULT_TF, rdd=C.DEFAULT_RDD, p_min=C.P_MIN, p_max=C.P_MAX):
    rng = np.random.default_rng(seed)
    p = rng.integers(p_min, p_max + 1, size=n).astype(np.int64)
    total = int(p.sum())
    frac = (1 - tf - rdd / 2) + rdd * rng.random(n)
    d = np.maximum(np.round(total * frac).astype(np.int64), p)
    w = rng.choice(C.WEIGHT_VALUES, size=n, p=C.WEIGHT_PROBS).astype(np.int64)
    return Instance(n, p, d, w, seed, tf, rdd)
