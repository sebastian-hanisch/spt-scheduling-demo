"""SPT (Shortest Processing Time first) für 1||ΣCⱼ: n Aufträge auf einer Maschine, Ziel ist die Summe der
Fertigstellungszeiten zu minimieren. SPT (aufsteigend nach Bearbeitungszeit sortieren) ist dafür beweisbar
optimal - Vertauschungsargument (siehe README/App für den Beweis): tauscht man zwei benachbarte Aufträge i, j
mit pᵢ > pⱼ (i steht vor j), sinkt die Summe um genau pᵢ - pⱼ: nur die beiden Fertigstellungszeiten ändern sich,
alle Aufträge davor und danach behalten ihre - eine Reihenfolge, die nicht sortiert ist, hat also immer eine
benachbarte Vertauschung, die verbessert; SPT ist (bis auf Gleichstände) der einzige Fixpunkt.

Hier zusätzlich: Brute-Force-Vollaufzählung als unabhängige Gegenprobe (nur für kleine n praktikabel), sowie die
Rüstzeit-Variante für Vehikel B (Werkstatt/Logistik) - SPT selbst kennt keine Rüstzeiten, die Frage, wie groß der
Fehler dadurch wird, ist Gegenstand der Messreihe, nicht vorab behauptet."""

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass
class Result:
    order: np.ndarray
    completion: np.ndarray
    total: float


def completion_times(p, order):
    return np.cumsum(np.asarray(p)[order].astype(np.float64))


def total_completion(p, order):
    return float(completion_times(p, order).sum())


def evaluate_order(p, order):
    order = np.asarray(order)
    return Result(order, completion_times(p, order), total_completion(p, order))


def spt_order(p):
    """SPT: aufsteigend nach Bearbeitungszeit; stabile Sortierung, damit Gleichstände reproduzierbar sind."""
    return np.argsort(p, kind="stable")


def spt(p):
    return evaluate_order(p, spt_order(p))


def lpt_order(p):
    """Längste zuerst - das Gegenteil von SPT, als "wie schlecht kann es werden"-Vergleich."""
    return np.argsort(p, kind="stable")[::-1]


def random_order(n, rng):
    order = np.arange(n)
    rng.shuffle(order)
    return order


def brute_force_optimal(p):
    """Volle Aufzählung aller n! Reihenfolgen - unabhängige Gegenprobe, nur für kleine n (siehe
    spt_constants.BRUTE_FORCE_MAX_N)."""
    n = len(p)
    best_order, best_total = None, np.inf
    for perm in itertools.permutations(range(n)):
        order = np.array(perm)
        total = total_completion(p, order)
        if total < best_total:
            best_total, best_order = total, order
    return evaluate_order(p, best_order)


# --- Mit Rüstzeiten (Vehikel B: Werkstatt/Logistik) ---------------------------------------------------------


def completion_times_with_setup(p, family, setup, order):
    t = 0.0
    out = np.empty(len(order), dtype=np.float64)
    prev_family = None
    for idx, j in enumerate(order):
        if prev_family is not None:
            t += float(setup[prev_family, family[j]])
        t += float(p[j])
        out[idx] = t
        prev_family = family[j]
    return out


def total_completion_with_setup(p, family, setup, order):
    return float(completion_times_with_setup(p, family, setup, order).sum())


def evaluate_order_with_setup(p, family, setup, order):
    order = np.asarray(order)
    return Result(order, completion_times_with_setup(p, family, setup, order), total_completion_with_setup(p, family, setup, order))


def brute_force_optimal_with_setup(p, family, setup):
    n = len(p)
    best_order, best_total = None, np.inf
    for perm in itertools.permutations(range(n)):
        order = np.array(perm)
        total = total_completion_with_setup(p, family, setup, order)
        if total < best_total:
            best_total, best_order = total, order
    return evaluate_order_with_setup(p, family, setup, best_order)
