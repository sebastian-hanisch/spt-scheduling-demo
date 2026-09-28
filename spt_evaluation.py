"""Auswertung der SPT-Demo: SPT gegen die längste-zuerst-Reihenfolge (LPT) und gegen zufällige Reihenfolgen,
gegen die Brute-Force-Vollaufzählung (nur kleine n), und das Vehikel-B-Experiment (bleibt SPT nahe am Optimum,
sobald Rüstzeiten zwischen Auftragsfamilien dazukommen)."""

import time
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import spt_algorithm as A
import spt_constants as C
import spt_scenario as S
import spt_scenario_logistik as SL


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED
    chain_seed: int = 0
    tf: float = C.DEFAULT_TF
    rdd: float = C.DEFAULT_RDD
    vehicle: str = C.DEFAULT_VEHICLE
    setup_time: int = C.DEFAULT_SETUP_TIME
    n_families: int = C.DEFAULT_N_FAMILIES


@lru_cache(maxsize=512)
def instance(n, seed, tf=C.DEFAULT_TF, rdd=C.DEFAULT_RDD):
    return S.generate(n, seed, tf=tf, rdd=rdd)


@lru_cache(maxsize=512)
def logistik_instance(n, seed, n_families, setup_time, tf=C.DEFAULT_TF, rdd=C.DEFAULT_RDD):
    return SL.generate(n, seed, n_families=n_families, setup_time=setup_time, tf=tf, rdd=rdd)


@dataclass
class Analysis:
    settings: Settings
    inst: object
    spt: object
    lpt: object
    random_mean: float
    random_runs: int
    optimal: object          # None, wenn n > BRUTE_FORCE_MAX_N

    @property
    def gap_lpt(self):
        return 100.0 * (self.lpt.total - self.spt.total) / self.spt.total

    @property
    def gap_random(self):
        return 100.0 * (self.random_mean - self.spt.total) / self.spt.total

    @property
    def spt_matches_optimum(self):
        return self.optimal is not None and abs(self.spt.total - self.optimal.total) < 1e-6


def analyse(settings, random_draws=20):
    """Wertet SPT auf dem gewählten Vehikel aus - Neutral (Zielfunktion hängt nur von der Reihenfolge ab) oder
    Werkstatt/Logistik (Rüstzeit beim Familienwechsel zählt mit). SPT selbst bleibt in beiden Fällen dieselbe
    Regel (sortiert nur nach Bearbeitungszeit, kennt keine Rüstzeiten) - nur die BEWERTUNG der Reihenfolgen
    (und damit auch der Vollaufzählung) wechselt mit dem Vehikel, damit die Haupt-Kennzahlen ehrlich
    widerspiegeln, was auf dem gewählten Vehikel tatsächlich passiert (statt nur in einer Zusatzbox)."""
    if settings.vehicle == "logistik":
        inst = logistik_instance(settings.n, settings.seed, settings.n_families, settings.setup_time, settings.tf, settings.rdd)
        p, family, setup = inst.p, inst.family, inst.setup

        def ev(order):
            return A.evaluate_order_with_setup(p, family, setup, order)

        optimal = A.brute_force_optimal_with_setup(p, family, setup) if settings.n <= C.BRUTE_FORCE_MAX_N else None
    else:
        inst = instance(settings.n, settings.seed, settings.tf, settings.rdd)
        p = inst.p

        def ev(order):
            return A.evaluate_order(p, order)

        optimal = A.brute_force_optimal(p) if settings.n <= C.BRUTE_FORCE_MAX_N else None

    spt = ev(A.spt_order(p))
    lpt = ev(A.lpt_order(p))
    rng = np.random.default_rng(settings.chain_seed)
    random_totals = [ev(A.random_order(settings.n, rng)).total for _ in range(random_draws)]
    return Analysis(settings, inst, spt, lpt, float(np.mean(random_totals)), random_draws, optimal)


# --- Sweeps und Tabellen -----------------------------------------------------------------------------------------------------------------------


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def run_config(base, seeds=C.SWEEP_SEEDS, chains=C.SWEEP_CHAINS, **changes):
    """Mittel über die festen Instanzen und je `chains` Ketten-Seeds für die Einstellungen `base` mit `changes`."""
    s0 = replace(base, **changes)
    rows = []
    for seed in seeds:
        for ch in range(chains):
            a = analyse(replace(s0, seed=seed, chain_seed=ch))
            rows.append({"gap_lpt": a.gap_lpt, "gap_random": a.gap_random})
    out = {k: _mean(rows, k) for k in rows[0]}
    out["n_runs"] = len(rows)
    return out


SWEEP_VALUES = {"n": (2, 5, 10, 20, 40, 60), "tf": (0.0, 0.2, 0.4, 0.6, 0.8), "rdd": (0.2, 0.4, 0.6, 0.8, 1.0)}
SWEEP_LABELS = {"n": "Aufträge", "tf": "Fristen-Anteil (TF)", "rdd": "Fristen-Streuung (RDD)"}


def sweep(param, base=Settings(), values=None):
    values = SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]


def optimality_check(ns=C.BRUTE_FORCE_SWEEP_N, seeds=C.SWEEP_SEEDS):
    """SPT gegen Brute-Force-Vollaufzählung über mehrere n und Instanzen - Anteil exakter Treffer (muss 100 % sein,
    sonst ist der Beweis oder die Implementierung falsch)."""
    rows = []
    for n in ns:
        matches = 0
        for seed in seeds:
            inst = instance(n, seed)
            spt_total = A.spt(inst.p).total
            opt_total = A.brute_force_optimal(inst.p).total
            if abs(spt_total - opt_total) < 1e-6:
                matches += 1
        rows.append({"value": n, "match_rate": matches / len(seeds)})
    return rows


def timing_sweep(ns=C.BRUTE_FORCE_SWEEP_N, seed=C.DEFAULT_SEED):
    """Gemessene Rechenzeit: Brute-Force-Vollaufzählung (O(n!)) gegen SPT (O(n log n))."""
    rows = []
    for n in ns:
        inst = instance(n, seed)
        t0 = time.perf_counter()
        A.brute_force_optimal(inst.p)
        t_bf = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(100):
            A.spt(inst.p)
        t_spt = (time.perf_counter() - t0) / 100
        rows.append({"value": n, "brute_force_seconds": t_bf, "spt_seconds": t_spt})
    return rows


def setup_gap(n=8, seeds=C.SWEEP_SEEDS, n_families=C.DEFAULT_N_FAMILIES, setup_time=C.DEFAULT_SETUP_TIME):
    """Vehikel-B-Härtetest: SPT (sortiert nur nach Bearbeitungszeit, ignoriert Rüstzeiten) gegen die echte
    Optimallösung MIT Rüstzeiten (Brute-Force, deshalb kleines n). Der Abstand ist eine echte Messfrage, kein
    behaupteter Befund."""
    gaps = []
    for seed in seeds:
        linst = SL.generate(n, seed, n_families=n_families, setup_time=setup_time)
        spt_ord = A.spt_order(linst.p)
        spt_total = A.evaluate_order_with_setup(linst.p, linst.family, linst.setup, spt_ord).total
        opt_total = A.brute_force_optimal_with_setup(linst.p, linst.family, linst.setup).total
        gaps.append(100.0 * (spt_total - opt_total) / opt_total)
    return {"gap_mean": float(np.mean(gaps)), "gap_min": float(np.min(gaps)), "gap_max": float(np.max(gaps)), "n_runs": len(gaps)}


def setup_gap_sweep(setup_times=(0, 5, 15, 30, 60), n=8, seeds=C.SWEEP_SEEDS, n_families=C.DEFAULT_N_FAMILIES):
    return [{"value": s, **setup_gap(n=n, seeds=seeds, n_families=n_families, setup_time=s)} for s in setup_times]
