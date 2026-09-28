"""Vehikel A (Neutral) und Vehikel B (Werkstatt/Logistik): Erzeugung, Determinismus; Auswertung: Kennzahlen,
Sweeps, Optimalitäts- und Timing-Messreihe, Vehikel-B-Härtetest."""

from dataclasses import replace

import numpy as np
import pytest

import spt_constants as C
import spt_evaluation as ev
import spt_scenario as S
import spt_scenario_logistik as SL


# --- Vehikel A ----------------------------------------------------------------------------------------------------------------------------------


def test_instance_shape_and_bounds():
    inst = S.generate(20, 3)
    assert inst.n == 20 and inst.p.shape == (20,) and inst.d.shape == (20,) and inst.w.shape == (20,)
    assert inst.p.min() >= C.P_MIN and inst.p.max() <= C.P_MAX
    assert (inst.d >= inst.p).all()                             # Fälligkeit nie vor der eigenen Bearbeitungszeit
    assert set(inst.w.tolist()) <= set(C.WEIGHT_VALUES)


def test_instance_is_deterministic_and_seed_dependent():
    a, b, c = S.generate(30, 5), S.generate(30, 5), S.generate(30, 6)
    assert np.array_equal(a.p, b.p) and np.array_equal(a.d, b.d) and np.array_equal(a.w, b.w)
    assert not np.array_equal(a.p, c.p)


# --- Vehikel B ------------------------------------------------------------------------------------------------------------------------------


def test_logistik_instance_shape_and_setup_matrix():
    linst = SL.generate(15, 4, n_families=3, setup_time=15)
    assert linst.n == 15 and linst.family.shape == (15,)
    assert set(linst.family.tolist()) <= set(range(3))
    assert linst.setup.shape == (3, 3)
    assert np.all(np.diag(linst.setup) == 0)
    assert np.all(linst.setup[~np.eye(3, dtype=bool)] == 15)


def test_logistik_instance_shares_the_same_processing_times_as_neutral_for_the_same_seed():
    """Vehikel B ändert nur Familie/Rüstzeiten, nicht die Grunddaten - sonst wäre ein Vehikel-Vergleich nicht fair."""
    neutral = S.generate(20, 7)
    logistik = SL.generate(20, 7)
    assert np.array_equal(neutral.p, logistik.p) and np.array_equal(neutral.d, logistik.d) and np.array_equal(neutral.w, logistik.w)


def test_logistik_instance_is_deterministic():
    a, b = SL.generate(10, 2), SL.generate(10, 2)
    assert np.array_equal(a.family, b.family) and np.array_equal(a.setup, b.setup)


# --- Analyse --------------------------------------------------------------------------------------------------------------------------------


def test_analysis_fields_are_consistent():
    a = ev.analyse(ev.Settings(n=20))
    assert a.spt.total <= a.lpt.total
    assert a.gap_lpt >= 0.0 and a.gap_random >= -1e-6
    assert a.optimal is None                                    # n=20 > BRUTE_FORCE_MAX_N


def test_analysis_matches_the_optimum_for_small_n():
    a = ev.analyse(ev.Settings(n=6))
    assert a.optimal is not None
    assert a.spt_matches_optimum


def test_analysis_is_deterministic_given_the_chain_seed():
    s = ev.Settings(n=20, seed=1, chain_seed=0)
    a, b, c = ev.analyse(s), ev.analyse(s), ev.analyse(replace(s, chain_seed=1))
    assert a.gap_random == pytest.approx(b.gap_random)
    assert a.gap_random != pytest.approx(c.gap_random)


# --- Sweeps und Messreihe -------------------------------------------------------------------------------------------------------------------


def test_run_config_counts_runs_and_aggregates():
    r = ev.run_config(ev.Settings(n=15))
    assert r["n_runs"] == len(C.SWEEP_SEEDS) * C.SWEEP_CHAINS
    assert r["gap_lpt"] > 0.0


def test_sweep_values_labels_and_ordering():
    assert set(ev.SWEEP_VALUES) == set(ev.SWEEP_LABELS)
    rows = ev.sweep("n", ev.Settings(), (5, 40))
    assert [r["value"] for r in rows] == [5, 40]


def test_optimality_check_always_matches():
    """SPT muss auf jeder getesteten Größe und jedem Seed exakt mit der Vollaufzählung übereinstimmen - sonst
    ist entweder der Beweis oder die Implementierung falsch."""
    rows = ev.optimality_check(ns=(3, 4, 5), seeds=C.SWEEP_SEEDS)
    assert all(r["match_rate"] == 1.0 for r in rows)


def test_timing_sweep_shows_brute_force_growing_far_faster_than_spt():
    rows = ev.timing_sweep(ns=(4, 8))
    small, large = rows[0], rows[1]
    assert large["brute_force_seconds"] > small["brute_force_seconds"] * 10
    assert large["spt_seconds"] < large["brute_force_seconds"] / 100


def test_setup_gap_is_zero_when_setup_time_is_zero():
    """Vehikel B mit Rüstzeit 0 muss exakt auf Vehikel A zusammenfallen - Konsistenz-Test wie bei jeder
    'kollabiert zum Spezialfall'-Erweiterung dieser Website."""
    row = ev.setup_gap(n=6, setup_time=0)
    assert row["gap_mean"] == pytest.approx(0.0, abs=1e-6)


def test_setup_gap_grows_with_the_setup_time():
    small = ev.setup_gap(n=6, setup_time=5)
    large = ev.setup_gap(n=6, setup_time=60)
    assert large["gap_mean"] > small["gap_mean"]
