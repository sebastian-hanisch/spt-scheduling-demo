"""Jede Zahl der App-Texte ist hier über die fünf festen Sweep-Instanzen (je drei Ketten) belegt. Positive UND
negative Aussagen: SPT ist beweisbar optimal auf dem neutralen Vehikel - UND hört auf, beweisbar optimal zu sein,
sobald Rüstzeiten (Vehikel Werkstatt/Logistik) dazukommen. Rechenzeiten nur als Größenordnung geprüft."""

from functools import lru_cache

import pytest

import spt_evaluation as ev


@lru_cache(maxsize=None)
def _cfg(items):
    return ev.run_config(ev.Settings(), **dict(items))


def cfg(**kw):
    return _cfg(tuple(sorted(kw.items())))


def near(value, expected, tol):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


# --- Standardfall -------------------------------------------------------------------------------------------------------------------------------


def test_standard_case_numbers():
    std = cfg()
    near(std["gap_lpt"], 97.6, 15.0)
    near(std["gap_random"], 49.7, 10.0)


def test_spt_is_never_worse_than_lpt_or_random_on_any_swept_configuration():
    for n in (2, 5, 10, 20, 40, 60):
        row = cfg(n=n)
        assert row["gap_lpt"] >= 0.0 and row["gap_random"] >= 0.0


# --- Optimalität gegen Brute-Force ---------------------------------------------------------------------------------------------------------------


def test_spt_matches_brute_force_on_every_tested_size():
    rows = ev.optimality_check()
    assert all(r["match_rate"] == 1.0 for r in rows)


# --- Timing: n! gegen n log n -----------------------------------------------------------------------------------------------------------------


def test_brute_force_grows_far_faster_than_spt():
    rows = ev.timing_sweep()
    small, large = rows[0], rows[-1]
    assert large["brute_force_seconds"] > small["brute_force_seconds"] * 100
    assert large["spt_seconds"] < 0.01                            # SPT bleibt praktisch konstant


def test_brute_force_becomes_impractical_around_nine_jobs():
    rows = ev.timing_sweep()
    last = rows[-1]
    assert last["value"] == 9
    assert last["brute_force_seconds"] > 0.2                       # spürbar langsamer als ein Klick


# --- Vehikel B: Rüstzeit-Härtetest ------------------------------------------------------------------------------------------------------------


def test_setup_gap_is_exactly_zero_without_setup_time():
    row = ev.setup_gap(setup_time=0)
    near(row["gap_mean"], 0.0, 1e-6)


@pytest.mark.parametrize("setup_time,gap,tol", [(5, 1.6, 3.0), (15, 6.6, 5.0), (30, 14.3, 8.0), (60, 28.9, 12.0)])
def test_setup_gap_numbers(setup_time, gap, tol):
    row = ev.setup_gap(setup_time=setup_time)
    near(row["gap_mean"], gap, tol)


def test_setup_gap_grows_monotonically_with_the_setup_time():
    values = [ev.setup_gap(setup_time=s)["gap_mean"] for s in (0, 15, 30, 60)]
    assert values == sorted(values)


def test_spt_on_the_logistik_vehicle_can_be_strictly_worse_than_the_true_optimum():
    """Die zentrale Vehikel-B-Aussage: SPT bleibt hier NICHT beweisbar optimal - das ist ein echter Befund, kein
    Dekor."""
    row = ev.setup_gap(setup_time=60)
    assert row["gap_max"] > 1.0
