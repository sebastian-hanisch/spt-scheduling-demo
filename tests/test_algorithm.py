"""spt_algorithm: SPT-Optimalität gegen unabhängige Brute-Force-Vollaufzählung (Vertauschungsargument-Beweis
empirisch geprüft, nicht nur behauptet), Regressionsschutz, Determinismus, Rüstzeit-Variante."""

import itertools

import numpy as np
import pytest

import spt_algorithm as A


def _p(seed, n):
    rng = np.random.default_rng(seed)
    return rng.integers(1, 100, size=n).astype(np.int64)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
def test_spt_matches_brute_force_for_every_seed(n):
    for seed in range(10):
        p = _p(seed, n)
        assert A.spt(p).total == pytest.approx(A.brute_force_optimal(p).total)


def test_spt_is_the_unique_optimum_up_to_ties():
    """Unabhängige Nachrechnung: unter ALLEN Permutationen ist SPTs Zielfunktionswert das Minimum - nicht nur
    irgendeine gute Lösung."""
    p = _p(7, 6)
    spt_total = A.spt(p).total
    all_totals = [A.evaluate_order(p, perm).total for perm in itertools.permutations(range(6))]
    assert spt_total == pytest.approx(min(all_totals))


def test_completion_times_are_the_cumulative_sum_in_order():
    p = np.array([5, 2, 8, 1])
    order = np.array([3, 1, 0, 2])          # sortiert nach p: 1, 2, 5, 8
    result = A.evaluate_order(p, order)
    assert result.completion.tolist() == [1, 3, 8, 16]
    assert result.total == pytest.approx(1 + 3 + 8 + 16)


def test_spt_order_is_ascending_by_processing_time():
    p = np.array([5, 2, 8, 1, 2])
    order = A.spt_order(p)
    assert p[order].tolist() == sorted(p.tolist())


def test_lpt_order_is_the_reverse_of_spt():
    p = _p(3, 10)
    assert A.lpt_order(p).tolist() == A.spt_order(p).tolist()[::-1]


def test_random_order_is_deterministic_given_the_rng_state():
    n = 8
    a = A.random_order(n, np.random.default_rng(0))
    b = A.random_order(n, np.random.default_rng(0))
    assert a.tolist() == b.tolist()
    assert sorted(a.tolist()) == list(range(n))


def test_spt_beats_a_bad_order_on_a_hand_picked_instance():
    """Handrechnung: drei Aufträge, SPT-Reihenfolge von Hand nachgerechnet."""
    p = np.array([3, 1, 2])
    result = A.spt(p)
    assert result.order.tolist() == [1, 2, 0]                 # 1, 2, 3
    assert result.completion.tolist() == [1, 3, 6]
    assert result.total == pytest.approx(10.0)
    worst = A.evaluate_order(p, np.array([0, 2, 1]))            # 3, 2, 1 (LPT)
    assert worst.total == pytest.approx(3 + 5 + 6)
    assert result.total < worst.total


# --- Mit Rüstzeiten (Vehikel B) --------------------------------------------------------------------------------


def test_setup_variant_matches_the_plain_variant_when_setup_is_zero():
    p = _p(11, 6)
    family = np.array([0, 1, 0, 1, 0, 1])
    setup = np.zeros((2, 2))
    plain = A.spt(p)
    with_setup = A.evaluate_order_with_setup(p, family, setup, plain.order)
    assert with_setup.total == pytest.approx(plain.total)


def test_setup_time_is_only_charged_on_a_family_change():
    p = np.array([2, 2, 2])
    family = np.array([0, 0, 1])
    setup = np.array([[0, 10], [10, 0]])
    order = np.array([0, 1, 2])                                 # kein Wechsel, dann ein Wechsel
    result = A.evaluate_order_with_setup(p, family, setup, order)
    assert result.completion.tolist() == [2, 4, 4 + 10 + 2]


def test_brute_force_with_setup_matches_independent_full_enumeration():
    p = _p(13, 5)
    family = np.array([0, 1, 0, 1, 2])
    setup = np.array([[0, 5, 8], [5, 0, 3], [8, 3, 0]])
    best = A.brute_force_optimal_with_setup(p, family, setup)
    all_totals = [A.total_completion_with_setup(p, family, setup, np.array(perm)) for perm in itertools.permutations(range(5))]
    assert best.total == pytest.approx(min(all_totals))


def test_ignoring_setup_can_be_worse_than_the_true_optimum():
    """SPT (sortiert nur nach Bearbeitungszeit) muss nicht optimal bleiben, sobald Rüstzeiten dazukommen -
    genau die Frage, die Vehikel B stellt."""
    p = np.array([1, 1, 10, 10])
    family = np.array([0, 1, 0, 1])
    setup = np.array([[0, 100], [100, 0]])
    spt_total = A.evaluate_order_with_setup(p, family, setup, A.spt_order(p)).total
    true_opt = A.brute_force_optimal_with_setup(p, family, setup).total
    assert spt_total > true_opt + 1e-6
