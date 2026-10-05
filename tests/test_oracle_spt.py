"""Orakel-Test: SPT gegen die geschlossene Formel (kleinste Zeit bekommt den größten Faktor), Permutations-
Aufzählung mit eigener Summenbildung und die Rüstzeit-Variante gegen eine Held-Karp-DP (Restanzahl-Gewichtung)."""

import itertools
import random

import numpy as np

import spt_algorithm as A


def _total(p, order, fam=None, setup=None):
    t = tot = 0
    prev = None
    for j in order:
        if fam is not None and prev is not None:
            t += int(setup[prev][fam[j]])
        t += int(p[j])
        prev = fam[j] if fam is not None else None
        tot += t
    return tot


def _opt_closed(p):
    ps = sorted(int(x) for x in p)
    return sum((len(ps) - k) * ps[k] for k in range(len(ps)))


def _opt_dp_setup(p, fam, setup):
    n = len(p)
    inf = 10 ** 12
    g = [[inf] * n for _ in range(1 << n)]
    for j in range(n):
        g[1 << j][j] = n * int(p[j])
    for mask in range(1, 1 << n):
        k = bin(mask).count("1")
        for last in range(n):
            v = g[mask][last]
            if v >= inf:
                continue
            for j in range(n):
                if mask >> j & 1:
                    continue
                inc = int(setup[fam[last]][fam[j]]) + int(p[j])
                nm = mask | 1 << j
                g[nm][j] = min(g[nm][j], v + (n - k) * inc)
    return min(g[(1 << n) - 1])


def test_spt_equals_closed_form_and_enumerated_optimum():
    rng = random.Random(2)
    for _ in range(150):
        n = rng.randint(1, 6)
        hi = rng.choice([2, 5, 100])
        p = np.array([rng.randint(1, hi) for _ in range(n)])
        res = A.spt(p)
        assert res.total == _opt_closed(p) == _total(p, res.order)
        assert min(_total(p, perm) for perm in itertools.permutations(range(n))) == res.total
        assert A.brute_force_optimal(p).total == res.total
        assert A.total_completion(p, A.lpt_order(p)) == _total(p, A.lpt_order(p))


def test_setup_variant_matches_independent_total_and_held_karp_dp():
    rng = random.Random(4)
    for _ in range(60):
        n = rng.randint(1, 6)
        p = np.array([rng.randint(1, 30) for _ in range(n)])
        fam = np.array([rng.randint(0, 2) for _ in range(n)])
        s = rng.choice([0, 10, 40])
        setup = np.array([[0 if a == b else s for b in range(3)] for a in range(3)])
        order = rng.sample(range(n), n)
        assert A.evaluate_order_with_setup(p, fam, setup, order).total == _total(p, order, fam, setup)
        assert A.brute_force_optimal_with_setup(p, fam, setup).total == _opt_dp_setup(p, fam, setup)
