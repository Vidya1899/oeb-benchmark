"""
Reference implementation of the OEB metrics.

Both metrics use natural logarithms, matching the paper:
  - BSE is in nats and bounded by ln 2.
  - G_js = 1 - JSD / ln 2, in [0, 1].

A "direction distribution" here is a length-2 array [P(increase), P(decrease)]
obtained by projecting the model's first-token probability mass onto the
{increase, decrease} token clusters and renormalizing so the two sum to 1.
"""

from __future__ import annotations
import numpy as np

LN2 = np.log(2.0)
_EPS = 1e-12


def _clean(p):
    """Validate and normalize a 2-element direction distribution."""
    p = np.asarray(p, dtype=float)
    if p.shape != (2,):
        raise ValueError(f"expected a length-2 distribution, got shape {p.shape}")
    p = np.clip(p, _EPS, None)
    return p / p.sum()


def binary_semantic_entropy(p_dir) -> float:
    """Shannon entropy (nats) of a binary {increase, decrease} distribution.

    Returns a value in [0, ln 2]. Low BSE + wrong prediction => confident
    prior-driven error; high BSE => epistemic conflict.
    """
    p = _clean(p_dir)
    return float(-np.sum(p * np.log(p)))


def _kl(p, q):
    return float(np.sum(p * np.log(p / q)))


def jensen_shannon_divergence(p, q) -> float:
    """JSD (nats) between two direction distributions. Bounded by ln 2."""
    p, q = _clean(p), _clean(q)
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def semantic_gravity(p_fwd, p_rev) -> float:
    """G_js = 1 - JSD(P_rev, P_fwd) / ln 2, in [0, 1].

    G_js -> 1: output invariant under reversal (script retrieval / high gravity).
    G_js -> 0: output shifts under reversal (perceptual sensitivity).

    NOTE: G_js measures the *magnitude* of the shift, not its correctness.
    Whether the reversed prediction is right is a separate quantity (Acc_rev).
    """
    jsd = jensen_shannon_divergence(p_fwd, p_rev)
    return float(1.0 - jsd / LN2)


# Regime cutoffs from the paper.
GROUNDING_MAX = 0.1   # G_js < 0.1  -> Grounding
RETRIEVAL_MIN = 0.9   # G_js > 0.9  -> Retrieval


def regime(g_js: float) -> str:
    """Map a G_js value to its behavioral regime."""
    if g_js < GROUNDING_MAX:
        return "grounding"
    if g_js > RETRIEVAL_MIN:
        return "retrieval"
    return "conflict"


def regime_masses(g_js_values) -> dict:
    """Fraction of clips in each regime (percentages, summing to 100)."""
    g = np.asarray(g_js_values, dtype=float)
    n = len(g)
    if n == 0:
        return {"grounding": 0.0, "conflict": 0.0, "retrieval": 0.0}
    counts = {"grounding": 0, "conflict": 0, "retrieval": 0}
    for v in g:
        counts[regime(v)] += 1
    return {k: 100.0 * c / n for k, c in counts.items()}


if __name__ == "__main__":
    # Tiny self-check.
    invariant_f, invariant_r = [0.99, 0.01], [0.99, 0.01]   # no shift
    flipped_f, flipped_r = [0.99, 0.01], [0.01, 0.99]        # full shift
    print("G_js (invariant, expect ~1):", round(semantic_gravity(invariant_f, invariant_r), 3))
    print("G_js (flipped,   expect ~0):", round(semantic_gravity(flipped_f, flipped_r), 3))
    print("BSE  (confident, expect ~0):", round(binary_semantic_entropy([0.99, 0.01]), 3))
    print("BSE  (uniform, expect ln2): ", round(binary_semantic_entropy([0.5, 0.5]), 3))
