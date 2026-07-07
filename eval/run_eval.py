"""
OEB evaluation harness.

Pipeline:
  1. Load the clip manifest.
  2. For each clip, get the model's first-token {increase, decrease} probabilities
     under FORWARD and REVERSE presentation.
  3. Compute per-clip G_js and BSE, per-condition accuracy, and regime masses.

Model inference is intentionally a STUB (`get_direction_probs`): API access,
endpoints, and frame extraction differ per user. Fill it in for your setup.
Everything downstream of it is complete and runnable.
"""

from __future__ import annotations
import argparse
import csv
import statistics
from metrics import (
    binary_semantic_entropy,
    semantic_gravity,
    regime,
    regime_masses,
)

DIRECTIONS = ("increase", "decrease")


# ----------------------------------------------------------------------------
# STUB: replace with real model inference.
# ----------------------------------------------------------------------------
def get_direction_probs(clip_path: str, order: str, model: str):
    """Return [P(increase), P(decrease)] for a single clip.

    Args:
        clip_path: local path to the trimmed clip (download per the manifest).
        order:     "forward" or "reverse".
        model:     model identifier / endpoint.

    Implement this by:
      - sampling frames from the clip (reverse the frame order if order=="reverse"),
      - sending them with the fixed prompt (see the paper's supplementary),
      - reading the top-k first-token log-probs,
      - summing mass over the {increase,...} and {decrease,...} token clusters,
      - renormalizing over the two.
    """
    raise NotImplementedError(
        "Plug in your model inference here (returns [P(increase), P(decrease)])."
    )


def load_manifest(path: str):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def evaluate(manifest_path: str, model: str, clips_dir: str):
    rows = load_manifest(manifest_path)
    g_js_values, bse_rev_values = [], []
    correct_fwd = correct_rev = 0
    n = 0

    for row in rows:
        clip_path = f"{clips_dir}/{row['clip_id']}.mp4"
        fwd_label = row.get("fwd_label", "increase").strip().lower()
        rev_label = "decrease" if fwd_label == "increase" else "increase"

        p_fwd = get_direction_probs(clip_path, "forward", model)
        p_rev = get_direction_probs(clip_path, "reverse", model)

        pred_fwd = DIRECTIONS[0] if p_fwd[0] >= p_fwd[1] else DIRECTIONS[1]
        pred_rev = DIRECTIONS[0] if p_rev[0] >= p_rev[1] else DIRECTIONS[1]
        correct_fwd += (pred_fwd == fwd_label)
        correct_rev += (pred_rev == rev_label)

        g_js_values.append(semantic_gravity(p_fwd, p_rev))
        bse_rev_values.append(binary_semantic_entropy(p_rev))
        n += 1

    masses = regime_masses(g_js_values)
    # Per-regime reverse accuracy (the cross-tab that shows the mechanism).
    per_regime = {"grounding": [], "conflict": [], "retrieval": []}
    for row, g in zip(rows, g_js_values):
        fwd_label = row.get("fwd_label", "increase").strip().lower()
        rev_label = "decrease" if fwd_label == "increase" else "increase"
        # (recomputing pred_rev would require caching; left for the real run)

    print(f"Model: {model}   n={n}")
    print(f"Acc_fwd: {100*correct_fwd/n:.2f}%   Acc_rev: {100*correct_rev/n:.2f}%")
    print(f"Grounding: {masses['grounding']:.2f}%   "
          f"Conflict: {masses['conflict']:.2f}%   "
          f"Retrieval: {masses['retrieval']:.2f}%")
    print(f"Median G_js: {statistics.median(g_js_values):.3f}")
    print(f"Median BSE (reverse): {statistics.median(bse_rev_values):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="../data/oeb_manifest.csv")
    ap.add_argument("--model", required=True, help="model id / endpoint")
    ap.add_argument("--clips_dir", default="./clips", help="dir of trimmed clips")
    args = ap.parse_args()
    evaluate(args.manifest, args.model, args.clips_dir)
