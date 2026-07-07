# Observational Entropy Benchmark (OEB)

Diagnostic benchmark of **"chiral" (time-reversed) video pairs** of irreversible
physical processes, used to measure **Semantic Gravity** — the degree to which a
Video-LLM's prediction is driven by parametric/linguistic priors rather than the
visual evidence in the frames.

Each of the 300 pairs shows a high-entropy physical event (fluid dispersion,
material fracture, dissipative combustion, …) in its natural (entropy-increasing)
direction and in reverse (entropy-decreasing). A model that grounds its answer in
the pixels should flip its judgment under reversal; a model that retrieves a
memorized script will not.

This repository accompanies the paper *"Semantic Gravity: When Parametric Memory
Overpowers Visual Thermodynamics in Video-LLMs"* (ICML 2026 Workshop on the Impact
of Memorization on Trustworthy Foundation Models).

## Contents

```
oeb-benchmark/
├── data/
│   └── oeb_manifest.csv      # per-clip metadata (URLs, timestamps, labels)
├── eval/
│   ├── metrics.py           # BSE and Semantic Gravity (G_js) — reference implementation
│   ├── run_eval.py          # evaluation harness (model inference is a stub — see below)
│   └── requirements.txt
├── LICENSE
└── README.md
```

## ⚠️ We do not redistribute video

OEB releases **URLs, timestamps, and labels only** — not video files. The source
clips remain the property of their original creators under their original terms.
To reproduce the benchmark, download the clips from the URLs in
`data/oeb_manifest.csv` and trim them to the listed `clip_start`/`clip_end`.

## The manifest

`data/oeb_manifest.csv` has one row per clip:

| column | meaning |
|---|---|
| `clip_id` | stable OEB identifier (e.g. `oeb_001`) |
| `source_video_id` | YouTube video ID |
| `url` | source video URL |
| `clip_start`, `clip_end` | trim points (s) for the 10–15 s segment |
| `channel` | source channel |
| `video_title` | source video title |
| `category` | event type (e.g. fluid_dispersion, material_fracture, combustion) |
| `fwd_label` | ground-truth disorder change in the **forward** direction (default `increase`) |
| `license` | license / usage terms of the source video |
| `notes` | free text |

> The `channel`, `video_title`, `clip_start/end`, `category`, and `license`
> columns are placeholders to be filled from the curation log before release.
> Once `channel` and `category` are populated, the per-source and per-category
> breakdowns for the paper come straight out of this file.

## Metrics

`eval/metrics.py` is a **reference implementation** of the two metrics, matching
the paper (natural log throughout):

- **Binary Semantic Entropy (BSE)** — Shannon entropy (nats) of the renormalized
  {increase, decrease} distribution; bounded by `ln 2`.
- **Semantic Gravity (G_js)** — `1 - JSD(P_rev, P_fwd) / ln 2`, in `[0, 1]`.
  `G_js → 1`: prediction invariant under reversal (script retrieval).
  `G_js → 0`: prediction sensitive to frame order (perceptual updating).

## Reproducing the evaluation

1. Download and trim clips per the manifest.
2. For each clip, obtain first-token `{increase, decrease}` probabilities from the
   model under both FORWARD and REVERSE presentation.
3. Compute BSE and G_js with `eval/metrics.py`.

`eval/run_eval.py` wires this together; **model inference is left as a stub**
(`get_direction_probs`) because API access and model endpoints differ per user —
fill it in for your setup. The metric computation and regime decomposition are
complete and runnable on any probability file.

## Citation

```bibtex
@misc{oeb_dataset,
  author       = {Ganesh, Vidya and Sethuraman, T. V. and Britto, Aylmer and Ragunathan, Sibi Anitha},
  title        = {Observational Entropy Benchmark (OEB): video manifest, labels, and evaluation code},
  year         = {2026},
  howpublished = {\url{https://github.com/Vidya1899/oeb-benchmark}},
  note         = {Accessed 2026-07-07}
}
```

## License

Manifest and code in this repository are released under the MIT License
(see `LICENSE`). **This license covers our metadata and code only — not the
underlying videos**, which remain under their original creators' terms.
