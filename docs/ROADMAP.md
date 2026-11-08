# Project Roadmap — Camera Trap Image Enhancement

CENG376 Image Processing term project. Work is delivered in weekly stages so
that progress is visible in the Git history. Each stage ends with a Git tag
(`week-N`) on `main`.

| Week | Stage | Tag |
|------|-------|-----|
| 1 | Setup and global histogram equalization demo | `week-1` |
| 2 | Real camera trap dataset and quantitative metrics | `week-2` |
| 3 | CLAHE and gamma correction vs. global HE | `week-3` |
| 4 | Color images and noise-aware enhancement | `week-4` |
| 5 | Batch-processing CLI and method comparison table | `week-5` |
| 6 | Final report and presentation | `week-6` |

---

## Week 1 — Setup and global histogram equalization demo ✅

**Goal:** a working, reproducible first demo of the core idea:
`cv2.equalizeHist` on a dark night image.

**Deliverables**
- Project skeleton: `.gitignore`, pinned `requirements.txt`, `pytest.ini`.
- `src/enhancement.py`: `load_grayscale(path)` and `equalize_histogram(img)`
  with input validation (missing/undecodable file, color input, non-uint8,
  empty and constant images).
- `scripts/make_sample.py`: synthetic dark night scene (trees, grass, deer
  silhouette, sensor noise; pixel values 0–40), so the demo works offline.
- `demo.py`: headless 2×2 figure (before/after images + 0–255 histograms),
  `--input` / `--output` arguments.

**Tests:** shape/dtype preserved; dark input is stretched (mean rises, max
reaches 255); clear errors for invalid paths; constant images do not crash;
color input handled; demo writes the figure.

**Commits:** skeleton → roadmap → core function → sample generator → demo →
tests → README. **Tag:** `week-1`.

---

## Week 2 — Real dataset and quantitative metrics

**Goal:** move from the synthetic image to real camera trap night photos and
measure the enhancement with numbers instead of only by eye.

**Deliverables**
- Choose a public camera trap dataset (candidates: Caltech Camera Traps,
  Snapshot Serengeti via LILA BC). Add a small curated subset (10–20 night
  images, license-compatible) or a download script; document the source and
  license in `data/README.md`.
- `src/metrics.py`: mean brightness, standard deviation (global contrast),
  Shannon entropy of the histogram.
- Extend the demo to print/plot metrics before and after equalization.

**Tests:** metrics on hand-computed arrays (constant image → std 0, entropy 0;
two equally frequent values → entropy 1 bit); on dark inputs global HE raises
mean and std but never raises histogram entropy (it only remaps/merges bins).

**Commits:** dataset notes/download script → metrics module → metric tests →
demo update. **Tag:** `week-2`.

---

## Week 3 — CLAHE and gamma correction

**Goal:** compare global HE with two alternatives that are often better for
night images.

**Deliverables**
- `src/enhancement.py`: `apply_clahe(img, clip_limit, tile_grid)` and
  `gamma_correction(img, gamma)` (lookup-table based).
- Side-by-side figure: original / global HE / CLAHE / gamma, with histograms.
- Short written discussion of over-enhancement and noise amplification in
  global HE vs. CLAHE.

**Tests:** output shape/dtype; gamma = 1 is the identity; gamma < 1 brightens
a dark image; CLAHE with tiny clip limit changes the image less than global HE;
invalid parameters raise `ValueError`.

**Commits:** gamma → CLAHE → comparison figure → tests. **Tag:** `week-3`.

---

## Week 4 — Color images and noise handling

**Goal:** support color camera trap images and limit the noise that
equalization amplifies.

**Deliverables**
- Color enhancement that equalizes only the luminance channel (Y in YCrCb or
  L in LAB) and keeps chroma unchanged, instead of the Week 1 grayscale
  conversion.
- Optional pre-filtering before enhancement: Gaussian, median and bilateral
  filters; compare noise level in flat regions before/after.

**Tests:** chroma channels unchanged by luminance-only HE; output stays
3-channel uint8; denoising reduces std of a noisy flat patch.

**Commits:** color pipeline → pre-filters → tests → demo update.
**Tag:** `week-4`.

---

## Week 5 — Batch processing CLI and method comparison

**Goal:** process a whole folder and compare all methods objectively.

**Deliverables**
- `enhance.py` CLI: `--input-dir`, `--output-dir`, `--method {he,clahe,gamma}`
  plus method parameters; skips unreadable files with a warning.
- Comparison report: CSV/Markdown table of metrics (mean, std, entropy) per
  image and per method, plus averages.

**Tests:** CLI on a temporary folder writes one output per valid input and
reports invalid files; comparison table has one row per image/method.

**Commits:** CLI → metrics table → tests → README usage. **Tag:** `week-5`.

---

## Week 6 — Final report and presentation

**Goal:** wrap up and present the results.

**Deliverables**
- Final report (`docs/report.md` or PDF): problem, methods, dataset,
  quantitative results, visual examples, limitations, future work.
- Presentation slides with before/after examples.
- Cleaned-up README with final results.

**Tests:** full test suite passes on a fresh clone with
`pip install -r requirements.txt`.

**Commits:** report → slides → README polish. **Tag:** `week-6`
(final submission).
