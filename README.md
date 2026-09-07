# Saliency — peer-review notebook

A CATALINA / Global Workspace Theory population on a measured Kyiv. This repository is the **executable** copy of the instrument: same engine, same seeds, same files. Running `notebooks/saliency_review.ipynb` re-computes the same peaks the live instrument shows.

Van Dijcke, D., Wright, A. L., & Polyak, M. (2023). Public response to government alerts saves lives during Russian invasion of Ukraine. *PNAS*, 120(18), e2220160120.

## Run

```bash
git clone https://github.com/testing-crsscl/saliency-kyiv.git
cd saliency-kyiv
python -m pip install -r requirements.txt
jupyter notebook notebooks/saliency_review.ipynb
```

The first cell loads `saliency/` from this checkout (or `pip install git+https://github.com/testing-crsscl/saliency-kyiv.git` if you only have the `.ipynb`). Data is read from `data/` here, or fetched from GitHub raw URLs.

It then runs the same pipeline as the instrument:

1. Ingest (OSM grid, WorldPop, WVS Wave 7 Kyiv, alert times, reconstructed Van Dijcke curves)
2. Lattice + population
3. One-agent workspace cycle
4. Architecture tests (must be 21/21)
5. Event-study vs reconstructed Van Dijcke
6. Identifying design (habituation vs calendar)

Expected numbers, seed 21 / 31 / 32 / 33, N = 140 then 120:

| Period | n alerts | Simulated peak | Reconstructed VD peak | Shape MSE |
|--------|----------|----------------|------------------------|-----------|
| Mar–Apr | 40 | 31.1 m | 2100 m | 0.056 |
| May–Jun | 220 | 6.7 m | 1303 m | 0.110 |
| Jul–Sep | 360 | silent | 733 m | 0.258 |

Architecture checks 21/21. Pre-trend identically 0. Levels fail on purpose: the PNAS series is a distance *sum*, not mean metres per agent.

## Honest claims

This is **not** a fitted replication of the original coefficients (those files are not in hand). τ = 230 is calibrated to a reconstructed Mar–Apr / Jul–Sep ratio. Not policy-ready.

## Layout

```
data/          measured + reconstructed JSON (also served from GitHub raw)
saliency/      the engine (Python port of the TypeScript instrument)
notebooks/     executable review notebook
```
