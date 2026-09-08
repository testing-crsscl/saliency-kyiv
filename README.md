# Saliency — peer-review notebook

A CATALINA / Global Workspace Theory population on a measured Kyiv. This repository is the **executable** copy of the instrument: same engine, same seeds, same files. Running `notebooks/saliency_review.ipynb` re-computes the same peaks the live instrument shows — ingest → lattice → WVS joints and processor draws → 12-layer cycle → event-study curves → 25 architecture checks.

Van Dijcke, D., Wright, A. L., & Polyak, M. (2023). Public response to government alerts saves lives during Russian invasion of Ukraine. *PNAS*, 120(18), e2220160120.

## Run

```bash
git clone https://github.com/testing-crsscl/saliency-kyiv.git
cd saliency-kyiv
python -m pip install -r requirements.txt
jupyter notebook notebooks/saliency_review.ipynb
```

The first cell loads `saliency/` from this checkout (or `pip install git+https://github.com/testing-crsscl/saliency-kyiv.git` if you only have the `.ipynb`). Data is read from `data/` here, or fetched from GitHub raw URLs.

The notebook is the whole pipeline:

1. Ingest (OSM grid, WorldPop, WVS Wave 7 Kyiv, alert times, reconstructed Van Dijcke curves)
2. Lattice + WorldPop placement audit
3. Population: WVS joints, Dirichlet/Beta/log-normal processor draws, motor distributions
4. Sense: acoustic habituation and idle-bar ignition shares
5. Twelve-layer GWT cycle and a worked minute (union appraisal table)
6. Architecture tests (must be **25/25**)
7. Event-study vs reconstructed Van Dijcke (shape + levels + action mix)
8. Three periods (Mar–Apr, May–Jun, Jul–Sep)
9. Identifying design (habituation vs calendar on a synthetic city)
10. Verdict, falsifiers, reproducibility

Expected numbers, same seeds as the instrument: architecture **25/25**, pre-trend identically **0**. Peaks move because mapping weights, idle bars and τ are sampled per agent. Levels fail on purpose: the PNAS series is a distance *sum*, not mean metres per agent.

## Honest claims

This is **not** a fitted replication of the original coefficients (those files are not in hand). τ = 230 is calibrated to a reconstructed Mar–Apr / Jul–Sep ratio. Appraisal Dirichlet means are a structural prior — WVS has no shelter-seeking item. Not policy-ready.

## Layout

```
data/          measured + reconstructed JSON (also served from GitHub raw)
saliency/      the engine (Python port of the TypeScript instrument)
notebooks/     executable review notebook
```
