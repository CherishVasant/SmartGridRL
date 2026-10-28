# Colab notebooks (GPU)

Each notebook has its own folder with everything it needs. Everything else (RL training, MPC, scoring, LightGBM)
runs on the laptop.

| Folder | Notebook | What it trains | Status |
|---|---|---|---|
| `F1/` | `F1_deep_forecasters.ipynb` | per-building load / PV forecasters (MLP, LSTM, Transformer) | done, results installed in `F1/deep_forecasters/` |
| `F2/` | `F2_district_total.ipynb` | district-total load forecasters (MLP, patch Transformer) | done, results installed in `F2/total_forecasters/` |
| `F3/` | `F3_automl_total.ipynb` | Optuna search over deep district-total forecasters (embedding MLP, GRU, TCN, N-BEATS, PatchTST) | done, results installed in `F3/total_forecasters/` |

The RL training stays on the laptop: with the small (64 x 64) networks it is limited by the CityLearn simulation and
the MPC's linear program (CPU), so a GPU does not speed it up.

## F3 – automated search over district-total forecasters (done)

`F3/F3_automl_total.ipynb` (saved with its outputs); Drive folder `MyDrive/SmartGridRL_F3/` (the search is resumable
from `search.db`). Results: `F3/F3_automl_results.zip`, installed with
`python tools/install_f2_results.py colab/F3/F3_automl_results.zip colab/F3/total_forecasters`. The MPC uses it with
the forecaster option `gbm_tuned_tot+deep_f3`; with it, the plain MPC gets the best private score (0.839).

## F2 – district-total load forecasters (done; about 45 minutes)

Why: the ramping and load-factor scores depend on the district's **total** load over the **next few hours** (a perfect
forecast of it would take the Phase III public score from 0.774 to about 0.65). F2 trains deep models of the next
24 h of the district total on "pseudo-districts" (random groups of the other districts' buildings, new ones every
epoch), never on the district they forecast.

1. Open <https://colab.research.google.com> → **File → Upload notebook** → `colab/F2/F2_district_total.ipynb`.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. **Runtime → Run all.** Allow Google Drive access. Everything is kept in **`MyDrive/SmartGridRL_F2/`**.
   - The data file is copied from `MyDrive/SmartGridRL_F1/` automatically (F1 was run before). Only if it is missing,
     an upload button appears under cell 2 → choose `colab/F2/citylearn2022_all_buildings.npz`.
4. Wait about 30–60 minutes (progress under cell 7, one line per epoch).
   - If Colab disconnects: reconnect and **Run all** again; it resumes from the last epoch
     (`MyDrive/SmartGridRL_F2/checkpoints/`).
5. When cell 8 finishes, the browser downloads **`F2_district_total_results.zip`** (a copy stays in Drive).
6. Put the zip in **`D:\DEV\Projects\1_ResumeWorthy\SmartGridRL\colab\F2\`** and say it is there.

Afterwards (on the laptop): `python tools/install_f2_results.py` unzips it into `colab/F2/total_forecasters/`,
verifies every model and prints the comparison with LightGBM; the MPC then uses it with the forecaster options
`gbm_tuned_totdeep_<arch>` (deep total) or `gbm_tuned_tot+deep_<arch>` (average with the LightGBM total).

## F1 – per-building deep forecasters (done)

`F1/F1_deep_forecasters.ipynb` trained per-building load / PV forecasters; Drive folder `MyDrive/SmartGridRL_F1/`.
Results: `F1/F1_deep_forecasters_results.zip`, installed by `python tools/install_f1_results.py` into
`F1/deep_forecasters/`. They were not more accurate than LightGBM, so the controller uses LightGBM.
