"""Installs the results of colab/F1/F1_deep_forecasters.ipynb for the controller.

Unzips colab/F1/F1_deep_forecasters_results.zip into colab/F1/deep_forecasters/, checks that every TorchScript model loaded
here gives exactly the outputs recorded in Colab (check_<tag>.npy, inputs built by colab/F1/deep_inputs.py at hour 1000),
and prints the forecast comparison with LightGBM. Afterwards the MPC can use them:
    python tools/phase3_eval.py P3_DEEPMPC_<arch> mpcdeep <mlp|lstm|transformer>
Usage (from the project folder): python tools/install_f1_results.py [path to the zip]
"""
import torch   # noqa: F401  (before numpy / pandas: WinError 1114)
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ZIP = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'colab' / 'F1' / 'F1_deep_forecasters_results.zip'
OUT = ROOT / 'colab' / 'F1' / 'deep_forecasters'
sys.path.insert(0, str(ROOT / 'colab' / 'F1'))
import deep_inputs as di   # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ZIP) as z:
    z.extractall(OUT)
print(f'unzipped {ZIP.name} -> {OUT}')

d = np.load(ROOT / 'colab' / 'F1' / 'citylearn2022_all_buildings.npz')
DISTRICTS = {'train': list(range(0, 5)), 'val': list(range(5, 10)), 'test': list(range(10, 17))}
ok = True
for model_path in sorted(OUT.glob('*_s*.pt')):
    tag = model_path.stem
    district = tag.split('_')[0]
    b = DISTRICTS[district]
    X, _, _ = di.inputs(d['load'][:, b], d['solar'][:, b], d['weather'], d['weather_pred'], d['cal'], 1000)
    with torch.no_grad():
        out = torch.jit.load(str(model_path), map_location='cpu')(torch.tensor(X)).numpy()
    expected = np.load(OUT / f'check_{tag}.npy')
    match = np.allclose(out, expected, atol=1e-4)
    ok &= match
    print(f'  {tag}: {"OK" if match else "MISMATCH"}')
print('all models verified' if ok else 'SOME MODELS DO NOT MATCH - do not use them')
metrics = OUT / 'metrics.csv'
if metrics.exists():
    m = pd.read_csv(metrics)
    print(m[['district', 'arch', 'net_mae_mean', 'gbm_net_mae_mean', 'seasonal_net_mae_mean', 'load_r2', 'solar_r2']]
          .round(3).to_string(index=False))
