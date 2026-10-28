"""Installs the results of colab/F2/F2_district_total.ipynb for the controller.

Unzips colab/F2/F2_district_total_results.zip into colab/F2/total_forecasters/, checks that every TorchScript model loaded
here gives exactly the outputs recorded in Colab (check_<tag>.npy, inputs built by colab/F2/total_inputs.py at hour 1000
from the district's total load), and prints the accuracy comparison with LightGBM. Afterwards the MPC can use them:
    python tools/phase3_eval.py P3_TOTDEEP_MPC_<arch> mpcfc gbm_tuned_totdeep_<arch> 24 1 3
    python tools/phase3_eval.py P3_TOTBLEND_MPC_<arch> mpcfc gbm_tuned_tot+deep_<arch> 24 1 3
Usage (from the project folder): python tools/install_f2_results.py [path to the zip] [output folder]
"""
import torch   # noqa: F401  (before numpy / pandas: WinError 1114)
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ZIP = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'colab' / 'F2' / 'F2_district_total_results.zip'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'colab' / 'F2' / 'total_forecasters'
sys.path.insert(0, str(ROOT / 'colab' / 'F2'))
import total_inputs as ti   # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ZIP) as z:
    z.extractall(OUT)
print(f'unzipped {ZIP.name} -> {OUT}')

d = np.load(ROOT / 'colab' / 'F2' / 'citylearn2022_all_buildings.npz')
DISTRICTS = {'train': list(range(0, 5)), 'val': list(range(5, 10)), 'test': list(range(10, 17))}
ok = True
for model_path in sorted(OUT.glob('*_s*.pt')):
    tag = model_path.stem
    X, _ = ti.inputs(d['load'][:, DISTRICTS[tag.split('_')[0]]].sum(1, keepdims=True), d['weather'], d['weather_pred'],
                     d['cal'], 1000)
    with torch.no_grad():
        out = torch.jit.load(str(model_path), map_location='cpu')(torch.tensor(X)).numpy()
    match = np.allclose(out, np.load(OUT / f'check_{tag}.npy'), atol=1e-4)
    ok &= match
    print(f'  {tag}: {"OK" if match else "MISMATCH"}')
print('all models verified' if ok else 'SOME MODELS DO NOT MATCH - do not use them')
metrics = OUT / 'metrics.csv'
if metrics.exists():
    m = pd.read_csv(metrics)
    print(m[['district', 'arch', 'total_acc', 'gbm_current_acc', 'gbm_best_acc', 'h1', 'h6', 'h24']].round(3)
          .to_string(index=False))
