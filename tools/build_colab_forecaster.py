"""Builds colab/deep_forecaster_colab.ipynb: trains deep load / PV forecasters (MLP, LSTM, Transformer) on a GPU,
leave-district-out, and exports them as TorchScript for the controller (DeepForecaster in the core).
The notebook embeds colab/deep_inputs.py verbatim (the same input definition the controller uses).
Usage (from the project folder): python tools/build_colab_forecaster.py
"""
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

COLAB = Path(__file__).resolve().parent.parent / 'colab'
deep_inputs_src = (COLAB / 'deep_inputs.py').read_text(encoding='utf-8')

cells = [
    new_markdown_cell(r"""# Deep load / PV forecasters for CityLearn 2022 (run on a GPU in Colab)

**What this does.** Trains three deep forecasters of the next 24 hours of each building's non-shiftable load and PV
generation: an **MLP**, an **LSTM** and a **Transformer** (patches of one day), each with 3 seeds. Training is
**leave-district-out**: the model for a district never sees that district's buildings (train = buildings 1-5,
val = 6-10, test = 11-17), so nothing is leaked. Every model is compared with the LightGBM forecaster already used by
the controller, and all models are exported as TorchScript (they then run on the CPU inside the MPC).

**How to run**
1. Colab menu: *Runtime → Change runtime type → T4 GPU* (any GPU works; CPU works too, just slower).
2. *Runtime → Run all*. The first cell asks you to upload **`citylearn2022_all_buildings.npz`**
   (from the project's `colab/` folder).
3. It takes roughly 30-60 minutes on a T4. At the end it downloads **`deep_forecasters.zip`**.
4. Send back `deep_forecasters.zip` (and, if you like, this notebook with its outputs, *File → Download → .ipynb*).
   Put the zip in the project's `colab/` folder.

Nothing else is needed: no Drive, no extra packages (PyTorch, numpy and pandas are preinstalled in Colab)."""),
    new_code_cell(r"""import torch                      # first: on Windows, torch must be imported before numpy / pandas
import torch.nn as nn
import os, time, json, zipfile
import numpy as np
import pandas as pd

DATA = 'citylearn2022_all_buildings.npz'
if not os.path.exists(DATA):
    try:
        from google.colab import files
        print('Upload citylearn2022_all_buildings.npz')
        files.upload()
    except ImportError:
        raise FileNotFoundError(DATA + ' not found next to the notebook')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device, torch.cuda.get_device_name(0) if device == 'cuda' else '')
QUICK = os.environ.get('DEEPFC_QUICK') == '1'      # tiny settings for a local code check (leave off in Colab)"""),
    new_markdown_cell("## Input definition (identical to `colab/deep_inputs.py`, used by the controller)"),
    new_code_cell(deep_inputs_src),
    new_markdown_cell("## Data: all 17 buildings, one year, hourly"),
    new_code_cell(r"""d = np.load(DATA)
LOAD, SOLAR, WEATHER, WPRED, CAL = d['load'], d['solar'], d['weather'], d['weather_pred'], d['cal']
T = len(LOAD)
DISTRICTS = {'train': list(range(0, 5)), 'val': list(range(5, 10)), 'test': list(range(10, 17))}
# LightGBM forecaster already in the controller: district net-load MAE (kWh, summed over the district's buildings),
# mean over horizons 1-24, on the same evaluation hours (every 6 h from hour 168)
GBM_MAE = {'train': 2.296, 'val': 2.831, 'test': 3.707}
SEASONAL_MAE = {'train': 2.654, 'val': 3.282, 'test': 3.818}

def build(buildings, steps):
    X, Y, S = [], [], []
    for t in steps:
        x, sl, ss = inputs(LOAD[:, buildings], SOLAR[:, buildings], WEATHER, WPRED, CAL, t)
        X.append(x); Y.append(targets(LOAD[:, buildings], SOLAR[:, buildings], t, sl, ss)); S.append(np.stack([sl, ss], 1))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(S)

train_step = 4 if QUICK else 1
eval_steps = list(range(168, T - H - 1, 6))
print('inputs per sample:', N_INPUTS)"""),
    new_markdown_cell("## Models (all take the same flat input vector)"),
    new_code_cell(r"""N_SHARED = N_INPUTS - 2 * W

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_INPUTS, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 512), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(512, 2 * H))
    def forward(self, x):
        return self.net(x)

class LSTMNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(2, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden + N_SHARED, 256), nn.GELU(), nn.Linear(256, 2 * H))
    def forward(self, x):
        seq = torch.stack([x[:, :W], x[:, W:2 * W]], dim=-1)
        out, _ = self.lstm(seq)
        return self.head(torch.cat([out[:, -1], x[:, 2 * W:]], dim=1))

class PatchTransformer(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.embed = nn.Linear(2 * 24, d_model)
        self.pos = nn.Parameter(torch.zeros(1, W // 24, d_model))
        layer = nn.TransformerEncoderLayer(d_model, 4, 128, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.head = nn.Sequential(nn.Linear(d_model + N_SHARED, 256), nn.GELU(), nn.Linear(256, 2 * H))
    def forward(self, x):
        n = x.shape[0]
        patches = torch.cat([x[:, :W].reshape(n, W // 24, 24), x[:, W:2 * W].reshape(n, W // 24, 24)], dim=-1)
        z = self.encoder(self.embed(patches) + self.pos).mean(dim=1)
        return self.head(torch.cat([z, x[:, 2 * W:]], dim=1))

ARCHS = {'mlp': MLP, 'lstm': LSTMNet, 'transformer': PatchTransformer}
SEEDS = [0] if QUICK else [0, 1, 2]
EPOCHS = 2 if QUICK else 60
PATIENCE = 6"""),
    new_markdown_cell("## Training (L1 loss, early stopping on 10% of the training samples)"),
    new_code_cell(r"""def train_model(arch, seed, Xtr, Ytr):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ARCHS[arch]().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    idx = np.random.permutation(len(Xtr)); n_val = len(idx) // 10
    va, tr = idx[:n_val], idx[n_val:]
    Xt, Yt = torch.tensor(Xtr), torch.tensor(Ytr)
    Xv, Yv = Xt[va].to(device), Yt[va].to(device)
    best, best_state, bad = np.inf, None, 0
    for epoch in range(EPOCHS):
        model.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), 1024):
            b = perm[i:i + 1024]
            loss = nn.functional.l1_loss(model(Xt[b].to(device)), Yt[b].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = np.mean([nn.functional.l1_loss(model(Xv[i:i + 4096]), Yv[i:i + 4096]).item() for i in range(0, len(Xv), 4096)])
        if v < best - 1e-5:
            best, best_state, bad = v, {k: x.detach().clone() for k, x in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model.eval(), best, epoch + 1

def predict(models, X):
    with torch.no_grad():
        Xd = torch.tensor(X)
        outs = [torch.cat([m(Xd[i:i + 8192].to(device)).cpu() for i in range(0, len(Xd), 8192)]).numpy() for m in models]
    return np.mean(outs, axis=0)

def evaluate(models, buildings):
    X, Y, S = build(buildings, eval_steps)
    P = predict(models, X)
    B = len(buildings)
    n = len(eval_steps)
    sl, ss = S[:, 0:1], S[:, 1:2]
    pl, ps = np.maximum(P[:, :H] * sl, 0), np.maximum(P[:, H:] * ss, 0)
    al, as_ = Y[:, :H] * sl, Y[:, H:] * ss
    net_err = np.abs((pl - ps) - (al - as_)).reshape(n, B, H).sum(axis=1).mean(axis=0)       # per horizon
    def r2(p, a): return 1 - ((p - a) ** 2).sum() / ((a - a.mean()) ** 2).sum()
    def nmae(p, a): return np.abs(p - a).mean() / np.abs(a).mean()
    return {'net_mae_mean': net_err.mean(), 'net_mae_h1': net_err[0], 'load_r2': r2(pl, al), 'solar_r2': r2(ps, as_),
            'load_nmae': nmae(pl, al), 'solar_nmae': nmae(ps, as_)}"""),
    new_markdown_cell("## Train every architecture for every district (leave-district-out) and compare"),
    new_code_cell(r"""os.makedirs('deep_forecasters', exist_ok=True)
rows, trained = [], {}
for district, target_b in DISTRICTS.items():
    others = [b for dd, bs in DISTRICTS.items() if dd != district for b in bs]
    Xtr, Ytr, _ = build(others, range(W, T - H - 1, train_step))
    print(f'\n=== {district}: training on {len(others)} other buildings, {len(Xtr):,} samples')
    for arch in ARCHS:
        start = time.time()
        models = []
        for seed in SEEDS:
            m, val_loss, epochs = train_model(arch, seed, Xtr, Ytr)
            models.append(m)
            print(f'  {arch} seed {seed}: val L1 {val_loss:.4f} after {epochs} epochs')
        res = evaluate(models, target_b)
        trained[(district, arch)] = models
        rows.append({'district': district, 'arch': arch, **res, 'gbm_net_mae_mean': GBM_MAE[district],
                     'seasonal_net_mae_mean': SEASONAL_MAE[district], 'minutes': (time.time() - start) / 60})
        print(f"  {arch}: net-load MAE {res['net_mae_mean']:.3f} (LightGBM {GBM_MAE[district]:.3f}, seasonal "
              f"{SEASONAL_MAE[district]:.3f}) | load R2 {res['load_r2']:.2f} NMAE {res['load_nmae']:.0%} | "
              f"PV R2 {res['solar_r2']:.2f} | {(time.time() - start) / 60:.1f} min")
metrics = pd.DataFrame(rows)
metrics.to_csv('deep_forecasters/metrics.csv', index=False)
metrics"""),
    new_markdown_cell("## Export as TorchScript (CPU) with check vectors, and download"),
    new_code_cell(r"""check = {}
for (district, arch), models in trained.items():
    X, _, _ = inputs(LOAD[:, DISTRICTS[district]], SOLAR[:, DISTRICTS[district]], WEATHER, WPRED, CAL, 1000)
    example = torch.tensor(X)
    for k, m in enumerate(models):
        m_cpu = m.to('cpu').eval()
        with torch.no_grad():
            # check_trace=False: the Transformer's fast path makes the tracer's own graph check fail spuriously;
            # instead the exported model's outputs are compared with the original model's below
            traced = torch.jit.trace(m_cpu, example, check_trace=False)
            eager, exported = m_cpu(example).numpy(), traced(example).numpy()
        assert np.allclose(eager, exported, atol=1e-4), f'{district} {arch} s{k}: export mismatch'
        path = f'deep_forecasters/{district}_{arch}_s{k}.pt'
        traced.save(path)
        check[f'{district}_{arch}_s{k}'] = exported
np.savez('deep_forecasters/check_t1000.npz', **check)
with open('deep_forecasters/info.json', 'w') as f:
    json.dump({'W': W, 'H': H, 'N_INPUTS': N_INPUTS, 'seeds': SEEDS, 'archs': list(ARCHS), 'quick': QUICK,
               'torch': torch.__version__}, f, indent=2)
with zipfile.ZipFile('deep_forecasters.zip', 'w') as z:
    for name in os.listdir('deep_forecasters'):
        z.write(os.path.join('deep_forecasters', name), name)
print('wrote deep_forecasters.zip')
try:
    from google.colab import files
    files.download('deep_forecasters.zip')
except ImportError:
    pass"""),
]
nb = new_notebook(cells=cells, metadata={'accelerator': 'GPU', 'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}})
nbformat.write(nb, str(COLAB / 'deep_forecaster_colab.ipynb'))
print('wrote', COLAB / 'deep_forecaster_colab.ipynb')
