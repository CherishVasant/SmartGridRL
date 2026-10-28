"""Builds colab/F1/F1_deep_forecasters.ipynb: trains deep load / PV forecasters (MLP, LSTM, Transformer) on a Colab GPU,
leave-district-out, and exports them as TorchScript for the controller (DeepForecaster in the core).

Everything is kept in Google Drive (MyDrive/SmartGridRL_F1/): the data, a checkpoint after every epoch, each finished
model and the results zip, so a disconnected session resumes with Run all. The notebook embeds colab/F1/deep_inputs.py
verbatim (the same input definition the controller uses). Without Colab it runs in the current folder (local test).
Usage (from the project folder): python tools/build_colab_forecaster.py
"""
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

COLAB = Path(__file__).resolve().parent.parent / 'colab' / 'F1'
deep_inputs_src = (COLAB / 'deep_inputs.py').read_text(encoding='utf-8')

cells = [
    new_markdown_cell(r"""# F1 – Deep load / PV forecasters for CityLearn 2022 (Colab GPU)

**What this does.** Trains three deep forecasters of the next 24 hours of each building's load and solar generation –
an **MLP**, an **LSTM** and a **Transformer**, 3 seeds each – *leave-district-out*: the model for a district never
sees that district's buildings. They are compared with the LightGBM forecaster already used by the controller and
exported as TorchScript, so they can run on a normal CPU inside the battery controller afterwards.

**How to run (Google Colab)**
1. *Runtime → Change runtime type → **T4 GPU*** → Save.
2. *Runtime → **Run all***. Accept the Google Drive permission pop-up (it stores everything in
   **`MyDrive/SmartGridRL_F1/`**).
3. The first time only, the data cell asks you to upload **`citylearn2022_all_buildings.npz`** (from the project's
   `colab/F1/` folder). It is copied to the Drive folder, so you never upload it again.
4. Training takes roughly 30–60 minutes on a T4. A checkpoint is saved to Drive after every epoch: **if Colab
   disconnects, just reconnect and Run all again** – finished models are skipped and the current one resumes.
5. At the end it saves **`MyDrive/SmartGridRL_F1/F1_deep_forecasters_results.zip`** and downloads it.

**Send back:** `F1_deep_forecasters_results.zip` → put it in the project's `colab/F1/` folder
(`D:\DEV\Projects\1_ResumeWorthy\SmartGridRL\colab\F1\`). No other files are needed."""),
    new_markdown_cell("## 1. Setup: GPU, Google Drive folder"),
    new_code_cell(r"""import torch                      # first: on Windows, torch must be imported before numpy / pandas
import torch.nn as nn
import os, time, json, zipfile, shutil
import numpy as np
import pandas as pd

try:
    from google.colab import drive, files
    drive.mount('/content/drive')
    WORK = '/content/drive/MyDrive/SmartGridRL_F1'
    IN_COLAB = True
except ImportError:                 # local test: work in the current folder
    WORK, IN_COLAB = '.', False
MODELS = os.path.join(WORK, 'deep_forecasters')        # finished models (TorchScript), metrics, check vectors
CKPT = os.path.join(WORK, 'checkpoints')               # training state after every epoch (for resuming)
os.makedirs(MODELS, exist_ok=True); os.makedirs(CKPT, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
QUICK = os.environ.get('DEEPFC_QUICK') == '1'          # tiny settings for a local code check (off in Colab)
print('device:', device, torch.cuda.get_device_name(0) if device == 'cuda' else '', '| storage:', WORK)"""),
    new_markdown_cell("## 2. Data (uploaded once, then kept in Drive)"),
    new_code_cell(r"""DATA = os.path.join(WORK, 'citylearn2022_all_buildings.npz')
if not os.path.exists(DATA):
    if IN_COLAB:
        print('Upload citylearn2022_all_buildings.npz (from the project colab/F1/ folder)')
        up = files.upload()
        shutil.move(next(iter(up)), DATA)
    else:
        raise FileNotFoundError(DATA)
print('data:', DATA)"""),
    new_markdown_cell("## 3. Input definition (identical to `colab/F1/deep_inputs.py`, used by the controller)"),
    new_code_cell(deep_inputs_src),
    new_markdown_cell("## 4. Load the 17 buildings"),
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
eval_steps = list(range(168, T - H - 1, 6))[::8 if QUICK else 1]
print('inputs per sample:', N_INPUTS)"""),
    new_markdown_cell("## 5. Models (all take the same flat input vector)"),
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
    new_markdown_cell("## 6. Training: L1 loss, early stopping, a Drive checkpoint after every epoch (resumable)"),
    new_code_cell(r"""def save_atomic(obj, path):
    torch.save(obj, path + '.tmp'); os.replace(path + '.tmp', path)

def train_model(arch, seed, Xtr, Ytr, tag):
    # Resumes from CKPT/<tag>.pt if it exists; returns the best model (eval mode)
    path = os.path.join(CKPT, f'{tag}.pt')
    torch.manual_seed(seed)
    model = ARCHS[arch]().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    state = {'epoch': 0, 'best': np.inf, 'best_state': None, 'bad': 0, 'done': False}
    if os.path.exists(path):
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state['model']); opt.load_state_dict(state['opt'])
        print(f'    resuming {tag} at epoch {state["epoch"]}' + (' (already finished)' if state['done'] else ''))
    idx = np.random.RandomState(seed).permutation(len(Xtr)); n_val = len(idx) // 10
    va, tr = idx[:n_val], idx[n_val:]
    Xt, Yt = torch.tensor(Xtr), torch.tensor(Ytr)
    Xv, Yv = Xt[va].to(device), Yt[va].to(device)
    while not state['done'] and state['epoch'] < EPOCHS:
        model.train()
        perm = np.random.RandomState(seed * 1000 + state['epoch']).permutation(tr)
        for i in range(0, len(perm), 1024):
            b = perm[i:i + 1024]
            loss = nn.functional.l1_loss(model(Xt[b].to(device)), Yt[b].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = np.mean([nn.functional.l1_loss(model(Xv[i:i + 4096]), Yv[i:i + 4096]).item() for i in range(0, len(Xv), 4096)])
        state['epoch'] += 1
        if v < state['best'] - 1e-5:
            state.update(best=v, bad=0, best_state={k: x.detach().cpu().clone() for k, x in model.state_dict().items()})
        else:
            state['bad'] += 1
        state['done'] = state['bad'] >= PATIENCE or state['epoch'] >= EPOCHS
        save_atomic(dict(state, model=model.state_dict(), opt=opt.state_dict()), path)
    model.load_state_dict(state['best_state'])
    print(f"    {tag}: val L1 {state['best']:.4f} after {state['epoch']} epochs")
    return model.eval()

def export(model, tag, district):
    # TorchScript (CPU) + check vector; the exported model's outputs must match the original's
    X, _, _ = inputs(LOAD[:, DISTRICTS[district]], SOLAR[:, DISTRICTS[district]], WEATHER, WPRED, CAL, 1000)
    example = torch.tensor(X)
    m_cpu = model.to('cpu').eval()
    with torch.no_grad():
        # check_trace=False: the Transformer fast path makes the tracer's own graph check fail spuriously
        traced = torch.jit.trace(m_cpu, example, check_trace=False)
        eager, exported = m_cpu(example).numpy(), traced(example).numpy()
    assert np.allclose(eager, exported, atol=1e-4), f'{tag}: export mismatch'
    traced.save(os.path.join(MODELS, f'{tag}.pt'))
    np.save(os.path.join(MODELS, f'check_{tag}.npy'), exported)
    return model.to(device)

def predict(models, X):
    with torch.no_grad():
        Xd = torch.tensor(X)
        outs = [torch.cat([m(Xd[i:i + 8192].to(device)).cpu() for i in range(0, len(Xd), 8192)]).numpy() for m in models]
    return np.mean(outs, axis=0)

def evaluate(models, buildings):
    X, Y, S = build(buildings, eval_steps)
    P = predict(models, X)
    B, n = len(buildings), len(eval_steps)
    sl, ss = S[:, 0:1], S[:, 1:2]
    pl, ps = np.maximum(P[:, :H] * sl, 0), np.maximum(P[:, H:] * ss, 0)
    al, as_ = Y[:, :H] * sl, Y[:, H:] * ss
    net_err = np.abs((pl - ps) - (al - as_)).reshape(n, B, H).sum(axis=1).mean(axis=0)       # per horizon
    def r2(p, a): return 1 - ((p - a) ** 2).sum() / ((a - a.mean()) ** 2).sum()
    def nmae(p, a): return np.abs(p - a).mean() / np.abs(a).mean()
    return {'net_mae_mean': net_err.mean(), 'net_mae_h1': net_err[0], 'load_r2': r2(pl, al), 'solar_r2': r2(ps, as_),
            'load_nmae': nmae(pl, al), 'solar_nmae': nmae(ps, as_)}"""),
    new_markdown_cell("## 7. Train every architecture for every district (leave-district-out) and compare"),
    new_code_cell(r"""rows = []
for district, target_b in DISTRICTS.items():
    others = [b for dd, bs in DISTRICTS.items() if dd != district for b in bs]
    Xtr, Ytr, _ = build(others, range(W, T - H - 1, train_step))
    print(f'\n=== {district}: training on {len(others)} other buildings, {len(Xtr):,} samples')
    for arch in ARCHS:
        start, models = time.time(), []
        for seed in SEEDS:
            tag = f'{district}_{arch}_s{seed}'
            models.append(export(train_model(arch, seed, Xtr, Ytr, tag), tag, district))
        res = evaluate(models, target_b)
        rows.append({'district': district, 'arch': arch, **res, 'gbm_net_mae_mean': GBM_MAE[district],
                     'seasonal_net_mae_mean': SEASONAL_MAE[district], 'minutes': (time.time() - start) / 60})
        print(f"  {arch}: net-load MAE {res['net_mae_mean']:.3f} (LightGBM {GBM_MAE[district]:.3f}, seasonal "
              f"{SEASONAL_MAE[district]:.3f}) | load R2 {res['load_r2']:.2f} NMAE {res['load_nmae']:.0%} | "
              f"PV R2 {res['solar_r2']:.2f} | {(time.time() - start) / 60:.1f} min")
        pd.DataFrame(rows).to_csv(os.path.join(MODELS, 'metrics.csv'), index=False)
metrics = pd.DataFrame(rows)
metrics"""),
    new_markdown_cell("## 8. Pack the results (saved in Drive and downloaded)"),
    new_code_cell(r"""with open(os.path.join(MODELS, 'info.json'), 'w') as f:
    json.dump({'W': W, 'H': H, 'N_INPUTS': N_INPUTS, 'seeds': SEEDS, 'archs': list(ARCHS), 'quick': QUICK,
               'torch': torch.__version__}, f, indent=2)
ZIP = os.path.join(WORK, 'F1_deep_forecasters_results.zip')
with zipfile.ZipFile(ZIP, 'w') as z:
    for name in os.listdir(MODELS):
        z.write(os.path.join(MODELS, name), name)
print('saved', ZIP)
if IN_COLAB:
    files.download(ZIP)"""),
]
nb = new_notebook(cells=cells, metadata={'accelerator': 'GPU', 'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}})
nbformat.write(nb, str(COLAB / 'F1_deep_forecasters.ipynb'))
print('wrote', COLAB / 'F1_deep_forecasters.ipynb')
