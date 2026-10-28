"""Builds colab/F2/F2_district_total.ipynb: deep forecasters of the DISTRICT-TOTAL load (the quantity the ramping and
load-factor KPIs use), trained on a Colab GPU, leave-district-out, exported as TorchScript for the controller.

Training series are pseudo-districts: sums of random subsets of the other districts' buildings, drawn afresh every
epoch (unlimited augmentation, which LightGBM cannot use as cheaply). Everything is kept in Google Drive
(MyDrive/SmartGridRL_F2/), with a checkpoint after every epoch, so a disconnected session resumes with Run all.
The notebook embeds colab/F2/total_inputs.py verbatim. The LightGBM references are read from the load-lab results.
Usage (from the project folder): python tools/build_colab_f2.py
"""
from pathlib import Path

import nbformat
import pandas as pd
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
COLAB = ROOT / 'colab' / 'F2'
LAB = ROOT / 'results' / 'forecast' / 'load_lab'
inputs_src = (COLAB / 'total_inputs.py').read_text(encoding='utf-8')

# LightGBM references (district-total accuracy on the same evaluation hours): current controller and best lab model
lab = pd.concat([pd.read_csv(LAB / f) for f in ['load_lab.csv', 'load_lab2.csv'] if (LAB / f).exists()])
current = {d: round(float(lab[(lab.district == d) & (lab.forecast == 'sum_gbm')].total_acc.iloc[0]), 3)
           for d in ['train', 'val', 'test']}
best = {d: round(float(lab.loc[lab.district == d, 'total_acc'].max()), 3) for d in ['train', 'val', 'test']}

cells = [
    new_markdown_cell(r"""# F2 – Deep district-total load forecasters for CityLearn 2022 (Colab GPU)

**What this does.** The ramping and load-factor scores depend on the **district's total** load, not on each house.
This notebook trains deep forecasters of the next 24 hours of the district total – an **MLP** and a
**patch Transformer**, 3 seeds each – *leave-district-out*: the model for a district never sees that district's
buildings. Training series are *pseudo-districts* (sums of random groups of the other districts' buildings), drawn
afresh every epoch. The models are compared with the LightGBM forecasters and exported as TorchScript, so they run
on a normal CPU inside the battery controller afterwards.

**How to run (Google Colab)**
1. *Runtime → Change runtime type → **T4 GPU*** → Save.
2. *Runtime → **Run all***. Accept the Google Drive permission pop-up (it stores everything in
   **`MyDrive/SmartGridRL_F2/`**).
3. The data file is taken from `MyDrive/SmartGridRL_F1/` if F1 was run before; otherwise the data cell asks you to
   upload **`citylearn2022_all_buildings.npz`** (from the project's `colab/F2/` folder) once.
4. Training takes roughly 30–60 minutes on a T4. A checkpoint is saved to Drive after every epoch: **if Colab
   disconnects, just reconnect and Run all again** – finished models are skipped and the current one resumes.
5. At the end it saves **`MyDrive/SmartGridRL_F2/F2_district_total_results.zip`** and downloads it.

**Send back:** `F2_district_total_results.zip` → put it in the project's `colab/F2/` folder
(`D:\DEV\Projects\1_ResumeWorthy\SmartGridRL\colab\F2\`). No other files are needed."""),
    new_markdown_cell('## 1. Setup: GPU, Google Drive folder'),
    new_code_cell(r"""import torch                      # first: on Windows, torch must be imported before numpy / pandas
import torch.nn as nn
import os, time, json, zipfile, shutil
import numpy as np
import pandas as pd

try:
    from google.colab import drive, files
    drive.mount('/content/drive')
    WORK = '/content/drive/MyDrive/SmartGridRL_F2'
    F1_DATA = '/content/drive/MyDrive/SmartGridRL_F1/citylearn2022_all_buildings.npz'
    IN_COLAB = True
except ImportError:                 # local test: work in the current folder
    WORK, F1_DATA, IN_COLAB = '.', None, False
MODELS = os.path.join(WORK, 'total_forecasters')       # finished models (TorchScript), metrics, check vectors
CKPT = os.path.join(WORK, 'checkpoints')               # training state after every epoch (for resuming)
os.makedirs(MODELS, exist_ok=True); os.makedirs(CKPT, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
QUICK = os.environ.get('DEEPFC_QUICK') == '1'          # tiny settings for a local code check (off in Colab)
print('device:', device, torch.cuda.get_device_name(0) if device == 'cuda' else '', '| storage:', WORK)"""),
    new_markdown_cell('## 2. Data (uploaded once, then kept in Drive)'),
    new_code_cell(r"""DATA = os.path.join(WORK, 'citylearn2022_all_buildings.npz')
if not os.path.exists(DATA):
    if F1_DATA and os.path.exists(F1_DATA):
        shutil.copy(F1_DATA, DATA)
    elif IN_COLAB:
        print('Upload citylearn2022_all_buildings.npz (from the project colab/F2/ folder)')
        up = files.upload()
        shutil.move(next(iter(up)), DATA)
    else:
        raise FileNotFoundError(DATA)
print('data:', DATA)"""),
    new_markdown_cell('## 3. Input definition (identical to `colab/F2/total_inputs.py`, used by the controller)'),
    new_code_cell(inputs_src),
    new_markdown_cell('## 4. Load the 17 buildings; pseudo-districts'),
    new_code_cell(f"""d = np.load(DATA)
LOAD, WEATHER, WPRED, CAL = d['load'], d['weather'], d['weather_pred'], d['cal']
T = len(LOAD)
DISTRICTS = {{'train': list(range(0, 5)), 'val': list(range(5, 10)), 'test': list(range(10, 17))}}
# District-total accuracy (1 - MAE / mean |actual|, horizons 1-24, every 6 h from hour 168) of the LightGBM forecasts:
# the current controller (per-building models, summed) and the best direct total model of tools/load_lab*.py
GBM_CURRENT = {current}
GBM_BEST = {best}

def pseudo(buildings, n, rng):
    # (T, n) sums of random subsets (4 .. 8 buildings) of the given buildings
    cols = []
    for _ in range(n):
        k = rng.integers(4, min(8, len(buildings)) + 1)
        cols.append(LOAD[:, rng.choice(buildings, k, replace=False)].sum(1))
    return np.stack(cols, 1)

def build(x, steps):
    X, Y, S = [], [], []
    for t in steps:
        xi, s = inputs(x, WEATHER, WPRED, CAL, t)
        X.append(xi); Y.append(targets(x, t, s)); S.append(s)
    return np.concatenate(X), np.concatenate(Y), np.concatenate(S)

N_PSEUDO = 8 if QUICK else 96         # pseudo-districts per epoch
STRIDE = 24 if QUICK else 3           # hours between samples (random offset per epoch)
eval_steps = list(range(168, T - H - 1, 6))[::8 if QUICK else 1]
print('inputs per sample:', N_INPUTS)"""),
    new_markdown_cell('## 5. Models (both take the same flat input vector)'),
    new_code_cell(r"""N_SHARED = N_INPUTS - W

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_INPUTS, 768), nn.GELU(), nn.Dropout(0.1), nn.Linear(768, 768), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(768, H))
    def forward(self, x):
        return self.net(x) + x[:, W:W + H]          # residual on the 28-day same-hour mean

class PatchTransformer(nn.Module):
    def __init__(self, d_model=96):
        super().__init__()
        self.embed = nn.Linear(24, d_model)
        self.ctx = nn.Linear(N_SHARED, d_model)
        self.pos = nn.Parameter(torch.zeros(1, W // 24 + 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, 4, 4 * d_model, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 3)
        self.head = nn.Sequential(nn.Linear(2 * d_model, 256), nn.GELU(), nn.Linear(256, H))
    def forward(self, x):
        n = x.shape[0]
        tokens = torch.cat([self.ctx(x[:, W:]).unsqueeze(1), self.embed(x[:, :W].reshape(n, W // 24, 24))], dim=1)
        z = self.encoder(tokens + self.pos)
        return self.head(torch.cat([z[:, 0], z[:, 1:].mean(dim=1)], dim=1)) + x[:, W:W + H]

ARCHS = {'mlp': MLP, 'transformer': PatchTransformer}
SEEDS = [0] if QUICK else [0, 1, 2]
EPOCHS = 2 if QUICK else 40
PATIENCE = 6"""),
    new_markdown_cell('## 6. Training: L1 loss, fresh pseudo-districts every epoch, a Drive checkpoint after every epoch'),
    new_code_cell(r"""def save_atomic(obj, path):
    torch.save(obj, path + '.tmp'); os.replace(path + '.tmp', path)

def train_model(arch, seed, pool, tag):
    # pool: the other districts' buildings. Resumes from CKPT/<tag>.pt if it exists; returns the best model (eval mode)
    path = os.path.join(CKPT, f'{tag}.pt')
    torch.manual_seed(seed)
    model = ARCHS[arch]().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    state = {'epoch': 0, 'best': np.inf, 'best_state': None, 'bad': 0, 'done': False}
    if os.path.exists(path):
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state['model']); opt.load_state_dict(state['opt']); sched.load_state_dict(state['sched'])
        print(f'    resuming {tag} at epoch {state["epoch"]}' + (' (already finished)' if state['done'] else ''))
    # fixed validation set for early stopping: pseudo-districts of the same (non-target) buildings
    Xv, Yv, _ = build(pseudo(pool, 8 if QUICK else 32, np.random.default_rng(999)), range(W, T - H - 1, 48 if QUICK else 12))
    Xv, Yv = torch.tensor(Xv).to(device), torch.tensor(Yv).to(device)
    while not state['done'] and state['epoch'] < EPOCHS:
        rng = np.random.default_rng(seed * 1000 + state['epoch'])
        Xt, Yt, _ = build(pseudo(pool, N_PSEUDO, rng), range(W + int(rng.integers(STRIDE)), T - H - 1, STRIDE))
        Xt, Yt = torch.tensor(Xt), torch.tensor(Yt)
        model.train()
        perm = torch.randperm(len(Xt), generator=torch.Generator().manual_seed(seed * 1000 + state['epoch']))
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            loss = nn.functional.l1_loss(model(Xt[b].to(device)), Yt[b].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = np.mean([nn.functional.l1_loss(model(Xv[i:i + 4096]), Yv[i:i + 4096]).item() for i in range(0, len(Xv), 4096)])
        sched.step(v)
        state['epoch'] += 1
        if v < state['best'] - 1e-5:
            state.update(best=v, bad=0, best_state={k: x.detach().cpu().clone() for k, x in model.state_dict().items()})
        else:
            state['bad'] += 1
        state['done'] = state['bad'] >= PATIENCE or state['epoch'] >= EPOCHS
        save_atomic(dict(state, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict()), path)
        print(f'      epoch {state["epoch"]}: val L1 {v:.4f}', flush=True)
    model.load_state_dict(state['best_state'])
    print(f"    {tag}: val L1 {state['best']:.4f} after {state['epoch']} epochs")
    return model.eval()

def export(model, tag, district):
    # TorchScript (CPU) + check vector; the exported model's outputs must match the original's
    X, _ = inputs(LOAD[:, DISTRICTS[district]].sum(1, keepdims=True), WEATHER, WPRED, CAL, 1000)
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

def evaluate(models, buildings):
    # district-total accuracy on the target district's real total
    x = LOAD[:, buildings].sum(1, keepdims=True)
    X, Y, S = build(x, eval_steps)
    with torch.no_grad():
        P = np.mean([m(torch.tensor(X).to(device)).cpu().numpy() for m in models], axis=0)
    p, a = np.maximum(P, 0) * S[:, None], Y * S[:, None]
    acc = lambda p, a: 1 - np.abs(p - a).mean() / np.abs(a).mean()
    return {'total_acc': acc(p, a), **{f'h{h}': acc(p[:, h - 1], a[:, h - 1]) for h in [1, 3, 6, 12, 24]}}"""),
    new_markdown_cell('## 7. Train every architecture for every district (leave-district-out) and compare'),
    new_code_cell(r"""rows = []
for district, target_b in DISTRICTS.items():
    pool = [b for dd, bs in DISTRICTS.items() if dd != district for b in bs]
    print(f'\n=== {district}: pseudo-districts of the {len(pool)} other buildings')
    for arch in ARCHS:
        start, models = time.time(), []
        for seed in SEEDS:
            tag = f'{district}_{arch}_s{seed}'
            models.append(export(train_model(arch, seed, pool, tag), tag, district))
        res = evaluate(models, target_b)
        rows.append({'district': district, 'arch': arch, **res, 'gbm_current_acc': GBM_CURRENT[district],
                     'gbm_best_acc': GBM_BEST[district], 'minutes': (time.time() - start) / 60})
        print(f"  {arch}: district-total accuracy {res['total_acc']:.1%} (LightGBM: current {GBM_CURRENT[district]:.1%}, "
              f"best {GBM_BEST[district]:.1%}) | {(time.time() - start) / 60:.1f} min")
        pd.DataFrame(rows).to_csv(os.path.join(MODELS, 'metrics.csv'), index=False)
metrics = pd.DataFrame(rows)
metrics"""),
    new_markdown_cell('## 8. Pack the results (saved in Drive and downloaded)'),
    new_code_cell(r"""with open(os.path.join(MODELS, 'info.json'), 'w') as f:
    json.dump({'W': W, 'H': H, 'D': D, 'N_INPUTS': N_INPUTS, 'seeds': SEEDS, 'archs': list(ARCHS), 'quick': QUICK,
               'torch': torch.__version__}, f, indent=2)
ZIP = os.path.join(WORK, 'F2_district_total_results.zip')
with zipfile.ZipFile(ZIP, 'w') as z:
    for name in os.listdir(MODELS):
        z.write(os.path.join(MODELS, name), name)
print('saved', ZIP)
if IN_COLAB:
    files.download(ZIP)"""),
]
nb = new_notebook(cells=cells, metadata={'accelerator': 'GPU', 'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                                         'colab': {'provenance': [], 'gpuType': 'T4'}})
nbformat.write(nb, str(COLAB / 'F2_district_total.ipynb'))
print('wrote', COLAB / 'F2_district_total.ipynb', '| LightGBM references:', current, best)
