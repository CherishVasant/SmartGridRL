"""Builds colab/F3/F3_automl_total.ipynb: an automated search (Optuna) over deep model families and hyperparameters for the
DISTRICT-TOTAL load forecast, run on a Colab GPU, leave-district-out, exported as TorchScript for the controller.

Model families: MLP with learned hour / day embeddings and a seasonal-smoothing input (the recipe of 2nd-placed
ambitiousengineers), GRU, TCN (dilated causal convolutions), N-BEATS (residual MLP blocks) and PatchTST (patch
Transformer). For each target district, hyperparameters are chosen by inner cross-validation between the two OTHER
districts (fit on pseudo-districts of one, score on the other's real total, both ways; the loss weights the next
6 hours more, where forecasts pay most); the target district is used once, at the end. The best configurations are
then trained on both other districts with 3 seeds and exported.

Resumable: the Optuna study is an SQLite file in Drive (finished trials are kept, an interrupted one is re-run), and
the final models keep a checkpoint after every epoch, so Run all after a disconnect continues where it stopped.
The notebook embeds colab/F3/total_inputs.py (= colab/F2/total_inputs.py) verbatim.
Usage (from the project folder): python tools/build_colab_f3.py
"""
import shutil
from pathlib import Path

import nbformat
import pandas as pd
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
COLAB = ROOT / 'colab' / 'F3'
COLAB.mkdir(parents=True, exist_ok=True)
for f in ['total_inputs.py', 'citylearn2022_all_buildings.npz']:
    if not (COLAB / f).exists():
        shutil.copy(ROOT / 'colab' / 'F2' / f, COLAB / f)
inputs_src = (COLAB / 'total_inputs.py').read_text(encoding='utf-8')

LAB = ROOT / 'results' / 'forecast' / 'load_lab'
lab = pd.concat([pd.read_csv(LAB / f) for f in ['load_lab.csv', 'load_lab2.csv'] if (LAB / f).exists()])
current = {d: round(float(lab[(lab.district == d) & (lab.forecast == 'sum_gbm')].total_acc.iloc[0]), 3)
           for d in ['train', 'val', 'test']}
best = {d: round(float(lab.loc[lab.district == d, 'total_acc'].max()), 3) for d in ['train', 'val', 'test']}
f2 = {'train': 0.768, 'val': 0.721, 'test': 0.790}           # best F2 deep model per district (colab/F2/total_forecasters)

cells = [
    new_markdown_cell(r"""# F3 – Automated search for the best deep district-total load forecaster (Colab GPU)

**What this does.** Searches over five deep model families – **MLP with hour/day embeddings + seasonal smoothing**
(the 2nd-place team's recipe), **GRU**, **TCN**, **N-BEATS** and **PatchTST** – and their hyperparameters with
**Optuna**, separately for each district, *leave-district-out*: the settings are chosen by cross-validation between
the two other districts, and the district being forecast is used only once, at the end. The best settings are
trained with 3 seeds, compared with LightGBM and F2, and exported as TorchScript for the battery controller.

**How to run (Google Colab)**
1. *Runtime → Change runtime type → **T4 GPU*** → Save.
2. *Runtime → **Run all***. Accept the Google Drive pop-up (everything is stored in **`MyDrive/SmartGridRL_F3/`**).
3. The data file is copied from `MyDrive/SmartGridRL_F2/` or `_F1/` automatically; only if neither exists, the data
   cell asks you to upload **`citylearn2022_all_buildings.npz`** (from the project's `colab/F3/` folder).
4. It takes roughly **2–4 hours** on a T4 (progress: one line per search trial and per training epoch).
   **If Colab disconnects or stops, reconnect and Run all again** – finished trials and epochs are kept in Drive and it
   continues where it stopped. You can also stop it on purpose and resume later.
5. At the end it saves **`MyDrive/SmartGridRL_F3/F3_automl_results.zip`** and downloads it.

**Send back:** `F3_automl_results.zip` → put it in the project's `colab/F3/` folder
(`D:\DEV\Projects\1_ResumeWorthy\SmartGridRL\colab\F3\`). No other files are needed."""),
    new_markdown_cell('## 1. Setup: GPU, Google Drive folder, Optuna'),
    new_code_cell(r"""import torch                      # first: on Windows, torch must be imported before numpy / pandas
import torch.nn as nn
import os, time, json, zipfile, shutil, subprocess, sys
import numpy as np
import pandas as pd
try:
    import optuna
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'optuna'], check=True)
    import optuna

try:
    from google.colab import drive, files
    drive.mount('/content/drive')
    WORK = '/content/drive/MyDrive/SmartGridRL_F3'
    OLD_DATA = ['/content/drive/MyDrive/SmartGridRL_F2/citylearn2022_all_buildings.npz',
                '/content/drive/MyDrive/SmartGridRL_F1/citylearn2022_all_buildings.npz']
    IN_COLAB = True
except ImportError:                 # local test: work in the current folder
    WORK, OLD_DATA, IN_COLAB = '.', [], False
MODELS = os.path.join(WORK, 'total_forecasters')      # finished models (TorchScript), metrics, check vectors
CKPT = os.path.join(WORK, 'checkpoints')              # final-training state after every epoch (for resuming)
os.makedirs(MODELS, exist_ok=True); os.makedirs(CKPT, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
QUICK = os.environ.get('DEEPFC_QUICK') == '1'         # tiny settings for a local code check (off in Colab)
optuna.logging.set_verbosity(optuna.logging.WARNING)
print('device:', device, torch.cuda.get_device_name(0) if device == 'cuda' else '', '| storage:', WORK)"""),
    new_markdown_cell('## 2. Data (taken from the F2 / F1 Drive folders, or uploaded once)'),
    new_code_cell(r"""DATA = os.path.join(WORK, 'citylearn2022_all_buildings.npz')
if not os.path.exists(DATA):
    old = [p for p in OLD_DATA if os.path.exists(p)]
    if old:
        shutil.copy(old[0], DATA)
    elif IN_COLAB:
        print('Upload citylearn2022_all_buildings.npz (from the project colab/F3/ folder)')
        up = files.upload()
        shutil.move(next(iter(up)), DATA)
    else:
        raise FileNotFoundError(DATA)
print('data:', DATA)"""),
    new_markdown_cell('## 3. Input definition (identical to `colab/F3/total_inputs.py`, used by the controller)'),
    new_code_cell(inputs_src),
    new_markdown_cell('## 4. Data sets: pseudo-districts, inner cross-validation folds'),
    new_code_cell(f"""d = np.load(DATA)
LOAD, WEATHER, WPRED, CAL = d['load'], d['weather'], d['weather_pred'], d['cal']
T = len(LOAD)
DISTRICTS = {{'train': list(range(0, 5)), 'val': list(range(5, 10)), 'test': list(range(10, 17))}}
# District-total accuracy (1 - MAE / mean |actual|, horizons 1-24, every 6 h from hour 168) of the references:
GBM_CURRENT = {current}      # current controller (per-building LightGBM, summed)
GBM_BEST = {best}         # best direct LightGBM total model (tools/load_lab*.py)
F2_BEST = {f2}         # best F2 deep model

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

eval_steps = list(range(168, T - H - 1, 6))[::8 if QUICK else 1]
FOLD_STEPS = range(W, T - H - 1, 24 if QUICK else 3)
N_FOLD_PSEUDO = 6 if QUICK else 40
_fold_cache = {{}}

def fold_data(fit_b, val_b):
    # inner fold: pseudo-districts of fit_b (training), the real total of val_b (scoring); cached in memory
    key = (tuple(fit_b), tuple(val_b))
    if key not in _fold_cache:
        Xf, Yf, _ = build(pseudo(fit_b, N_FOLD_PSEUDO, np.random.default_rng(7)), FOLD_STEPS)
        Xv, Yv, Sv = build(LOAD[:, val_b].sum(1, keepdims=True), eval_steps)
        _fold_cache[key] = tuple(torch.tensor(a) for a in (Xf, Yf, Xv, Yv, Sv))
    return _fold_cache[key]
print('inputs per sample:', N_INPUTS)"""),
    new_markdown_cell('## 5. Model families (all take the same flat input vector; output = next 24 h / scale)'),
    new_code_cell(r"""N_SHARED = N_INPUTS - W - H                 # weather, hours, month, day types
SAME = slice(W, W + H)                      # the 28-day same-hour mean of the next 24 h (a seasonal baseline)
# positions of the hour-of-next-hours sin / cos, month and day-type one-hots inside the shared block
I_DAY_NOW = N_INPUTS - 14; I_DAY_TGT = N_INPUTS - 7

def mlp(sizes, dropout, act=nn.GELU):
    layers = []
    for a, b in zip(sizes[:-2], sizes[1:-1]):
        layers += [nn.Linear(a, b), act(), nn.Dropout(dropout)]
    return nn.Sequential(*layers, nn.Linear(sizes[-2], sizes[-1]))

class EmbedMLP(nn.Module):
    # ambitiousengineers-style: lags + seasonal exponential smoothing of the history + learned day embeddings
    def __init__(self, width, depth, dropout, emb, alpha, lags):
        super().__init__()
        self.alpha, self.lags = alpha, lags
        self.day_now, self.day_tgt = nn.Embedding(7, emb), nn.Embedding(7, emb)
        n_in = lags + H + H + (N_SHARED - 14) + 2 * emb
        self.net = mlp([n_in] + [width] * depth + [H], dropout)
    def forward(self, x):
        hist = x[:, :W]
        days = hist.reshape(x.shape[0], W // 24, 24)               # (n, 14 days, 24 h), oldest first
        w = torch.tensor([(1 - self.alpha) ** (W // 24 - 1 - i) for i in range(W // 24)], device=x.device)
        smooth = (days * w[None, :, None]).sum(1) / w.sum()         # seasonal exponential smoothing per hour
        dn = self.day_now(torch.argmax(x[:, I_DAY_NOW:I_DAY_NOW + 7], 1))
        dt = self.day_tgt(torch.argmax(x[:, I_DAY_TGT:I_DAY_TGT + 7], 1))
        z = torch.cat([hist[:, -self.lags:], smooth, x[:, SAME], x[:, W + H:I_DAY_NOW], dn, dt], 1)
        return self.net(z) + x[:, SAME]

class GRUNet(nn.Module):
    def __init__(self, hidden, layers, dropout, width):
        super().__init__()
        self.gru = nn.GRU(1, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = mlp([hidden + N_INPUTS - W, width, H], dropout)
    def forward(self, x):
        out, _ = self.gru(x[:, :W].unsqueeze(-1))
        return self.head(torch.cat([out[:, -1], x[:, W:]], 1)) + x[:, SAME]

class TCN(nn.Module):
    def __init__(self, channels, levels, kernel, dropout, width):
        super().__init__()
        convs, c_in = [], 1
        for i in range(levels):
            convs += [nn.Conv1d(c_in, channels, kernel, dilation=2 ** i, padding=(kernel - 1) * 2 ** i), nn.GELU(),
                      nn.Dropout(dropout)]
            c_in = channels
        self.convs = nn.ModuleList(convs)
        self.head = mlp([channels + N_INPUTS - W, width, H], dropout)
    def forward(self, x):
        z = x[:, :W].unsqueeze(1)
        for layer in self.convs:
            z = layer(z)
            if isinstance(layer, nn.Conv1d):
                z = z[:, :, :W]                                       # causal: drop the right padding
        return self.head(torch.cat([z[:, :, -1], x[:, W:]], 1)) + x[:, SAME]

class NBeats(nn.Module):
    def __init__(self, width, blocks, layers, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([mlp([N_INPUTS] + [width] * layers + [N_INPUTS + H], dropout) for _ in range(blocks)])
    def forward(self, x):
        res, out = x, x[:, SAME].clone()
        for b in self.blocks:
            o = b(res)
            res = res - o[:, :N_INPUTS]
            out = out + o[:, N_INPUTS:]
        return out

class PatchTST(nn.Module):
    def __init__(self, d_model, layers, heads, patch, dropout):
        super().__init__()
        self.patch = patch
        self.embed = nn.Linear(patch, d_model)
        self.ctx = nn.Linear(N_INPUTS - W, d_model)
        self.pos = nn.Parameter(torch.zeros(1, W // patch + 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = mlp([2 * d_model, 256, H], dropout)
    def forward(self, x):
        n = x.shape[0]
        tok = torch.cat([self.ctx(x[:, W:]).unsqueeze(1), self.embed(x[:, :W].reshape(n, W // self.patch, self.patch))], 1)
        z = self.encoder(tok + self.pos)
        return self.head(torch.cat([z[:, 0], z[:, 1:].mean(1)], 1)) + x[:, SAME]

def suggest(trial):
    # one configuration: model family + its hyperparameters + training settings
    fam = trial.suggest_categorical('family', ['embed_mlp', 'gru', 'tcn', 'nbeats', 'patchtst'])
    dr = trial.suggest_float('dropout', 0.0, 0.3)
    if fam == 'embed_mlp':
        cfg = dict(width=trial.suggest_categorical(fam + '_width', [256, 512, 1024]), depth=trial.suggest_int(fam + '_depth', 2, 4),
                   dropout=dr, emb=trial.suggest_categorical(fam + '_emb', [4, 8, 16]),
                   alpha=trial.suggest_float(fam + '_alpha', 0.1, 0.7), lags=trial.suggest_categorical(fam + '_lags', [24, 48, 168]))
    elif fam == 'gru':
        cfg = dict(hidden=trial.suggest_categorical(fam + '_hidden', [64, 128, 256]), layers=trial.suggest_int(fam + '_layers', 1, 2),
                   dropout=dr, width=trial.suggest_categorical(fam + '_width', [256, 512]))
    elif fam == 'tcn':
        cfg = dict(channels=trial.suggest_categorical(fam + '_channels', [32, 64, 128]), levels=trial.suggest_int(fam + '_levels', 4, 8),
                   kernel=trial.suggest_categorical(fam + '_kernel', [2, 3]), dropout=dr,
                   width=trial.suggest_categorical(fam + '_width', [256, 512]))
    elif fam == 'nbeats':
        cfg = dict(width=trial.suggest_categorical(fam + '_width', [256, 512, 1024]), blocks=trial.suggest_int(fam + '_blocks', 2, 6),
                   layers=trial.suggest_int(fam + '_layers', 2, 4), dropout=dr)
    else:
        cfg = dict(d_model=trial.suggest_categorical(fam + '_d_model', [64, 128]), layers=trial.suggest_int(fam + '_layers', 2, 4),
                   heads=trial.suggest_categorical(fam + '_heads', [4, 8]), patch=trial.suggest_categorical(fam + '_patch', [12, 24]),
                   dropout=dr)
    train = dict(lr=trial.suggest_float('lr', 2e-4, 3e-3, log=True), batch=trial.suggest_categorical('batch', [256, 512, 1024]),
                 wd=trial.suggest_float('wd', 1e-6, 1e-3, log=True), w_short=trial.suggest_float('w_short', 1.0, 4.0))
    return fam, cfg, train

FAMILIES = {'embed_mlp': EmbedMLP, 'gru': GRUNet, 'tcn': TCN, 'nbeats': NBeats, 'patchtst': PatchTST}

def make_model(fam, cfg):
    return FAMILIES[fam](**cfg).to(device)

def loss_fn(pred, y, w_short):
    # L1, with the next 6 hours weighted w_short times (they carry most of the value for the controller)
    w = torch.ones(H, device=pred.device); w[:6] = w_short
    return ((pred - y).abs() * w).mean() / w.mean()"""),
    new_markdown_cell('## 6. Search: Optuna per district, inner cross-validation between the other two districts '
                      '(kept in Drive, resumable)'),
    new_code_cell(r"""N_TRIALS = 3 if QUICK else 30            # per district; raise it for a longer search
SEARCH_EPOCHS = 2 if QUICK else 14

def total_acc(model, Xv, Yv, Sv):
    model.eval()
    with torch.no_grad():
        P = torch.cat([model(Xv[i:i + 4096].to(device)).cpu() for i in range(0, len(Xv), 4096)])
    p, a = torch.clamp(P, min=0) * Sv[:, None], Yv * Sv[:, None]
    return float(1 - (p - a).abs().mean() / a.abs().mean())

def fit_fold(fam, cfg, train, Xf, Yf, Xv, Yv, Sv, trial=None, step0=0):
    torch.manual_seed(0)
    model = make_model(fam, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=train['lr'], weight_decay=train['wd'])
    best = -np.inf
    for ep in range(SEARCH_EPOCHS):
        model.train()
        perm = torch.randperm(len(Xf))
        for i in range(0, len(perm), train['batch']):
            b = perm[i:i + train['batch']]
            loss = loss_fn(model(Xf[b].to(device)), Yf[b].to(device), train['w_short'])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        acc = total_acc(model, Xv, Yv, Sv)
        best = max(best, acc)
        if trial is not None:
            trial.report(acc, step0 + ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return best

def search(district):
    others = [dd for dd in DISTRICTS if dd != district]
    folds = [(DISTRICTS[others[0]], DISTRICTS[others[1]]), (DISTRICTS[others[1]], DISTRICTS[others[0]])]
    storage = f"sqlite:///{os.path.join(WORK, 'search.db')}"
    study = optuna.create_study(study_name=f'{district}_quick' if QUICK else district, storage=storage,
                                load_if_exists=True, direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=0),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=4))
    done = len([t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)])
    def objective(trial):
        fam, cfg, train = suggest(trial)
        accs = [fit_fold(fam, cfg, train, *fold_data(fb, vb), trial=trial if k == 0 else None, step0=0)
                for k, (fb, vb) in enumerate(folds)]
        acc = float(np.mean(accs))
        print(f'    trial {trial.number}: {fam} {cfg} lr {train["lr"]:.1e} w_short {train["w_short"]:.1f} '
              f'-> inner accuracy {acc:.3f}', flush=True)
        return acc
    if done < N_TRIALS:
        print(f'  {district}: {done} trials done, running {N_TRIALS - done} more', flush=True)
        study.optimize(objective, n_trials=N_TRIALS - done, catch=(RuntimeError, ValueError))
    return study"""),
    new_markdown_cell('## 7. Final models: the best settings, 3 seeds, fresh pseudo-districts every epoch '
                      '(checkpoint after every epoch)'),
    new_code_cell(r"""SEEDS = [0] if QUICK else [0, 1, 2]
EPOCHS = 2 if QUICK else 40
PATIENCE = 6
N_PSEUDO = 8 if QUICK else 96
STRIDE = 24 if QUICK else 3

def save_atomic(obj, path):
    torch.save(obj, path + '.tmp'); os.replace(path + '.tmp', path)

def config_of(params):
    # rebuild (family, model cfg, train cfg) from a finished trial's parameters
    class Fixed:
        def __init__(self, p): self.p = p
        def suggest_categorical(self, k, _): return self.p[k]
        def suggest_float(self, k, *a, **kw): return self.p[k]
        def suggest_int(self, k, *a, **kw): return self.p[k]
    return suggest(Fixed(params))

def train_final(fam, cfg, train, seed, pool, tag):
    path = os.path.join(CKPT, f'{tag}.pt')
    torch.manual_seed(seed)
    model = make_model(fam, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=train['lr'], weight_decay=train['wd'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    state = {'epoch': 0, 'best': np.inf, 'best_state': None, 'bad': 0, 'done': False}
    if os.path.exists(path):
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state['model']); opt.load_state_dict(state['opt']); sched.load_state_dict(state['sched'])
        print(f'    resuming {tag} at epoch {state["epoch"]}' + (' (already finished)' if state['done'] else ''))
    Xv, Yv, _ = build(pseudo(pool, 8 if QUICK else 32, np.random.default_rng(999)), range(W, T - H - 1, 48 if QUICK else 12))
    Xv, Yv = torch.tensor(Xv).to(device), torch.tensor(Yv).to(device)
    while not state['done'] and state['epoch'] < EPOCHS:
        rng = np.random.default_rng(seed * 1000 + state['epoch'])
        Xt, Yt, _ = build(pseudo(pool, N_PSEUDO, rng), range(W + int(rng.integers(STRIDE)), T - H - 1, STRIDE))
        Xt, Yt = torch.tensor(Xt), torch.tensor(Yt)
        model.train()
        perm = torch.randperm(len(Xt), generator=torch.Generator().manual_seed(seed * 1000 + state['epoch']))
        for i in range(0, len(perm), train['batch']):
            b = perm[i:i + train['batch']]
            loss = loss_fn(model(Xt[b].to(device)), Yt[b].to(device), train['w_short'])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            v = np.mean([loss_fn(model(Xv[i:i + 4096]), Yv[i:i + 4096], train['w_short']).item()
                         for i in range(0, len(Xv), 4096)])
        sched.step(v)
        state['epoch'] += 1
        if v < state['best'] - 1e-5:
            state.update(best=v, bad=0, best_state={k: x.detach().cpu().clone() for k, x in model.state_dict().items()})
        else:
            state['bad'] += 1
        state['done'] = state['bad'] >= PATIENCE or state['epoch'] >= EPOCHS
        save_atomic(dict(state, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict()), path)
        print(f'      {tag} epoch {state["epoch"]}: val loss {v:.4f}', flush=True)
    model.load_state_dict(state['best_state'])
    return model.eval()

def export(model, tag, district):
    X, _ = inputs(LOAD[:, DISTRICTS[district]].sum(1, keepdims=True), WEATHER, WPRED, CAL, 1000)
    example = torch.tensor(X)
    m_cpu = model.to('cpu').eval()
    with torch.no_grad():
        traced = torch.jit.trace(m_cpu, example, check_trace=False)
        eager, exported = m_cpu(example).numpy(), traced(example).numpy()
    assert np.allclose(eager, exported, atol=1e-4), f'{tag}: export mismatch'
    traced.save(os.path.join(MODELS, f'{tag}.pt'))
    np.save(os.path.join(MODELS, f'check_{tag}.npy'), exported)
    return model.to(device)

def evaluate(models, buildings):
    x = LOAD[:, buildings].sum(1, keepdims=True)
    X, Y, S = build(x, eval_steps)
    with torch.no_grad():
        P = np.mean([m(torch.tensor(X).to(device)).cpu().numpy() for m in models], axis=0)
    p, a = np.maximum(P, 0) * S[:, None], Y * S[:, None]
    acc = lambda p, a: 1 - np.abs(p - a).mean() / np.abs(a).mean()
    return {'total_acc': acc(p, a), 'acc_h1_6': acc(p[:, :6], a[:, :6]),
            **{f'h{h}': acc(p[:, h - 1], a[:, h - 1]) for h in [1, 3, 6, 12, 24]}}"""),
    new_markdown_cell('## 8. Run: search, then final training and evaluation, for every district'),
    new_code_cell(r"""rows, configs = [], {}
for district, target_b in DISTRICTS.items():
    pool = [b for dd, bs in DISTRICTS.items() if dd != district for b in bs]
    print(f'\n=== {district}: search on the {len(pool)} other buildings')
    study = search(district)
    ranked = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE], key=lambda t: -t.value)
    best_t = ranked[0]
    fam, cfg, train = config_of(best_t.params)
    configs[district] = {'family': fam, 'cfg': cfg, 'train': train, 'inner_acc': best_t.value,
                         'top5': [(t.params['family'], round(t.value, 4)) for t in ranked[:5]]}
    print(f'  best: {fam} {cfg} {train} (inner accuracy {best_t.value:.3f})')
    start, models = time.time(), []
    for seed in SEEDS:
        tag = f'{district}_f3_s{seed}'
        models.append(export(train_final(fam, cfg, train, seed, pool, tag), tag, district))
    res = evaluate(models, target_b)
    rows.append({'district': district, 'family': fam, **res, 'gbm_current_acc': GBM_CURRENT[district],
                 'gbm_best_acc': GBM_BEST[district], 'f2_best_acc': F2_BEST[district], 'minutes': (time.time() - start) / 60})
    print(f"  F3 {fam}: district-total accuracy {res['total_acc']:.1%} (next 6 h {res['acc_h1_6']:.1%}) | LightGBM: "
          f"current {GBM_CURRENT[district]:.1%}, best {GBM_BEST[district]:.1%} | F2 {F2_BEST[district]:.1%}")
    pd.DataFrame(rows).to_csv(os.path.join(MODELS, 'metrics.csv'), index=False)
    with open(os.path.join(MODELS, 'configs.json'), 'w') as f:
        json.dump(configs, f, indent=2, default=str)
metrics = pd.DataFrame(rows)
metrics"""),
    new_markdown_cell('## 9. Pack the results (saved in Drive and downloaded)'),
    new_code_cell(r"""with open(os.path.join(MODELS, 'info.json'), 'w') as f:
    json.dump({'W': W, 'H': H, 'D': D, 'N_INPUTS': N_INPUTS, 'seeds': SEEDS, 'n_trials': N_TRIALS, 'quick': QUICK,
               'torch': torch.__version__}, f, indent=2)
ZIP = os.path.join(WORK, 'F3_automl_results.zip')
with zipfile.ZipFile(ZIP, 'w') as z:
    for name in os.listdir(MODELS):
        z.write(os.path.join(MODELS, name), name)
print('saved', ZIP)
if IN_COLAB:
    files.download(ZIP)"""),
]
nb = new_notebook(cells=cells, metadata={'accelerator': 'GPU', 'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                                         'colab': {'provenance': [], 'gpuType': 'T4'}})
nbformat.write(nb, str(COLAB / 'F3_automl_total.ipynb'))
print('wrote', COLAB / 'F3_automl_total.ipynb')
