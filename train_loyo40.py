"""
train_loyo40.py — LOYO 40-fold training script per PBS job array.

Usage:
    python train_loyo40.py --fold_idx 0      # train fold 0 (test=1985)
    python train_loyo40.py --fold_idx 39     # train fold 39 (test=2024)
    python train_loyo40.py --fold_idx all    # tutti i fold in sequenza (JupyterHub)

Il fold_idx corrisponde a: test_year = 1985 + fold_idx
Il checkpoint viene salvato in OUTPUT_DIR/best_unet_extended_LOYO_test{year}.pt
"""

import argparse, gc, json, shutil, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import xarray as xr
import netCDF4 as nc

# ── Parse argomenti ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--fold_idx', type=str, default='all',
                    help='Indice fold 0-39 oppure "all" per tutti in sequenza')
parser.add_argument('--skip_existing', action='store_true', default=True,
                    help='Salta fold con checkpoint già presente (default True)')
args = parser.parse_args()

# ── PATHS (NCAR Casper) ───────────────────────────────────────────────────────
NCAR_HOME         = Path('/glade/u/home/cionni')
NCAR_SCRATCH      = NCAR_HOME / 'derecho_scratch'
SPORTLIS_DATA_DIR = NCAR_SCRATCH / 'sportlis_swe'
NARR_RAW_DIR      = NCAR_SCRATCH / 'narr_extended_raw'
ZARR_DIR          = NCAR_SCRATCH / 'zarr_extended'
MEMMAP_DIR        = NCAR_SCRATCH / 'memmap_extended'
PROJECT_DIR       = NCAR_HOME / 'unet_sportlis'
OUTPUT_DIR        = PROJECT_DIR / 'output_extended'
AUX_DIR           = PROJECT_DIR / 'auxiliary'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATIC_FILE = AUX_DIR / 'sportlis_static_extended.nc'
CANOPY_FILE = AUX_DIR / 'sportlis_canopy_extended_3km.nc'

# ── IPERPARAMETRI ─────────────────────────────────────────────────────────────
EXT_H, EXT_W     = 484, 698
LEAN_TEMPORAL_VARS = ['precip_7d','precip_30d','precip_60d','precip_wytd',
                       'tair_30d_mean','tair_30d_max','degree_day_30d']
TOPO_VARS  = ['elevation','slope','aspect_sin','aspect_cos',
              'tpi_short','tpi_long','canopy_fraction']
INPUT_VARS = LEAN_TEMPORAL_VARS
TARGET_VAR = 'swe_target_filled'
MASK_VAR   = 'swe_mask'

N_FEAT          = len(INPUT_VARS)        # 7
N_COORD         = 2                      # lat_norm, lon_norm
N_TOPO          = len(TOPO_VARS)         # 7
N_IN_CHANNELS   = N_FEAT + N_COORD + N_TOPO  # 16

BATCH_SIZE      = 8
LR              = 1e-4
WEIGHT_DECAY    = 1e-4
EPOCHS          = 50
PATIENCE        = 12
DROPOUT_P       = 0.1
PATCH_SIZE      = 128
N_PATCHES_EPOCH = 3000
TIME_STRIDE     = 2
USE_AMP         = True
AUGMENT_NOISE   = True
NUM_WORKERS     = 4
CHECKPOINT_TEMPLATE = 'best_unet_extended_LOYO_test{test}.pt'

YEARS_ALL = list(range(1985, 2025))   # 40 anni

# ── Fold index ────────────────────────────────────────────────────────────────
if args.fold_idx == 'all':
    fold_indices = list(range(40))
else:
    fold_indices = [int(args.fold_idx)]

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if device.type == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Memoria: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

# ── UNet (identica al notebook) ───────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8, dropout=0.0):
        super().__init__()
        g = min(groups, out_ch)
        while out_ch % g != 0: g -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch), nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch), nn.GELU(),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch, base=32, dropout=0.1):
        super().__init__()
        b = base
        self.enc1 = DoubleConv(in_ch, b,   dropout=dropout)
        self.enc2 = DoubleConv(b,    b*2,  dropout=dropout)
        self.enc3 = DoubleConv(b*2,  b*4,  dropout=dropout)
        self.enc4 = DoubleConv(b*4,  b*8,  dropout=dropout)
        self.bot  = DoubleConv(b*8,  b*16, dropout=dropout)
        self.up4  = nn.ConvTranspose2d(b*16, b*8,  2, stride=2)
        self.dec4 = DoubleConv(b*16, b*8,  dropout=dropout)
        self.up3  = nn.ConvTranspose2d(b*8,  b*4,  2, stride=2)
        self.dec3 = DoubleConv(b*8,  b*4,  dropout=dropout)
        self.up2  = nn.ConvTranspose2d(b*4,  b*2,  2, stride=2)
        self.dec2 = DoubleConv(b*4,  b*2,  dropout=dropout)
        self.up1  = nn.ConvTranspose2d(b*2,  b,    2, stride=2)
        self.dec1 = DoubleConv(b*2,  b,    dropout=dropout)
        self.head = nn.Conv2d(b, 1, 1)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bot(self.pool(e4))
        d  = self.dec4(torch.cat([self.up4(b),  e4], 1))
        d  = self.dec3(torch.cat([self.up3(d),  e3], 1))
        d  = self.dec2(torch.cat([self.up2(d),  e2], 1))
        d  = self.dec1(torch.cat([self.up1(d),  e1], 1))
        return torch.relu(self.head(d))

# ── Loss ──────────────────────────────────────────────────────────────────────
def masked_hybrid_loss(pred, target, mask, alpha=2.0):
    m = mask[:,0] > 0
    if not m.any(): return torch.tensor(0.0, device=pred.device, requires_grad=True)
    p = pred[:,0][m]; t = target[:,0][m]
    mse = ((p - t)**2).mean()
    mae = (p - t).abs().mean()
    return mse + alpha * mae

# ── Dataset memmap ────────────────────────────────────────────────────────────
class SWEMemmapDataset(Dataset):
    def __init__(self, year_list, mean_arr, std_arr, topo_t, lat_n, lon_n,
                 patch=128, n_patches=3000, stride=2, augment=True):
        self.years    = year_list
        self.mean     = mean_arr
        self.std      = std_arr
        self.topo     = topo_t        # (N_TOPO, H, W) numpy
        self.lat_n    = lat_n         # (H, W)
        self.lon_n    = lon_n         # (H, W)
        self.patch    = patch
        self.n        = n_patches
        self.augment  = augment
        self.mms      = {}            # lazy-load
        self.msks     = {}
        self.ts       = {}            # T per anno
        T_tot = 0
        for y in year_list:
            p = MEMMAP_DIR / f'y{y}_feat.npy'
            if p.exists():
                mm = np.lib.format.open_memmap(str(p), mode='r')
                self.ts[y] = mm.shape[0]; T_tot += mm.shape[0] // stride
                del mm
        self.T_total = T_tot
        print(f'SWEMemmapDataset: {len(year_list)} anni  {T_tot} ts  n_patches={n_patches}')

    def _get_mm(self, year):
        if year not in self.mms:
            self.mms[year]  = np.lib.format.open_memmap(
                str(MEMMAP_DIR / f'y{year}_feat.npy'), mode='r')
            self.msks[year] = np.lib.format.open_memmap(
                str(MEMMAP_DIR / f'y{year}_mask.npy'), mode='r')
        return self.mms[year], self.msks[year]

    def __len__(self): return self.n

    def __getitem__(self, _):
        # Campiona anno casuale con probabilità proporzionale a T
        weights = np.array([self.ts.get(y, 0) for y in self.years], dtype=float)
        weights /= weights.sum()
        year = np.random.choice(self.years, p=weights)
        mm, msk = self._get_mm(year)
        T = mm.shape[0]
        ti = np.random.randint(0, T)
        H, W = mm.shape[2], mm.shape[3]
        ph = pw = self.patch
        i = np.random.randint(0, H - ph)
        j = np.random.randint(0, W - pw)
        feat  = mm[ti,  :, i:i+ph, j:j+pw].copy()   # (N_FEAT+1, ph, pw)
        mask  = msk[ti,    i:i+ph, j:j+pw].copy()   # (ph, pw)
        x = (feat[:N_FEAT] - self.mean.reshape(-1,1,1)) / self.std.reshape(-1,1,1)
        # Aggiungi lat/lon
        lat_p = self.lat_n[i:i+ph, j:j+pw][None]
        lon_p = self.lon_n[i:i+ph, j:j+pw][None]
        # Aggiungi topo
        topo_p = self.topo[:, i:i+ph, j:j+pw]
        X = np.concatenate([x, lat_p, lon_p, topo_p], axis=0).astype(np.float32)
        y = np.log1p(np.maximum(feat[N_FEAT:N_FEAT+1], 0.0)).astype(np.float32)
        m = mask[None].astype(np.float32)
        if self.augment:
            noise = np.random.normal(0, 0.01, X.shape).astype(np.float32)
            X = X + noise
        return torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(m)

# ── Carica statiche (topo, lat, lon) ─────────────────────────────────────────
def load_static():
    ds = xr.open_dataset(STATIC_FILE)
    lat_1d = ds['lat'].values.astype(np.float32)
    lon_1d = ds['lon'].values.astype(np.float32)
    lat2d, lon2d = np.meshgrid(lat_1d, lon_1d, indexing='ij')
    lat_n = ((lat2d - lat2d.mean()) / (lat2d.std() or 1.0)).astype(np.float32)
    lon_n = ((lon2d - lon2d.mean()) / (lon2d.std() or 1.0)).astype(np.float32)
    elev   = ds['elevation'].values.astype(np.float32)
    slope  = ds['slope'].values.astype(np.float32)
    asp_s  = ds['aspect_sin'].values.astype(np.float32)
    asp_c  = ds['aspect_cos'].values.astype(np.float32)
    tpi_s  = ds['tpi_short'].values.astype(np.float32)
    tpi_l  = ds['tpi_long'].values.astype(np.float32)
    ds.close()
    ds_c = xr.open_dataset(CANOPY_FILE)
    canopy = ds_c['canopy_fraction'].values.astype(np.float32)
    ds_c.close()
    for arr in [elev, slope, asp_s, asp_c, tpi_s, tpi_l, canopy]:
        mu, sd = arr.mean(), arr.std()
        arr[:] = (arr - mu) / (sd if sd > 0 else 1.0)
    topo = np.stack([elev, slope, asp_s, asp_c, tpi_s, tpi_l, canopy], axis=0)
    return lat_n, lon_n, topo

# ── Calcola statistiche normalizzazione ───────────────────────────────────────
def compute_norm_stats(train_years):
    cache = OUTPUT_DIR / f'norm_stats_train{"_".join(map(str,sorted(train_years)[:3]))}_n{len(train_years)}.npz'
    # Usa cache per test_year come chiave
    if cache.exists():
        sc = np.load(str(cache))
        return sc['mean_arr'].astype(np.float32), sc['std_arr'].astype(np.float32)
    n = np.zeros(N_FEAT); s = np.zeros(N_FEAT); s2 = np.zeros(N_FEAT); count = 0
    for y in train_years:
        p = MEMMAP_DIR / f'y{y}_feat.npy'
        if not p.exists(): continue
        mm = np.lib.format.open_memmap(str(p), mode='r')
        for ch in range(N_FEAT):
            flat = mm[:, ch].ravel().astype(np.float64)
            n[ch] += len(flat); s[ch] += flat.sum(); s2[ch] += (flat**2).sum()
        del mm; gc.collect()
    mean_arr = (s / np.maximum(n, 1)).astype(np.float32)
    std_arr  = np.sqrt(np.maximum(s2 / np.maximum(n, 1) - mean_arr**2, 1e-8)).astype(np.float32)
    np.savez(str(cache), mean_arr=mean_arr, std_arr=std_arr)
    return mean_arr, std_arr

# ── Eval su anno ──────────────────────────────────────────────────────────────
def eval_fold(test_year, ckpt_path, mean_arr, std_arr, lat_n, lon_n, topo):
    mm_path  = MEMMAP_DIR / f'y{test_year}_feat.npy'
    msk_path = MEMMAP_DIR / f'y{test_year}_mask.npy'
    if not mm_path.exists(): return None
    mm  = np.lib.format.open_memmap(str(mm_path),  mode='r')
    msk = np.lib.format.open_memmap(str(msk_path), mode='r')
    T   = mm.shape[0]
    model = UNet(N_IN_CHANNELS, dropout=DROPOUT_P).to(device)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
    model.eval()
    sa = sb = sq = sn = 0.0
    with torch.no_grad():
        for ti in range(0, T, 8):
            tc = min(8, T - ti)
            feat = mm[ti:ti+tc].astype(np.float32)
            mask = msk[ti:ti+tc].astype(bool)
            x = (feat[:, :N_FEAT] - mean_arr.reshape(1,-1,1,1)) / std_arr.reshape(1,-1,1,1)
            lat_b = np.tile(lat_n[None,None], (tc,1,1,1))
            lon_b = np.tile(lon_n[None,None], (tc,1,1,1))
            topo_b = np.tile(topo[None],      (tc,1,1,1))
            X = np.concatenate([x, lat_b, lon_b, topo_b], axis=1)
            pred = model(torch.from_numpy(X).to(device)).cpu().numpy()[:,0]
            pred_mm = np.expm1(np.clip(pred, 0, np.log1p(3000)))
            obs_mm  = np.clip(feat[:, N_FEAT], 0, 3000)
            m = mask if mask.ndim == 3 else mask[:,0]
            for b in range(tc):
                mk = m[b]
                if not mk.any(): continue
                e = pred_mm[b][mk] - obs_mm[b][mk]
                sa += np.abs(e).sum(); sb += e.sum()
                sq += (e**2).sum(); sn += mk.sum()
    del model, mm, msk; gc.collect(); torch.cuda.empty_cache()
    n = max(sn, 1)
    return dict(mae=sa/n, bias=sb/n, rmse=np.sqrt(sq/n), n_pix=int(sn))

# ── MAIN TRAINING LOOP ────────────────────────────────────────────────────────
print(f'\nAnni disponibili: {YEARS_ALL[0]}–{YEARS_ALL[-1]}  ({len(YEARS_ALL)} anni)')
print(f'Fold da eseguire: {fold_indices}')

lat_n, lon_n, topo = load_static()
print('Statiche caricate.')

results = []
metrics_csv = OUTPUT_DIR / 'loyo40_fold_metrics.csv'

for fi in fold_indices:
    test_year = 1985 + fi
    val_year  = test_year + 1 if test_year < 2024 else 1985
    train_yrs = [y for y in YEARS_ALL if y not in (test_year, val_year)]

    ckpt     = OUTPUT_DIR / CHECKPOINT_TEMPLATE.format(test=test_year)
    res_ckpt = OUTPUT_DIR / f'resume_unet_extended_LOYO_test{test_year}.pt'

    print(f'\n{"="*65}')
    print(f' FOLD {fi+1}/40  test={test_year}  val={val_year}  train({len(train_yrs)}): {train_yrs[0]}..{train_yrs[-1]}')

    # Skip se checkpoint già presente e nessun resume
    if args.skip_existing and ckpt.exists() and not res_ckpt.exists():
        print(f' ✓ Checkpoint già presente, skip training.')
    else:
        # ── Statistiche normalizzazione ──────────────────────────────────────
        stats_cache = OUTPUT_DIR / f'norm_stats_test{test_year}.npz'
        if stats_cache.exists():
            sc = np.load(str(stats_cache))
            mean_arr = sc['mean_arr'].astype(np.float32)
            std_arr  = sc['std_arr'].astype(np.float32)
        else:
            mean_arr, std_arr = compute_norm_stats(train_yrs)
            np.savez(str(stats_cache), mean_arr=mean_arr, std_arr=std_arr)
        print(f' Stats: mean={mean_arr[:3].round(2)}  std={std_arr[:3].round(2)}')

        # ── Dataset + DataLoader ─────────────────────────────────────────────
        train_yrs_ok = [y for y in train_yrs if (MEMMAP_DIR / f'y{y}_feat.npy').exists()]
        val_yrs_ok   = [val_year] if (MEMMAP_DIR / f'y{val_year}_feat.npy').exists() else []

        if len(train_yrs_ok) < 5:
            print(f' SKIP: solo {len(train_yrs_ok)} memmap disponibili'); continue

        tr_ds = SWEMemmapDataset(train_yrs_ok, mean_arr, std_arr, topo, lat_n, lon_n,
                                  patch=PATCH_SIZE, n_patches=N_PATCHES_EPOCH,
                                  stride=TIME_STRIDE, augment=AUGMENT_NOISE)
        vl_ds = SWEMemmapDataset(val_yrs_ok, mean_arr, std_arr, topo, lat_n, lon_n,
                                  patch=None, n_patches=500, stride=4, augment=False) \
                if val_yrs_ok else None

        tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True)

        # ── Modello ──────────────────────────────────────────────────────────
        model  = UNet(N_IN_CHANNELS, dropout=DROPOUT_P).to(device)
        opt    = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched  = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=3)
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and device.type == 'cuda')

        start_epoch = 0; best_val = float('inf'); stale = 0

        if res_ckpt.exists():
            print(f' *** RESUME da {res_ckpt.name} ***')
            state = torch.load(res_ckpt, map_location=device)
            model.load_state_dict(state['model'])
            opt.load_state_dict(state['optimizer'])
            sched.load_state_dict(state['scheduler'])
            if 'scaler' in state: scaler.load_state_dict(state['scaler'])
            start_epoch = state['epoch'] + 1
            best_val    = state['best_val']
            stale       = state['stale']
            print(f'   → epoch {start_epoch+1}/{EPOCHS}  best_val={best_val:.4f}  stale={stale}')

        t0 = time.time()
        for ep in range(start_epoch, EPOCHS):
            model.train(); rn = 0.0; nb_b = 0
            for Xb, yb, mb in tr_ld:
                Xb = Xb.to(device); yb = yb.to(device); mb = mb.to(device)
                opt.zero_grad()
                with torch.cuda.amp.autocast(enabled=USE_AMP and device.type == 'cuda'):
                    pred = model(Xb)
                    if torch.isnan(pred).any(): continue
                    loss = masked_hybrid_loss(pred, yb, mb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                rn += loss.item(); nb_b += 1
            tr_loss = rn / max(nb_b, 1)

            # Validation
            vl_loss = tr_loss  # fallback se no val set
            if vl_ds is not None:
                vl_ld = DataLoader(vl_ds, batch_size=2, shuffle=False, num_workers=0)
                model.eval(); rv = 0.0; nv = 0
                with torch.no_grad():
                    for Xb, yb, mb in vl_ld:
                        with torch.cuda.amp.autocast(enabled=USE_AMP and device.type == 'cuda'):
                            pred = model(Xb.to(device))
                        if torch.isnan(pred).any(): continue
                        rv += masked_hybrid_loss(pred, yb.to(device), mb.to(device)).item()
                        nv += 1
                vl_loss = rv / max(nv, 1)

            improved = vl_loss < best_val
            if improved:
                best_val = vl_loss; stale = 0
                torch.save(model.state_dict(), ckpt)
            else:
                stale += 1

            elapsed = (time.time() - t0) / 60
            print(f'  E{ep+1:02d}/{EPOCHS} train={tr_loss:.4f} val={vl_loss:.4f} '
                  f'lr={opt.param_groups[0]["lr"]:.1e} '
                  f'{"*" if improved else f"stale {stale}/{PATIENCE}"} '
                  f'[{elapsed:.1f}min]')

            # Resume checkpoint
            torch.save({'epoch': ep, 'model': model.state_dict(),
                        'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(),
                        'scaler': scaler.state_dict(), 'best_val': best_val,
                        'stale': stale, 'test_year': test_year}, res_ckpt)

            sched.step(vl_loss)
            if stale >= PATIENCE:
                print(f'  Early stop.'); break

        if res_ckpt.exists(): res_ckpt.unlink()
        del model, opt, sched, scaler, tr_ds, tr_ld; gc.collect()
        torch.cuda.empty_cache()

    # ── Eval ─────────────────────────────────────────────────────────────────
    if ckpt.exists():
        stats_cache = OUTPUT_DIR / f'norm_stats_test{test_year}.npz'
        if stats_cache.exists():
            sc = np.load(str(stats_cache))
            mean_arr = sc['mean_arr'].astype(np.float32)
            std_arr  = sc['std_arr'].astype(np.float32)
        else:
            mean_arr, std_arr = compute_norm_stats(
                [y for y in YEARS_ALL if y not in (test_year, val_year)])
        metrics = eval_fold(test_year, ckpt, mean_arr, std_arr, lat_n, lon_n, topo)
        if metrics:
            metrics['test_year'] = test_year; metrics['val_year'] = val_year
            results.append(metrics)
            print(f' Eval: MAE={metrics["mae"]:.2f} mm  bias={metrics["bias"]:.2f}  '
                  f'RMSE={metrics["rmse"]:.2f}  n_pix={metrics["n_pix"]:,}')
            # Append a CSV
            row = pd.DataFrame([metrics])
            row.to_csv(metrics_csv, mode='a',
                       header=not metrics_csv.exists(), index=False)

print(f'\n{"="*65}')
print(f' COMPLETATO: {len(results)} fold')
if results:
    df = pd.DataFrame(results)
    print(f' MAE medio = {df.mae.mean():.2f} ± {df.mae.std():.2f} mm')
    print(f' bias medio = {df.bias.mean():.2f} mm')
