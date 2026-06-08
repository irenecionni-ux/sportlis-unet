"""SWEZarrDataset estratto per permettere DataLoader con num_workers>0.

Features:
  - add_doy=True: 2 canali day-of-year (sin/cos) per fase stagionale
  - patch_size=N: random crop NxN durante training (val/test usa snapshot intero)
  - lat_1d/lon_1d esterni: usato per zarr SPORTLIS (lat/lon 2D nel store)
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SWEZarrDataset(Dataset):
    def __init__(self, ds, input_vars, target_var, mask_var,
                 mean_ds, std_ds, topo_tensor,
                 augment=False, add_doy=False,
                 patch_size=None,
                 lat_1d=None, lon_1d=None):
        """
        Parametri:
          ds: xr.Dataset con dim (time, lat, lon)
          input_vars: list di forcing variables
          target_var, mask_var: nomi delle var target / mask
          mean_ds, std_ds: xr.Dataset con stats per-var (per normalizzazione)
          topo_tensor: np.ndarray shape (C_topo, H, W) gia' normalizzato
          augment: rumore gaussiano sui forcing
          add_doy: aggiunge 2 canali DoY (sin, cos)
          patch_size: se int, restituisce random crop NxN durante __getitem__.
                      Se None, restituisce snapshot intero. Per train: 128.
                      Per val/test: None (snapshot intero per metriche corrette).
          lat_1d, lon_1d: array 1D di coordinate spaziali. Se None, usa
                          ds["lat"].values / ds["lon"].values. Necessari per
                          SPORTLIS dove lat/lon sono 2D nel zarr.
        """
        self.ds = ds
        self.input_vars = input_vars
        self.target_var = target_var
        self.mask_var = mask_var
        self.mean_ds = mean_ds
        self.std_ds = std_ds
        self.n = ds.sizes["time"]
        self.augment = augment
        self.add_doy = add_doy
        self.patch_size = patch_size

        self.topo_tensor = topo_tensor.astype(np.float32)

        # Coordinate spaziali 1D
        if lat_1d is None:
            lat_1d = np.asarray(ds["lat"].values, dtype=np.float32)
        else:
            lat_1d = np.asarray(lat_1d, dtype=np.float32)
        if lon_1d is None:
            lon_1d = np.asarray(ds["lon"].values, dtype=np.float32)
        else:
            lon_1d = np.asarray(lon_1d, dtype=np.float32)

        lat2d, lon2d = np.meshgrid(lat_1d, lon_1d, indexing="ij")
        self.lat2d = lat2d.astype(np.float32)
        self.lon2d = lon2d.astype(np.float32)
        self.lat_mean = float(self.lat2d.mean())
        self.lat_std = float(self.lat2d.std()) if self.lat2d.std() != 0 else 1.0
        self.lon_mean = float(self.lon2d.mean())
        self.lon_std = float(self.lon2d.std()) if self.lon2d.std() != 0 else 1.0

        # pre-compute doy_sin/cos per ogni timestep
        if self.add_doy:
            times = pd.to_datetime(ds["time"].values)
            doy = times.dayofyear.values.astype(np.float32)   # 1..366
            angle = 2.0 * np.pi * doy / 366.0
            self.doy_sin = np.sin(angle).astype(np.float32)
            self.doy_cos = np.cos(angle).astype(np.float32)

    def __len__(self):
        return self.n

    def _random_patch(self, x, y, m):
        """Estrae random crop di patch_size x patch_size con almeno 10% di
        pixel validi nella mask. Se non trova, prende l'ultimo crop tentato."""
        H, W = x.shape[-2], x.shape[-1]
        ph = pw = self.patch_size
        if H < ph or W < pw:
            # snapshot piu' piccolo della patch: ritorna intero
            return x, y, m
        i = j = 0
        for _ in range(8):
            i = np.random.randint(0, H - ph + 1)
            j = np.random.randint(0, W - pw + 1)
            m_p = m[..., i:i+ph, j:j+pw]
            if m_p.sum() > 0.10 * ph * pw:
                break
        return (x[..., i:i+ph, j:j+pw],
                y[..., i:i+ph, j:j+pw],
                m[..., i:i+ph, j:j+pw])

    def __getitem__(self, idx):
        sample = self.ds.isel(time=idx).load()

        x_list = []
        for v in self.input_vars:
            arr = sample[v].values.astype(np.float32)
            mean = self.mean_ds[v].values.astype(np.float32)
            std  = self.std_ds[v].values.astype(np.float32)
            arr = (arr - mean) / std
            x_list.append(arr)

        lat_norm = (self.lat2d - self.lat_mean) / self.lat_std
        lon_norm = (self.lon2d - self.lon_mean) / self.lon_std
        x_list.append(lat_norm)
        x_list.append(lon_norm)

        x = np.stack(x_list, axis=0).astype(np.float32)
        x = np.concatenate([x, self.topo_tensor], axis=0)

        # Day-of-year channels (2): broadcast a HxW
        if self.add_doy:
            H, W = self.lat2d.shape
            doy_s = np.full((1, H, W), self.doy_sin[idx], dtype=np.float32)
            doy_c = np.full((1, H, W), self.doy_cos[idx], dtype=np.float32)
            x = np.concatenate([x, doy_s, doy_c], axis=0)

        y = sample[self.target_var].values.astype(np.float32)
        m = sample[self.mask_var].values.astype(np.float32)
        y = np.log1p(np.maximum(y, 0.0))

        y = y[None, :, :]
        m = m[None, :, :]

        # Random patch crop (solo training)
        if self.patch_size is not None:
            x, y, m = self._random_patch(x, y, m)

        if self.augment:
            n_forcing = len(self.input_vars)
            noise = np.random.randn(n_forcing, *x.shape[1:]).astype(np.float32) * 0.02
            x[:n_forcing] += noise

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(m),
        )
