#!/usr/bin/env python3
"""
Costruisce sportlis_pp_{year}.zarr: forcing NLDAS interpolato sul grid
SPORTLIS (243x223) + SWE target alta-res + maschere + snowy flags.

Ingresso:
  - NLDAS zarr (.../nldas_pp_{Y}.zarr)   <-- forcing a 0.125 deg (59x53)
  - SPORTLIS   (sportlis_swe_2017010100_2021123123.nc4)
               data[time, lat, lon, feat=1]  = SWE (WEASD)
               latitude/longitude 2D (curvilineo)
               static group: elevation/slope/aspect

Uscita: sportlis_pp_{Y}.zarr con stessa struttura / nomi degli NLDAS zarr
(cosi' il notebook esistente va avanti con modifiche minime):

  dims:  time, lat=243, lon=223
  vars:  precip, precip_24h, tair, tair_24h_mean, qair, psurf,
         wind_u, wind_v, wind_speed, swdown, lwdown,           (forcing interp)
         swe_target, swe_target_filled, swe_mask,              (target + mask)
         is_snowy_time, snow_fraction                          (flags per timestep)
         latitude, longitude                                   (2D, riferimento)

Usage:
  python build_sportlis_zarr.py --year 2017
  python build_sportlis_zarr.py --years 2017 2018 2019 2020 2021
"""
from __future__ import annotations
import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr
import zarr

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("build_sportlis")

# --------------------------------------------------------------------------- #
# Percorsi default (coincidono con la setup del notebook)
NLDAS_DIR_DEFAULT    = Path("/Users/irene/PROJECTS.index/Hydrology/projects/sportlis_project/prepared_pp_nldas")
SPORTLIS_FILE_DEFAULT = Path("/Users/irene/PROJECTS.index/Hydrology/projects/sportlis_project/sportlis_swe_2017010100_2021123123.nc4")
OUT_DIR_DEFAULT       = Path("/Users/irene/PROJECTS.index/Hydrology/projects/sportlis_project/prepared_pp_sportlis")

FORCING_VARS = [
    "precip", "precip_24h",
    "tair", "tair_24h_mean",
    "qair", "psurf",
    "wind_u", "wind_v", "wind_speed",
    "swdown", "lwdown",
]

# Soglia snowy (stesso criterio usato dallo script NLDAS)
SNOWY_SWE_THRESHOLD_MM = 5.0
SNOWY_MIN_PIXELS       = 100       # >= N pixel con SWE>threshold per flaggare snowy

# Batch size temporale per il processing (compromesso RAM/IO)
TIME_CHUNK = 480   # ~20 giorni
# chunking finale dello zarr
OUT_CHUNKS = {"time": 720, "lat": 243, "lon": 223}


# --------------------------------------------------------------------------- #
def parse_timestamps(ts_array) -> pd.DatetimeIndex:
    """Converte l'array di stringhe YYYYMMDDHH in DatetimeIndex."""
    as_str = [str(s) for s in ts_array]
    return pd.to_datetime(as_str, format="%Y%m%d%H")


def open_sportlis(path: Path):
    """Apre il file SPORTLIS con netCDF4 (mantiene i gruppi)."""
    return nc.Dataset(path)


def load_sportlis_static(sp_ds) -> dict:
    """Carica il gruppo static + ritorna dizionario con domain_mask e static raw.
    Applica NaN ai -9999; calcola aspect_sin / aspect_cos."""
    g = sp_ds.groups["static"]
    elev  = np.asarray(g.variables["elevation"][:], dtype=np.float32)
    slope = np.asarray(g.variables["slope"][:],     dtype=np.float32)
    aspect= np.asarray(g.variables["aspect"][:],    dtype=np.float32)

    domain = (elev > -9000) & (slope > -9000) & (aspect > -9000)
    # sostituisco i -9999 con NaN per trasparenza, poi tu li filleri in normalize
    elev   = np.where(domain, elev,   np.nan).astype(np.float32)
    slope  = np.where(domain, slope,  np.nan).astype(np.float32)
    aspect = np.where(domain, aspect, np.nan).astype(np.float32)

    # aspect in radianti (range 0..~2pi), calcolo sin/cos sul dominio
    aspect_sin = np.where(domain, np.sin(aspect), np.nan).astype(np.float32)
    aspect_cos = np.where(domain, np.cos(aspect), np.nan).astype(np.float32)

    n_valid = int(domain.sum())
    log.info(f"Static SPORTLIS: shape={elev.shape}  domain valid pixels={n_valid}/"
             f"{elev.size} ({100*n_valid/elev.size:.1f}%)")

    return {
        "elevation":  elev,
        "slope":      slope,
        "aspect":     aspect,
        "aspect_sin": aspect_sin,
        "aspect_cos": aspect_cos,
        "domain":     domain.astype(np.uint8),
    }


def save_sportlis_static(static_dict: dict, lat2d: np.ndarray, lon2d: np.ndarray,
                         out_path: Path):
    """Salva lo static SPORTLIS come NetCDF che il notebook potra' leggere."""
    ds = xr.Dataset(
        data_vars={
            "elevation":  (("lat", "lon"), static_dict["elevation"]),
            "slope":      (("lat", "lon"), static_dict["slope"]),
            "aspect":     (("lat", "lon"), static_dict["aspect"]),
            "aspect_sin": (("lat", "lon"), static_dict["aspect_sin"]),
            "aspect_cos": (("lat", "lon"), static_dict["aspect_cos"]),
            "domain":     (("lat", "lon"), static_dict["domain"]),
            "latitude":   (("lat", "lon"), lat2d.astype(np.float32)),
            "longitude":  (("lat", "lon"), lon2d.astype(np.float32)),
        },
        attrs={
            "description": "Static predictors from SPORTLIS (elevation, slope, aspect)",
            "source": "sportlis_swe_2017010100_2021123123.nc4",
            "note": "domain=1 where all static are valid (no -9999 fill)",
        },
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # clobber: we rewrite each call (fast)
    if out_path.exists():
        out_path.unlink()
    ds.to_netcdf(out_path)
    log.info(f"Static file scritto: {out_path}")


# --------------------------------------------------------------------------- #
def build_one_year(year: int, nldas_dir: Path, sportlis_file: Path,
                   out_dir: Path, overwrite: bool = False):
    out_path = out_dir / f"sportlis_pp_{year}.zarr"
    if out_path.exists():
        if overwrite:
            log.warning(f"Rimuovo esistente: {out_path}")
            shutil.rmtree(out_path)
        else:
            log.warning(f"{out_path} esiste gia' -> skip (usa --overwrite per rifare)")
            return

    nldas_path = nldas_dir / f"nldas_pp_{year}.zarr"
    if not nldas_path.exists():
        raise FileNotFoundError(nldas_path)

    log.info(f"=== anno {year} ===")
    log.info(f"  NLDAS    : {nldas_path}")
    log.info(f"  SPORTLIS : {sportlis_file}")
    log.info(f"  output   : {out_path}")

    # 1) Apri SPORTLIS e costruisci indice tempi
    sp = open_sportlis(sportlis_file)
    try:
        sp_ts = parse_timestamps(sp.variables["timestamps"][:])
        sp_lat2d = np.asarray(sp.variables["latitude"][:],  dtype=np.float32)
        sp_lon2d = np.asarray(sp.variables["longitude"][:], dtype=np.float32)
        H, W = sp_lat2d.shape

        mask_year = (sp_ts.year == year)
        sp_idx_year = np.where(mask_year)[0]
        if sp_idx_year.size == 0:
            raise ValueError(f"Nessun timestep SPORTLIS per l'anno {year}")
        log.info(f"  SPORTLIS timesteps anno: {sp_idx_year.size}")

        # 2) Apri NLDAS zarr
        nl = xr.open_zarr(nldas_path, consolidated=True, chunks={})
        nl_lat = np.asarray(nl["lat"].values, dtype=np.float32)
        nl_lon = np.asarray(nl["lon"].values, dtype=np.float32)
        nl_ts  = pd.to_datetime(nl["time"].values)
        log.info(f"  NLDAS grid: ({nl_lat.size}x{nl_lon.size})")
        log.info(f"  NLDAS timesteps anno: {nl_ts.size}")

        # 3) Allinea tempi: trova l'intersezione
        common, nl_pos, sp_pos = np.intersect1d(
            nl_ts.values.astype("datetime64[ns]"),
            sp_ts[sp_idx_year].values.astype("datetime64[ns]"),
            return_indices=True,
        )
        if common.size == 0:
            raise ValueError(f"Nessuna ora comune NLDAS<->SPORTLIS per {year}")
        n_hours = common.size
        nl_time_sel = nl_pos
        sp_time_sel = sp_idx_year[sp_pos]
        log.info(f"  intersezione tempi: {n_hours}")

        # ordina per tempo (dovrebbe gia' esserlo, per sicurezza)
        order = np.argsort(common)
        common = common[order]
        nl_time_sel = nl_time_sel[order]
        sp_time_sel = sp_time_sel[order]

        # 4) Static (domain mask per swe_mask)
        static = load_sportlis_static(sp)
        domain2d = static["domain"].astype(bool)  # (H, W)

        # 5) Inizializza zarr di uscita
        #    Creiamo le variabili lazy con xarray + zarr backend.
        out_dir.mkdir(parents=True, exist_ok=True)
        t_coord = pd.DatetimeIndex(common)

        # inizializza store vuoto con shape nota
        empty = {
            v: (("time", "lat", "lon"), np.zeros((n_hours, H, W), dtype=np.float32))
            for v in FORCING_VARS
        }
        empty["swe_target"]        = (("time", "lat", "lon"), np.zeros((n_hours, H, W), dtype=np.float32))
        empty["swe_target_filled"] = (("time", "lat", "lon"), np.zeros((n_hours, H, W), dtype=np.float32))
        empty["swe_mask"]          = (("time", "lat", "lon"), np.zeros((n_hours, H, W), dtype=np.uint8))
        empty["is_snowy_time"]     = (("time",),              np.zeros((n_hours,), dtype=np.uint8))
        empty["snow_fraction"]     = (("time",),              np.zeros((n_hours,), dtype=np.float32))

        ds_out = xr.Dataset(
            data_vars=empty,
            coords={
                "time": t_coord,
                "latitude":  (("lat", "lon"), sp_lat2d),
                "longitude": (("lat", "lon"), sp_lon2d),
            },
        )
        enc = {v: {"chunks": (OUT_CHUNKS["time"], OUT_CHUNKS["lat"], OUT_CHUNKS["lon"])}
               for v in FORCING_VARS + ["swe_target", "swe_target_filled", "swe_mask"]}
        enc["is_snowy_time"] = {"chunks": (OUT_CHUNKS["time"],)}
        enc["snow_fraction"] = {"chunks": (OUT_CHUNKS["time"],)}
        log.info("Scrivo layout vuoto zarr...")
        ds_out.to_zarr(out_path, mode="w", consolidated=True, encoding=enc)
        # non serve tenere ds_out in memoria
        del ds_out

        # 6) Loop a batch temporali: interp forcing + carica SWE + scrivi
        #    Usiamo xarray interp che e' vectorized sui 2D target.
        lat_target = xr.DataArray(sp_lat2d, dims=("lat", "lon"))
        lon_target = xr.DataArray(sp_lon2d, dims=("lat", "lon"))

        # SWE raw: la leggiamo via netCDF4 per evitare xarray groups complexity
        swe_raw_all = sp.variables["data"]

        store = zarr.open(str(out_path), mode="r+")

        n_chunks = int(np.ceil(n_hours / TIME_CHUNK))
        for ci in range(n_chunks):
            a = ci * TIME_CHUNK
            b = min((ci + 1) * TIME_CHUNK, n_hours)
            n = b - a
            log.info(f"  [batch {ci+1}/{n_chunks}]  t[{a}:{b}]  ({n} h)")

            nl_idx_b = nl_time_sel[a:b]
            sp_idx_b = sp_time_sel[a:b]

            # --- forcing NLDAS -> SPORTLIS grid ---
            nl_batch = nl[FORCING_VARS].isel(time=nl_idx_b)
            interp = nl_batch.interp(
                lat=lat_target, lon=lon_target, method="linear",
                kwargs={"fill_value": None},
            )
            # interp ha dims (time, lat, lon); trasformo in dict di numpy
            for v in FORCING_VARS:
                arr = interp[v].values.astype(np.float32)  # (n, H, W)
                store[v][a:b] = arr

            # --- target SWE SPORTLIS ---
            swe_b = swe_raw_all[sp_idx_b, :, :, 0].astype(np.float32)  # (n, H, W)
            # In SPORTLIS non ci sono NaN esplicite nella nostra ispezione, ma
            # filtro eventuali valori negativi/fill residui per sicurezza.
            swe_valid = (swe_b > -0.5) & np.isfinite(swe_b)
            swe_b_filled = np.where(swe_valid, swe_b, 0.0).astype(np.float32)

            # mask per training: dominio static & SWE valida
            valid_mask = (swe_valid & domain2d[None, :, :]).astype(np.uint8)

            store["swe_target"][a:b]        = swe_b.astype(np.float32)
            store["swe_target_filled"][a:b] = swe_b_filled
            store["swe_mask"][a:b]          = valid_mask

            # --- is_snowy & fraction ---
            snowy_px = (swe_b_filled > SNOWY_SWE_THRESHOLD_MM) & (valid_mask > 0)
            snowy_count = snowy_px.sum(axis=(1, 2))
            is_snowy = (snowy_count >= SNOWY_MIN_PIXELS).astype(np.uint8)
            valid_count = valid_mask.sum(axis=(1, 2)).clip(min=1)
            snow_frac = (snowy_count / valid_count).astype(np.float32)

            store["is_snowy_time"][a:b] = is_snowy
            store["snow_fraction"][a:b] = snow_frac

        # riconsolida i metadati dopo le scritture a blocchi
        zarr.consolidate_metadata(str(out_path))
        nl.close()

        # riepilogo
        with xr.open_zarr(out_path, consolidated=True, chunks={}) as check:
            log.info(f"OK anno {year}: "
                     f"shape SWE {check['swe_target'].shape}  "
                     f"snowy timesteps = {int(check['is_snowy_time'].sum().values)}")
    finally:
        sp.close()


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", type=int, nargs="+", default=[2017, 2018, 2019, 2020, 2021],
                   help="Anni da processare (default 2017-2021)")
    p.add_argument("--nldas-dir", type=Path, default=NLDAS_DIR_DEFAULT)
    p.add_argument("--sportlis-file", type=Path, default=SPORTLIS_FILE_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--only-static", action="store_true",
                   help="Salva solo il file static_sportlis.nc e esci")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Prima scrivo il file static (serve al notebook, e' una tantum)
    sp = open_sportlis(args.sportlis_file)
    try:
        sp_lat2d = np.asarray(sp.variables["latitude"][:],  dtype=np.float32)
        sp_lon2d = np.asarray(sp.variables["longitude"][:], dtype=np.float32)
        static = load_sportlis_static(sp)
    finally:
        sp.close()

    static_out = args.out_dir / "sportlis_static.nc"
    save_sportlis_static(static, sp_lat2d, sp_lon2d, static_out)

    if args.only_static:
        log.info("--only-static: fatto.")
        return

    for y in args.years:
        try:
            build_one_year(y, args.nldas_dir, args.sportlis_file,
                           args.out_dir, overwrite=args.overwrite)
        except Exception as e:
            log.error(f"Errore anno {y}: {e}", exc_info=True)

    log.info("Fatto.")


if __name__ == "__main__":
    sys.exit(main())
