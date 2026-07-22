"""Month-by-month radiative-cooling performance model for Bangkok.

Couples the daily climatology in data/bangkok_daily.csv (monthly means of
T_max/T_min, RH, cloud cover, solar irradiation) with a two-band
radiative-cooling energy balance:

    equilibrium:  P_rad(T_s) = P_atm + P_solar_abs + h·(T_amb - T_s)
    cooling power at ambient:  P_cool = P_rad(T_amb) - P_atm - P_solar_abs

Bands: the 8-13 µm atmospheric window (sky emissivity driven by humidity +
cloud) and everything outside it (sky ~opaque, eps_sky ≈ 0.97). This lets
broadband and spectrally-selective emitters be compared fairly.

Calibration anchors (consistent with agents/edward/sim.py and config.yaml):
  - window sky emissivity 0.84 at RH 80 % (Bangkok baseline)
  - dry-climate reference: RH 30 %, cloud 10 % (humidity correction factor)
  - h_conv = 5.5 W/m²K

Assumptions: insulated surface (no conduction into roof mass), monthly-mean
weather, half-sine daytime solar profile (noon peak used for day scenario).

Outputs: data/rc_monthly_model.csv + data/rc_monthly_summary.json
Runtime: < 2 s (Planck band fraction is interpolated from a lookup table).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DAILY_CSV = ROOT / "data" / "bangkok_daily.csv"
OUT_CSV = ROOT / "data" / "rc_monthly_model.csv"
OUT_JSON = ROOT / "data" / "rc_monthly_summary.json"

SIGMA = 5.670374419e-8
H_CONV = 5.5                 # W/m²K (config.yaml bangkok_atmosphere)
EPS_SKY_OUTSIDE = 0.97       # atmosphere ~opaque outside the window
DAYLIGHT_HOURS = 12.0

# Solar reflectance + emissivity inside/outside the 8-13 µm window.
# Nominal literature values, mid-tier selective coating.
MATERIALS = {
    "Bare concrete": {"R_solar": 0.35, "eps_win": 0.90, "eps_out": 0.90},
    "White TiO2 paint": {"R_solar": 0.80, "eps_win": 0.90, "eps_out": 0.90},
    "Coating C": {"R_solar": 0.88, "eps_win": 0.92, "eps_out": 0.90},
    "PVDF-HFP porous": {"R_solar": 0.96, "eps_win": 0.97, "eps_out": 0.95},
    "BaSO4 ultra-white": {"R_solar": 0.976, "eps_win": 0.96, "eps_out": 0.95},
    # Matched pair — identical R_solar and window emissivity, differing only
    # outside the window. Isolates whether spectral selectivity pays here.
    "Ideal selective emitter": {"R_solar": 0.97, "eps_win": 0.95, "eps_out": 0.05},
    "Ideal broadband emitter": {"R_solar": 0.97, "eps_win": 0.95, "eps_out": 0.95},
}


def _build_window_fraction_table(lo_um=8.0, hi_um=13.0,
                                 t_lo=150.0, t_hi=500.0, n_t=400):
    """Lookup table: fraction of blackbody power in [lo, hi] µm vs T.

    Built once; equilibrium solving then uses np.interp instead of
    re-integrating Planck's law on every iteration.
    """
    h, c, kb = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    lam = np.linspace(lo_um, hi_um, 600)[None, :] * 1e-6
    T = np.linspace(t_lo, t_hi, n_t)[:, None]
    spectral = (2 * np.pi * h * c**2 / lam**5) / (np.expm1(h * c / (lam * kb * T)))
    band = np.trapezoid(spectral, lam[0], axis=1)
    return T[:, 0], band / (SIGMA * T[:, 0] ** 4)


_T_GRID, _F_GRID = _build_window_fraction_table()


def planck_window_fraction(T_K: float) -> float:
    return float(np.interp(T_K, _T_GRID, _F_GRID))


def sky_window_emissivity(rh_pct: float, cloud_frac: float) -> float:
    """8-13 µm sky emissivity from humidity + cloud.

    Linear humidity model anchored to the project baseline (0.84 at RH 80 %,
    literature clear-sky range ~0.6 dry to ~0.95 tropical-humid); clouds fill
    the remaining window transparency toward opaque.
    """
    eps_clear = float(np.clip(0.50 + 0.00425 * rh_pct, 0.55, 0.95))
    return eps_clear + (0.98 - eps_clear) * float(np.clip(cloud_frac, 0, 1))


def p_rad(T_s: float, mat: dict) -> float:
    f = planck_window_fraction(T_s)
    return (mat["eps_win"] * f + mat["eps_out"] * (1 - f)) * SIGMA * T_s**4


def p_atm(T_amb: float, mat: dict, eps_sky_win: float) -> float:
    f = planck_window_fraction(T_amb)
    return (mat["eps_win"] * f * eps_sky_win
            + mat["eps_out"] * (1 - f) * EPS_SKY_OUTSIDE) * SIGMA * T_amb**4


def cooling_power(T_s: float, T_amb: float, mat: dict,
                  eps_sky_win: float, G: float) -> float:
    return p_rad(T_s, mat) - p_atm(T_amb, mat, eps_sky_win) - (1 - mat["R_solar"]) * G


def equilibrium_dT(T_amb: float, mat: dict, eps_sky_win: float,
                   G: float) -> tuple[float, float]:
    """Returns (dT, Ts_C). dT = T_amb - Ts_eq: + subambient, - heating."""
    def net(T_s):
        # convection is parasitic gain when T_s < T_amb (EDWARD's P_nonrad)
        return cooling_power(T_s, T_amb, mat, eps_sky_win, G) - H_CONV * (T_amb - T_s)

    ts = brentq(net, T_amb - 80, T_amb + 120, xtol=1e-3)
    return T_amb - ts, ts - 273.15


def monthly_climatology() -> pd.DataFrame:
    df = pd.read_csv(DAILY_CSV, parse_dates=["date"])
    return df.groupby("month").agg(
        t_max=("temperature_2m_max", "mean"),
        t_min=("temperature_2m_min", "mean"),
        rh=("relative_humidity_2m_mean", "mean"),
        cloud=("cloud_cover_mean", "mean"),
        solar_mj=("shortwave_radiation_sum", "mean"),
    ).round(2)


def main() -> None:
    t0 = time.perf_counter()
    clim = monthly_climatology()
    eps_sky_dry = sky_window_emissivity(30.0, 0.10)   # dry-climate reference

    rows = []
    for month, c in clim.iterrows():
        eps_sky = sky_window_emissivity(c.rh, c.cloud / 100)
        g_noon = c.solar_mj * 1e6 / (DAYLIGHT_HOURS * 3600) * np.pi / 2

        for name, mat in MATERIALS.items():
            for scen, t_amb_c, g in (("day", c.t_max, g_noon), ("night", c.t_min, 0.0)):
                t_amb = t_amb_c + 273.15
                p_amb = cooling_power(t_amb, t_amb, mat, eps_sky, g)
                dt, ts_c = equilibrium_dT(t_amb, mat, eps_sky, g)
                p_dry = cooling_power(t_amb, t_amb, mat, eps_sky_dry, g)
                absorbed = (1 - mat["R_solar"]) * g
                rows.append({
                    "month": month, "material": name, "scenario": scen,
                    "T_amb_C": round(t_amb_c, 1), "G_Wm2": round(g, 0),
                    "eps_sky_window": round(eps_sky, 3),
                    "P_solar_abs_Wm2": round(absorbed, 1),
                    "P_rad_net_Wm2": round(p_amb + absorbed, 1),
                    "P_cool_at_ambient_Wm2": round(p_amb, 1),
                    "dT_eq_C": round(dt, 2),
                    "T_surface_eq_C": round(ts_c, 1),
                    "P_cool_dry_climate_Wm2": round(p_dry, 1),
                    "humidity_correction": round(p_amb / p_dry, 3) if p_dry > 0 else None,
                })

    out = pd.DataFrame(rows)
    tmp = OUT_CSV.with_suffix(".csv.tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(OUT_CSV)

    coating_c = "Coating C"
    night = out.query("material == @coating_c and scenario == 'night'")
    day = out.query("material == @coating_c and scenario == 'day'")
    summary = {
        "climatology": clim.reset_index().to_dict("records"),
        "coating_c_night_P_cool_range_Wm2":
            [float(night["P_cool_at_ambient_Wm2"].min()),
             float(night["P_cool_at_ambient_Wm2"].max())],
        "coating_c_mean_humidity_correction_night":
            round(float(night["humidity_correction"].astype(float).mean()), 3),
        "best_night_month": int(night.loc[night["P_cool_at_ambient_Wm2"].idxmax(), "month"]),
        "worst_night_month": int(night.loc[night["P_cool_at_ambient_Wm2"].idxmin(), "month"]),
        "coating_c_day_dT_range_C":
            [float(day["dT_eq_C"].min()), float(day["dT_eq_C"].max())],
    }
    tmp = OUT_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    tmp.replace(OUT_JSON)

    print(f"Saved {len(out)} rows → {OUT_CSV.name}  "
          f"({time.perf_counter() - t0:.2f} s)")
    print("\nNight P_cool at ambient (W/m²):")
    print(out[out.scenario == "night"].pivot(
        index="month", columns="material", values="P_cool_at_ambient_Wm2").to_string())
    print("\nDay equilibrium dT (°C, + = below ambient):")
    print(out[out.scenario == "day"].pivot(
        index="month", columns="material", values="dT_eq_C").to_string())
    print(f"\nMean humidity correction (night, Coating C): "
          f"{summary['coating_c_mean_humidity_correction_night']}")


if __name__ == "__main__":
    main()
