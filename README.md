# Bangkok Weather & AQI Dashboard

Daily weather + PM2.5 dashboard for Bangkok with an ML forecast
(HistGradientBoosting) and a radiative-cooling monthly model.

## Run
    pip install -r requirements.txt
    streamlit run app.py

## Structure
- src/fetch_data.py — pulls weather + AQI data
- src/ml_forecast.py — trains PM2.5 forecast model
- src/rc_monthly_model.py — radiative-cooling monthly estimates
- app.py — Streamlit dashboard

# Bangkok Weather & AQI Analytics Dashboard

Interactive Streamlit dashboard analyzing **7+ years of Bangkok weather** and
**~4 years of PM2.5 / AQI** data, with a research twist: a daily
**outdoor-test validity index** that tells a materials researcher which days
(and months) are suitable for field-testing radiative-cooling coatings.

![Python](https://img.shields.io/badge/python-3.11+-blue) ![Streamlit](https://img.shields.io/badge/streamlit-1.58-red) ![Plotly](https://img.shields.io/badge/plotly-interactive-purple)

## Why this exists

Radiative-cooling (RC) coating field tests need clear sky, strong solar
irradiation, low haze, and no rain. Bangkok has a **burning season (Dec–Apr)**
where PM2.5 routinely exceeds the Thai NAAQS, and a rainy season (Jun–Oct)
that invalidates most outdoor measurements. This dashboard turns public
climate data into an experiment-planning tool — and doubles as a general
Bangkok weather/air-quality explorer.

## Features

- **Climate tab** — temperature min/mean/max band, humidity vs the ~70 % RH
  radiative-cooling penalty threshold, monthly rainfall, seasonal boxplots
- **Air Quality tab** — PM2.5 time series against WHO (15 µg/m³) and Thai
  NAAQS (37.5 µg/m³) 24-h guidelines, year×month heatmap, seasonality
  boxplots, exceedance-day table per year
- **RC Test Planner tab** — daily 0–100 validity score
  (solar 40 % + haze 30 % + cloud 30 %, rain gate), % valid days per month,
  haze-vs-solar attenuation scatter, best-month recommendation
- **Data tab** — filtered table + CSV download

## Data sources

| Dataset | Source | Coverage |
|---------|--------|----------|
| Daily weather (temp, RH, rain, solar, cloud, wind) | [Open-Meteo Historical Archive](https://open-meteo.com/en/docs/historical-weather-api) | 2019-01-01 → today |
| Hourly PM2.5 / PM10 / O₃ / US AQI (CAMS) | [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) | 2022-08-01 → today |

Open-Meteo is free and keyless. The fetcher is provider-agnostic at the
DataFrame level — an OpenWeatherMap or IQAir backend can be dropped in by
implementing the same `fetch_weather()` / `fetch_air_quality()` contract.

## Run it

```bash
pip install -r requirements.txt
python src/fetch_data.py          # builds data/bangkok_daily.csv (cached)
streamlit run app.py
```

Refresh data anytime with `python src/fetch_data.py --force`.

## ML — next-day forecasting (src/ml_forecast.py)

Two supervised tasks, evaluated honestly (chronological 80/20 split, test =
Oct 2025 → Jul 2026, always benchmarked against persistence):

| Task | Best result | Baseline | Verdict |
|------|-------------|----------|---------|
| Next-day PM2.5 regression | MAE 5.59 µg/m³ (HistGradientBoosting, lags only) | **5.61 (persistence)** | Persistence still wins at h+1 — daily PM2.5 autocorrelation is brutal; adding next-day weather forecast features only hurts (MAE 6.32) |
| "Is tomorrow a VALID RC-test day?" classification | **AUC 0.99, F1 0.90** (HistGradientBoosting) | F1 0.61 (naive: tomorrow = today) | Big skill — this is the useful product (go/no-go for scheduling outdoor tests) |

Key lesson demonstrated: *always compare against a strong naive baseline* —
three model families and 21 features cannot beat `tomorrow = today` on MAE,
but reframing the question ("can I test tomorrow?") yields real, deployable
skill. Next-day weather features use realized values as a stand-in for a
weather forecast (perfect-prognosis); a production version would plug the
Open-Meteo forecast API.

```bash
python src/ml_forecast.py   # trains, evaluates, writes data/ml_metrics.json
```

## Physics — monthly RC performance model (src/rc_monthly_model.py)

A two-band radiative-cooling energy balance (8–13 µm atmospheric window vs
rest of spectrum) driven by the dashboard's monthly climatology. For six
materials (bare concrete → BaSO₄ ultra-white → ideal selective emitter) it
predicts, per month: cooling power at ambient (W/m²), equilibrium surface
temperature, and ΔT vs ambient — day (noon peak solar) and night. Window
sky emissivity is modeled from RH + cloud cover, anchored to ε = 0.84 at
RH 80 % (Bangkok baseline); h = 5.5 W/m²K; insulated surface; runtime < 1 s.

Headline predictions (Bangkok, 2019–2026 climatology):

| Result | Value |
|--------|-------|
| Night cooling, good emitters | 24–27 W/m² (Dec–Mar) → 12–13 W/m² (Jul–Sep) |
| Humidity penalty | Bangkok night cooling ≈ **34 %** of dry-climate (RH 30 %) performance |
| Selective vs broadband | Selective emitter **loses** in every month (humid sky closes the window) |
| Daytime subambient threshold | Only R_solar ≳ 0.95 (BaSO₄) stays below ambient at noon, and only Nov–Apr |
| Cenosphere-acrylic at noon | 4–6 °C above ambient — but **27–32 °C cooler than bare concrete** |

```bash
python src/rc_monthly_model.py   # writes data/rc_monthly_model.csv
```

## Validity-index definition

```
solar = clip((SW_radiation − 10) / (22 − 10), 0, 1)     # MJ/m²/day
haze  = clip((75 − PM2.5) / (75 − 15), 0, 1)            # µg/m³
sky   = clip((85 − cloud_cover) / (85 − 25), 0, 1)      # %
score = 100 × (0.40·solar + 0.30·haze + 0.30·sky)
if precipitation > 1 mm: score ×= 0.15                  # wet samples → invalid
```

`VALID ≥ 70 · MARGINAL 40–70 · INVALID < 40`

## Project structure

```
bangkok_weather_aqi/
├── app.py               # Streamlit dashboard (4 tabs, Plotly)
├── src/fetch_data.py    # Open-Meteo fetcher + derived metrics, atomic CSV write
├── data/bangkok_daily.csv
└── requirements.txt
```
