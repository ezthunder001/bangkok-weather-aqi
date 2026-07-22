"""Bangkok Weather & AQI Analytics Dashboard (Streamlit + Plotly).

Run:  streamlit run app.py
Data: data/bangkok_daily.csv  (refresh with: python src/fetch_data.py --force)
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_CSV = Path(__file__).parent / "data" / "bangkok_daily.csv"
ML_METRICS = Path(__file__).parent / "data" / "ml_metrics.json"
ML_PREDICTIONS = Path(__file__).parent / "data" / "ml_predictions.csv"
RC_MODEL_CSV = Path(__file__).parent / "data" / "rc_monthly_model.csv"

MATERIAL_COLORS = {
    "Bare concrete": "#888780",
    "White TiO2 paint": "#E0A82E",
    "Cenosphere-acrylic (EGAT-C)": "#D85A30",
    "PVDF-HFP porous": "#378ADD",
    "BaSO4 ultra-white": "#7F77DD",
    "Ideal selective emitter": "#1D9E75",
    "Ideal broadband emitter": "#D4537E",
}

WHO_PM25_24H = 15.0       # WHO 2021 guideline, µg/m³ (24-h)
THAI_PM25_24H = 37.5      # Thailand NAAQS (24-h), µg/m³

SEASON_COLORS = {"Cool": "#4C9BE8", "Hot": "#E8674C", "Rainy": "#4CB87A"}
CLASS_COLORS = {"VALID": "#2E9E5B", "MARGINAL": "#E0A82E", "INVALID": "#D14545"}

st.set_page_config(page_title="Bangkok Weather & AQI", page_icon="🌇", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV, parse_dates=["date"])
    df["month_name"] = df["date"].dt.strftime("%b")
    return df


df = load_data()

# ------------------------------------------------------------- sidebar
st.sidebar.title("Filters")
min_d, max_d = df["date"].min().date(), df["date"].max().date()
date_range = st.sidebar.date_input(
    "Date range", (pd.Timestamp("2022-08-01").date(), max_d),
    min_value=min_d, max_value=max_d,
)
seasons = st.sidebar.multiselect(
    "Season (Thai Met. Dept.)", ["Cool", "Hot", "Rainy"], default=["Cool", "Hot", "Rainy"]
)
st.sidebar.caption(
    "Data: Open-Meteo historical archive + CAMS air quality. "
    "Weather from 2019; PM2.5 from Aug 2022."
)

if len(date_range) == 2:
    lo, hi = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    lo, hi = pd.Timestamp(min_d), pd.Timestamp(max_d)
view = df[(df["date"] >= lo) & (df["date"] <= hi) & df["season"].isin(seasons)]

# ------------------------------------------------------------- header + KPIs
st.title("🌇 Bangkok Weather & AQI Analytics")
st.caption(
    "Daily climate + air-quality intelligence for Bangkok, with an outdoor-test "
    "planner for radiative-cooling coating field experiments."
)

latest = df.dropna(subset=["pm2_5"]).iloc[-1]
last30 = df[df["date"] > df["date"].max() - pd.Timedelta(days=30)]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Latest mean temp", f"{latest['temperature_2m_mean']:.1f} °C")
k2.metric("Latest humidity", f"{latest['relative_humidity_2m_mean']:.0f} %")
k3.metric(
    "Latest PM2.5", f"{latest['pm2_5']:.1f} µg/m³",
    delta=f"{latest['pm2_5'] - WHO_PM25_24H:+.1f} vs WHO", delta_color="inverse",
)
k4.metric("Latest US AQI", f"{latest['us_aqi']:.0f}")
k5.metric("Valid RC-test days (30d)", f"{(last30['test_class'] == 'VALID').sum()}")

tab_climate, tab_aq, tab_rc, tab_ml, tab_model, tab_data = st.tabs(
    ["🌡️ Climate", "🌫️ Air Quality", "🧪 RC Test Planner", "🤖 ML Forecast",
     "❄️ RC Model", "📋 Data"]
)

# ------------------------------------------------------------- climate tab
with tab_climate:
    c1, c2 = st.columns([2, 1])

    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=view["date"], y=view["temperature_2m_max"], name="Max",
            line=dict(width=0.5, color="rgba(232,103,76,0.4)"), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=view["date"], y=view["temperature_2m_min"], name="Min–Max band",
            fill="tonexty", fillcolor="rgba(232,103,76,0.15)",
            line=dict(width=0.5, color="rgba(76,155,232,0.4)"),
        ))
        fig.add_trace(go.Scatter(
            x=view["date"], y=view["temperature_2m_mean"].rolling(14).mean(),
            name="Mean (14-d roll)", line=dict(width=2, color="#E8674C"),
        ))
        fig.update_layout(title="Temperature (°C)", height=380, margin=dict(t=40, b=10))
        st.plotly_chart(fig, width='stretch')

    with c2:
        monthly_rain = (
            view.groupby(["year", "month"], as_index=False)["precipitation_sum"].sum()
        )
        monthly_rain["ym"] = pd.to_datetime(
            monthly_rain[["year", "month"]].assign(day=1)
        )
        fig = px.bar(
            monthly_rain, x="ym", y="precipitation_sum",
            title="Monthly rainfall (mm)", color_discrete_sequence=["#4CB87A"],
        )
        fig.update_layout(height=380, margin=dict(t=40, b=10), xaxis_title=None,
                          yaxis_title=None)
        st.plotly_chart(fig, width='stretch')

    c3, c4 = st.columns(2)
    with c3:
        fig = px.line(
            view, x="date", y="relative_humidity_2m_mean",
            title="Relative humidity (%) — RC cooling penalty above ~70 %",
            color_discrete_sequence=["#4C9BE8"],
        )
        fig.add_hline(y=70, line_dash="dot", line_color="#D14545",
                      annotation_text="RC humidity penalty threshold")
        fig.update_layout(height=340, margin=dict(t=40, b=10), xaxis_title=None,
                          yaxis_title=None)
        st.plotly_chart(fig, width='stretch')
    with c4:
        fig = px.box(
            view, x="month_name", y="temperature_2m_mean", color="season",
            category_orders={"month_name": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]},
            color_discrete_map=SEASON_COLORS, title="Mean temp by month",
        )
        fig.update_layout(height=340, margin=dict(t=40, b=10), xaxis_title=None,
                          yaxis_title=None)
        st.plotly_chart(fig, width='stretch')

# ------------------------------------------------------------- air-quality tab
with tab_aq:
    aq = view.dropna(subset=["pm2_5"])
    if aq.empty:
        st.info("No air-quality data in the selected range (PM2.5 starts Aug 2022).")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=aq["date"], y=aq["pm2_5"], name="PM2.5 daily mean",
            line=dict(width=1, color="rgba(120,120,140,0.55)"),
        ))
        fig.add_trace(go.Scatter(
            x=aq["date"], y=aq["pm2_5"].rolling(30, min_periods=10).mean(),
            name="30-d rolling", line=dict(width=2.5, color="#7A4CE8"),
        ))
        fig.add_hline(y=WHO_PM25_24H, line_dash="dot", line_color="#2E9E5B",
                      annotation_text="WHO 24-h guideline (15)")
        fig.add_hline(y=THAI_PM25_24H, line_dash="dot", line_color="#D14545",
                      annotation_text="Thai NAAQS (37.5)")
        fig.update_layout(title="PM2.5 (µg/m³)", height=400, margin=dict(t=40, b=10))
        st.plotly_chart(fig, width='stretch')

        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(
                aq, x="month_name", y="pm2_5", color="season",
                category_orders={"month_name": ["Jan", "Feb", "Mar", "Apr", "May",
                                                "Jun", "Jul", "Aug", "Sep", "Oct",
                                                "Nov", "Dec"]},
                color_discrete_map=SEASON_COLORS,
                title="PM2.5 seasonality — burning-season spike Dec–Apr",
            )
            fig.update_layout(height=360, margin=dict(t=40, b=10), xaxis_title=None,
                              yaxis_title=None)
            st.plotly_chart(fig, width='stretch')
        with c2:
            pivot = aq.pivot_table(index="year", columns="month", values="pm2_5",
                                   aggfunc="mean")
            fig = px.imshow(
                pivot, color_continuous_scale="YlOrRd", aspect="auto",
                title="Mean PM2.5 by year × month",
                labels=dict(x="Month", y="Year", color="µg/m³"),
            )
            fig.update_layout(height=360, margin=dict(t=40, b=10))
            st.plotly_chart(fig, width='stretch')

        exceed = aq.groupby("year").agg(
            days=("pm2_5", "size"),
            over_who=("pm2_5", lambda s: int((s > WHO_PM25_24H).sum())),
            over_thai=("pm2_5", lambda s: int((s > THAI_PM25_24H).sum())),
        )
        exceed["% over WHO"] = (100 * exceed["over_who"] / exceed["days"]).round(1)
        exceed["% over Thai NAAQS"] = (100 * exceed["over_thai"] / exceed["days"]).round(1)
        st.dataframe(exceed, width='stretch')

# ------------------------------------------------------------- RC planner tab
with tab_rc:
    st.markdown(
        "**Outdoor test validity** scores each day 0–100 for radiative-cooling "
        "field testing: solar irradiation (40 %), PM2.5 haze (30 %), cloud cover "
        "(30 %); rain > 1 mm collapses the score. ≥70 = **VALID**, 40–70 = "
        "**MARGINAL**, <40 = **INVALID**."
    )
    rc = view.dropna(subset=["test_validity"])
    if rc.empty:
        st.info("Validity needs PM2.5 data — select a range from Aug 2022 onward.")
    else:
        fig = px.scatter(
            rc, x="date", y="test_validity", color="test_class",
            color_discrete_map=CLASS_COLORS, title="Daily test-validity score",
            hover_data=["pm2_5", "shortwave_radiation_sum", "cloud_cover_mean",
                        "precipitation_sum"],
        )
        fig.add_hline(y=70, line_dash="dot", line_color="#2E9E5B")
        fig.add_hline(y=40, line_dash="dot", line_color="#E0A82E")
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        fig.update_layout(height=380, margin=dict(t=40, b=10))
        st.plotly_chart(fig, width='stretch')

        c1, c2 = st.columns(2)
        with c1:
            monthly = (
                rc.groupby("month")["test_class"]
                .apply(lambda s: 100 * (s == "VALID").mean())
                .reset_index(name="pct_valid")
            )
            monthly["month_name"] = pd.to_datetime(
                monthly["month"], format="%m"
            ).dt.strftime("%b")
            fig = px.bar(
                monthly, x="month_name", y="pct_valid",
                title="% VALID test days by month — plan campaigns here",
                color="pct_valid", color_continuous_scale="Greens",
            )
            fig.update_layout(height=360, margin=dict(t=40, b=10), xaxis_title=None,
                              yaxis_title="% of days VALID", coloraxis_showscale=False)
            st.plotly_chart(fig, width='stretch')
        with c2:
            fig = px.scatter(
                rc, x="pm2_5", y="shortwave_radiation_sum", color="test_class",
                color_discrete_map=CLASS_COLORS,
                title="Haze vs solar — PM2.5 attenuates irradiance",
                labels={"pm2_5": "PM2.5 (µg/m³)",
                        "shortwave_radiation_sum": "Solar (MJ/m²/day)"},
            )
            fig.update_traces(marker=dict(size=5, opacity=0.6))
            fig.update_layout(height=360, margin=dict(t=40, b=10))
            st.plotly_chart(fig, width='stretch')

        best = monthly.sort_values("pct_valid", ascending=False).head(3)
        st.success(
            "Best months for outdoor RC campaigns: "
            + ", ".join(f"**{r.month_name}** ({r.pct_valid:.0f} % valid)"
                        for r in best.itertuples())
        )

# ------------------------------------------------------------- ML tab
with tab_ml:
    if not ML_METRICS.exists():
        st.info("Run `python src/ml_forecast.py` to train the models first.")
    else:
        m = json.loads(ML_METRICS.read_text(encoding="utf-8"))
        pred = pd.read_csv(ML_PREDICTIONS, parse_dates=["date"])

        st.markdown(
            f"**Two models, honest evaluation** — chronological split, "
            f"test = {m['test_period']} ({m['test_days']} unseen days), every model "
            "benchmarked against the *persistence* baseline (tomorrow = today)."
        )

        best = m["regression"][m["best_model"]]
        pers = m["regression"]["persistence"]
        clf = m["classification"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Persistence MAE", f"{pers['mae']:.2f} µg/m³")
        c2.metric(f"Best model MAE ({m['best_model'].split('__')[0]})",
                  f"{best['mae']:.2f} µg/m³",
                  delta=f"{m['skill_vs_persistence_pct']:+.1f} % skill")
        c3.metric("VALID-day classifier AUC", f"{clf['model']['roc_auc']:.2f}")
        c4.metric("Classifier F1 (vs naive)",
                  f"{clf['model']['f1']:.2f}",
                  delta=f"naive {clf['persistence']['f1']:.2f}")

        st.markdown(
            "**Finding 1 — persistence is brutal at h+1.** Daily-mean PM2.5 is so "
            "autocorrelated that no model beats `tomorrow = today` on MAE. "
            "**Finding 2 — the planner question is learnable.** Predicting whether "
            "tomorrow is a VALID outdoor-test day (with forecast weather features) "
            f"reaches AUC {clf['model']['roc_auc']:.2f} / F1 {clf['model']['f1']:.2f} "
            f"vs naive F1 {clf['persistence']['f1']:.2f} — that is the useful product."
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pred["date"], y=pred["actual"], name="Actual",
                                 line=dict(width=2, color="#444441")))
        fig.add_trace(go.Scatter(x=pred["date"], y=pred["best_model"], name="Model",
                                 line=dict(width=1.5, color="#7F77DD")))
        fig.add_trace(go.Scatter(x=pred["date"], y=pred["persistence"],
                                 name="Persistence", line=dict(width=1, dash="dot",
                                                               color="#B4B2A9")))
        fig.update_layout(title="Test set — next-day PM2.5 (µg/m³)", height=380,
                          margin=dict(t=40, b=10))
        st.plotly_chart(fig, width='stretch')

        c1, c2 = st.columns(2)
        with c1:
            imp = pd.DataFrame(m["feature_importance"])
            fig = px.bar(imp.sort_values("importance"), x="importance", y="feature",
                         orientation="h", title="Permutation importance (MAE, test set)",
                         color_discrete_sequence=["#7F77DD"])
            fig.update_layout(height=380, margin=dict(t=40, b=10), yaxis_title=None)
            st.plotly_chart(fig, width='stretch')
        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pred["date"], y=pred["valid_proba"], name="P(VALID tomorrow)",
                line=dict(width=1.5, color="#1D9E75"),
            ))
            valid_days = pred[pred["valid_actual"] == 1]
            fig.add_trace(go.Scatter(
                x=valid_days["date"], y=[1.02] * len(valid_days), mode="markers",
                name="Actual VALID day", marker=dict(symbol="line-ns-open", size=8,
                                                     color="#2E9E5B"),
            ))
            fig.add_hline(y=0.5, line_dash="dot", line_color="#E0A82E")
            fig.update_layout(title="Go/no-go probability vs reality", height=380,
                              margin=dict(t=40, b=10), yaxis_range=[0, 1.08])
            st.plotly_chart(fig, width='stretch')

        t = m["tomorrow"]
        st.success(
            f"Next-day inference (from {t['from_date']}): PM2.5 ≈ "
            f"**{t['pm25_pred']} µg/m³**, P(valid RC-test day) = "
            f"**{t['valid_probability']:.0%}**"
        )
        st.caption(
            "Caveat: next-day weather features use realized values as a stand-in "
            "for a weather forecast (perfect-prognosis). With a real forecast feed "
            "(Open-Meteo forecast API), expect slightly lower classifier scores."
        )

# ------------------------------------------------------------- RC model tab
with tab_model:
    if not RC_MODEL_CSV.exists():
        st.info("Run `python src/rc_monthly_model.py` to generate the model first.")
    else:
        rcm = pd.read_csv(RC_MODEL_CSV)
        st.markdown(
            "**Month-by-month radiative-cooling performance**, predicted by a "
            "two-band energy balance (8–13 µm window vs rest of spectrum) driven "
            "by this dashboard's monthly climatology (T, RH, cloud, solar). "
            "Window sky emissivity is anchored to 0.84 at RH 80 % "
            "(project baseline); h = 5.5 W/m²K; insulated surface."
        )

        mats = st.multiselect(
            "Materials", list(MATERIAL_COLORS),
            default=[m for m in MATERIAL_COLORS if m != "Bare concrete"],
        )
        sel = rcm[rcm["material"].isin(mats)]

        c1, c2 = st.columns(2)
        with c1:
            night = sel[sel["scenario"] == "night"]
            fig = px.line(
                night, x="month", y="P_cool_at_ambient_Wm2", color="material",
                color_discrete_map=MATERIAL_COLORS, markers=True,
                title="Night cooling power at ambient (W/m²)",
            )
            fig.update_layout(height=380, margin=dict(t=40, b=10),
                              xaxis=dict(dtick=1), legend_title=None)
            st.plotly_chart(fig, width='stretch')
        with c2:
            day = sel[sel["scenario"] == "day"]
            fig = px.line(
                day, x="month", y="dT_eq_C", color="material",
                color_discrete_map=MATERIAL_COLORS, markers=True,
                title="Noon equilibrium ΔT (°C, + = below ambient)",
            )
            fig.add_hline(y=0, line_dash="dot", line_color="#888780")
            fig.update_layout(height=380, margin=dict(t=40, b=10),
                              xaxis=dict(dtick=1), legend_title=None)
            st.plotly_chart(fig, width='stretch')

        c3, c4 = st.columns(2)
        with c3:
            fig = px.line(
                day, x="month", y="P_cool_at_ambient_Wm2", color="material",
                color_discrete_map=MATERIAL_COLORS, markers=True,
                title="Noon cooling power at ambient (W/m², + = net cooling)",
            )
            fig.add_hline(y=0, line_dash="dot", line_color="#888780")
            fig.update_layout(height=380, margin=dict(t=40, b=10),
                              xaxis=dict(dtick=1), legend_title=None)
            st.plotly_chart(fig, width='stretch')
        with c4:
            bm = st.selectbox("Budget month", range(1, 13), index=0,
                              format_func=lambda m: f"{m:02d}")
            budget = (rcm[(rcm["scenario"] == "day") & (rcm["month"] == bm)]
                      .sort_values("P_solar_abs_Wm2"))
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=budget["material"], x=budget["P_rad_net_Wm2"],
                orientation="h", name="Thermal radiation out",
                marker_color="#5DCAA5",
            ))
            fig.add_trace(go.Bar(
                y=budget["material"], x=-budget["P_solar_abs_Wm2"],
                orientation="h", name="Solar absorbed (heat in)",
                marker_color="#F0997B",
            ))
            fig.update_layout(
                barmode="relative", height=350,
                title=f"Noon energy budget — month {bm:02d} (W/m²)",
                margin=dict(t=40, b=10),
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig, width='stretch')
            st.caption(
                "Every coating radiates a near-identical ~22–31 W/m² — the humid "
                "sky caps the radiative term. Daytime ranking is decided almost "
                "entirely by solar reflectance."
            )

        ceno = rcm.query(
            "material == 'Cenosphere-acrylic (EGAT-C)' and scenario == 'night'"
        )
        hc = ceno["humidity_correction"].astype(float).mean()
        st.warning(
            f"**Bangkok humidity penalty (mandatory check):** night cooling for "
            f"the cenosphere coating is {ceno['P_cool_at_ambient_Wm2'].min():.0f}–"
            f"{ceno['P_cool_at_ambient_Wm2'].max():.0f} W/m² across the year — "
            f"only **{hc:.0%} of dry-climate performance** (RH 30 %, clear). "
            "The humid-sky window closure also erases the selective emitter's "
            "advantage: broadband emitters beat it in every month."
        )

        with st.expander("Full model table"):
            st.dataframe(rcm, width='stretch', height=400)
            st.download_button(
                "Download model CSV", rcm.to_csv(index=False).encode(),
                file_name="rc_monthly_model.csv", mime="text/csv",
            )

# ------------------------------------------------------------- data tab
with tab_data:
    st.dataframe(view, width='stretch', height=480)
    st.download_button(
        "Download filtered CSV", view.to_csv(index=False).encode(),
        file_name="bangkok_weather_aqi.csv", mime="text/csv",
    )
