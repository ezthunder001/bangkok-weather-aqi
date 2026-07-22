# -*- coding: utf-8 -*-
"""All-in-one self-contained HTML dashboard for P4 — Bangkok Weather & AQI.

Mirrors every Streamlit tab in a single offline file (v2 dashboard style:
embedded JSON, hand-built SVG charts, chip filters, dark/light theme,
zero external assets):

  climate · air quality · RC test planner · ML forecast · RC monthly model

Output: wiki/_reports/shared/bangkok-weather-aqi-dashboard.html
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent.parent
REPO = SRC.parent.parent
OUT_DIR = REPO / "wiki" / "_reports" / "shared"
OUT = OUT_DIR / "bangkok-weather-aqi-dashboard.html"

WHO, THAI = 15.0, 37.5
RC_COLORS = {
    "Bare concrete": "#888780",
    "White TiO2 paint": "#E0A82E",
    "Composite selective coating": "#D85A30",
    "PVDF-HFP porous": "#378ADD",
    "BaSO4 ultra-white": "#7F77DD",
    "Ideal selective emitter": "#1D9E75",
    "Ideal broadband emitter": "#D4537E",
}
CLS = {"INVALID": 0, "MARGINAL": 1, "VALID": 2, "NO_DATA": -1}


def _r(v, d=1):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), d)


def main():
    df = pd.read_csv(SRC / "data" / "bangkok_daily.csv", parse_dates=["date"]).sort_values("date")
    df["pm_roll"] = df["pm2_5"].rolling(30, min_periods=10).mean()

    daily = {
        "d": [x.strftime("%Y-%m-%d") for x in df["date"]],
        "tx": [_r(v) for v in df["temperature_2m_max"]],
        "tn": [_r(v) for v in df["temperature_2m_min"]],
        "tm": [_r(v) for v in df["temperature_2m_mean"]],
        "rh": [_r(v, 0) for v in df["relative_humidity_2m_mean"]],
        "rn": [_r(v) for v in df["precipitation_sum"]],
        "pm": [_r(v) for v in df["pm2_5"]],
        "pr": [_r(v) for v in df["pm_roll"]],
        "aq": [_r(v, 0) for v in df["us_aqi"]],
        "tv": [_r(v, 0) for v in df["test_validity"]],
        "tc": [CLS.get(c, -1) for c in df["test_class"]],
    }

    months = range(1, 13)
    g = df.groupby("month")
    aqd = df.dropna(subset=["pm2_5"]).groupby("month")
    rcd = df.dropna(subset=["test_validity"]).groupby("month")
    rain_m = df.groupby(["year", "month"])["precipitation_sum"].sum().groupby("month").mean()
    monthly = {
        "pm": [_r(aqd["pm2_5"].mean().get(m)) for m in months],
        "valid": [_r(100 * (rcd["test_class"].apply(lambda s: (s == "VALID").mean()).get(m, 0)), 1)
                  for m in months],
        "rh": [_r(g["relative_humidity_2m_mean"].mean().get(m), 0) for m in months],
        "tm": [_r(g["temperature_2m_mean"].mean().get(m)) for m in months],
        "rain": [_r(rain_m.get(m), 0) for m in months],
    }

    aq = df.dropna(subset=["pm2_5"])
    heat_years = sorted(aq["year"].unique().tolist())
    piv = aq.pivot_table(index="year", columns="month", values="pm2_5", aggfunc="mean")
    heat = {"years": heat_years,
            "grid": [[_r(piv.loc[y].get(m)) for m in months] for y in heat_years]}

    exceed = []
    for y, s in aq.groupby("year")["pm2_5"]:
        exceed.append({"y": int(y), "n": int(s.size),
                       "who": int((s > WHO).sum()), "thai": int((s > THAI).sum()),
                       "pwho": _r(100 * (s > WHO).mean(), 0),
                       "pthai": _r(100 * (s > THAI).mean(), 0)})

    mlm = json.loads((SRC / "data" / "ml_metrics.json").read_text(encoding="utf-8"))
    pred = pd.read_csv(SRC / "data" / "ml_predictions.csv", parse_dates=["date"])
    ml = {
        "d": [x.strftime("%Y-%m-%d") for x in pred["date"]],
        "act": [_r(v) for v in pred["actual"]],
        "mod": [_r(v) for v in pred["best_model"]],
        "proba": [_r(v, 2) for v in pred["valid_proba"]],
        "vact": [int(v) for v in pred["valid_actual"]],
        "metrics": {
            "persistence_mae": mlm["regression"]["persistence"]["mae"],
            "best_mae": mlm["regression"][mlm["best_model"]]["mae"],
            "best_name": mlm["best_model"].split("__")[0].replace("_", " "),
            "auc": mlm["classification"]["model"]["roc_auc"],
            "f1": mlm["classification"]["model"]["f1"],
            "naive_f1": mlm["classification"]["persistence"]["f1"],
            "period": mlm["test_period"],
        },
        "imp": mlm["feature_importance"][:8],
    }

    rcm = pd.read_csv(SRC / "data" / "rc_monthly_model.csv")
    rsum = json.loads((SRC / "data" / "rc_monthly_summary.json").read_text(encoding="utf-8"))
    mats = [m for m in RC_COLORS if m in set(rcm["material"])]

    def series(scen, col):
        return {m: [float(v) for v in
                    rcm[(rcm.material == m) & (rcm.scenario == scen)]
                    .sort_values("month")[col]] for m in mats}

    budget = {str(mo): {r["material"]: {"abs": float(r["P_solar_abs_Wm2"]),
                                        "rad": float(r["P_rad_net_Wm2"])}
                        for _, r in rcm[(rcm.scenario == "day") & (rcm.month == mo)].iterrows()}
              for mo in months}
    rc = {
        "mats": mats, "colors": RC_COLORS,
        "metrics": {
            "night_p": {"label": "Night cooling power (W/m²)", "zero": False,
                        "data": series("night", "P_cool_at_ambient_Wm2")},
            "day_p": {"label": "Noon cooling power (W/m², + = net cooling)", "zero": True,
                      "data": series("day", "P_cool_at_ambient_Wm2")},
            "day_dt": {"label": "Noon equilibrium ΔT (°C, + = below ambient)", "zero": True,
                       "data": series("day", "dT_eq_C")},
        },
        "budget": budget,
        "hc": rsum["composite_mean_humidity_correction_night"],
        "nightRange": rsum["composite_night_P_cool_range_Wm2"],
    }

    last = df.dropna(subset=["pm2_5"]).iloc[-1]
    last30 = df[df["date"] > df["date"].max() - pd.Timedelta(days=30)]
    kpi = {
        "date": last["date"].strftime("%Y-%m-%d"),
        "tm": _r(last["temperature_2m_mean"]), "rh": _r(last["relative_humidity_2m_mean"], 0),
        "pm": _r(last["pm2_5"]), "aq": _r(last["us_aqi"], 0),
        "valid30": int((last30["test_class"] == "VALID").sum()),
        "span": f"{df['date'].min():%Y-%m-%d} → {df['date'].max():%Y-%m-%d}",
        "nDays": len(df), "nAq": int(df["pm2_5"].notna().sum()),
    }

    data = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "daily": daily,
            "monthly": monthly, "heat": heat, "exceed": exceed, "ml": ml,
            "rc": rc, "kpi": kpi, "who": WHO, "thai": THAI}
    data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".html.tmp")
    tmp.write_text(HTML_TEMPLATE.replace("__DATA__", data_json), encoding="utf-8")
    tmp.replace(OUT)
    print(f"Saved → {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bangkok Weather &amp; AQI — Analytics Dashboard</title>
<style>
:root{
  --brand:#0D9373; --warn:#D97706; --bad:#EF4444; --info:#3B82F6;
  --bg:#FAFAF9; --card:#FFFFFF; --border:#E7E5E4;
  --text:#1C1917; --text2:#78716C; --chip:#F5F5F4;
  --shadow:0 1px 2px rgba(0,0,0,.05),0 4px 12px rgba(0,0,0,.04);
}
[data-theme="dark"]{
  --bg:#0C0E12; --card:#15181E; --border:#262B33;
  --text:#E7E5E4; --text2:#8E8E93; --chip:#1E222A;
  --shadow:0 1px 2px rgba(0,0,0,.4);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--text);
  font:15px/1.6 Inter,-apple-system,"Segoe UI",system-ui,sans-serif;
  font-variant-numeric:tabular-nums}
.wrap{max-width:1100px;margin:0 auto;padding:0 20px 70px}
.topbar{position:sticky;top:0;z-index:40;display:flex;gap:14px;align-items:center;
  background:color-mix(in srgb,var(--bg) 90%,transparent);backdrop-filter:blur(8px);
  border-bottom:1px solid var(--border);padding:9px 0;font-size:13.5px}
.topbar a{color:var(--text2);text-decoration:none;padding:3px 8px;border-radius:6px}
.topbar a:hover{color:var(--text);background:var(--chip)}
.topbar button{margin-left:auto;background:var(--chip);border:1px solid var(--border);
  color:var(--text);border-radius:8px;padding:5px 11px;cursor:pointer;font-size:13px}
header.hero{padding:30px 0 8px}
.hero h1{font-size:27px;font-weight:800;letter-spacing:-.02em;margin:0}
.hero h1 .accent{color:var(--brand)}
.hero p{color:var(--text2);margin:5px 0 0;font-size:14px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
  gap:12px;margin:16px 0 8px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;
  padding:13px 15px;box-shadow:var(--shadow)}
.card .lbl{font-size:12.5px;color:var(--text2)}
.card .val{font-size:22px;font-weight:700;margin-top:2px}
.card .sub{font-size:12px;color:var(--text2)}
h2{font-size:18px;font-weight:700;margin:38px 0 8px;scroll-margin-top:54px}
h3{font-size:14.5px;font-weight:600;margin:18px 0 6px;color:var(--text2)}
.panel{background:var(--card);border:1px solid var(--border);border-radius:12px;
  padding:14px;box-shadow:var(--shadow)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:860px){.grid2{grid-template-columns:1fr}}
.seg{display:inline-flex;background:var(--chip);border:1px solid var(--border);
  border-radius:10px;overflow:hidden;margin:6px 8px 10px 0;vertical-align:middle}
.seg button{background:none;border:none;color:var(--text2);padding:6px 13px;
  font-size:13px;cursor:pointer}
.seg button.on{background:var(--card);color:var(--text);font-weight:600;box-shadow:var(--shadow)}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}
.chip{display:flex;align-items:center;gap:6px;background:var(--chip);
  border:1px solid var(--border);border-radius:999px;padding:4px 11px;
  font-size:12.5px;cursor:pointer;user-select:none;opacity:.45}
.chip.on{opacity:1;border-color:currentColor}
.chip .dot{width:10px;height:10px;border-radius:3px}
.mchip{background:var(--chip);border:1px solid var(--border);border-radius:8px;
  padding:3px 9px;font-size:12.5px;cursor:pointer;display:inline-block;margin:0 4px 6px 0}
.mchip.on{background:var(--brand);color:#fff;border-color:var(--brand)}
svg text{fill:var(--text2);font-size:11px}
svg .axis{stroke:var(--border)}
.note{font-size:12.5px;color:var(--text2);margin-top:6px}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;color:var(--text2);margin:4px 0 8px}
.legend span{display:flex;align-items:center;gap:5px}
.sw{width:11px;height:11px;border-radius:3px;display:inline-block}
.ln{width:18px;height:0;border-top:2.5px solid;display:inline-block}
.ln.dash{border-top-style:dashed}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border-bottom:1px solid var(--border);padding:6px 8px;text-align:right}
th:first-child,td:first-child{text-align:left}
th{color:var(--text2);font-weight:600}
.warnBox{background:color-mix(in srgb,var(--warn) 10%,var(--card));
  border:1px solid color-mix(in srgb,var(--warn) 35%,var(--border));
  border-radius:12px;padding:11px 15px;font-size:13.5px;margin-top:14px}
.okBox{background:color-mix(in srgb,var(--brand) 10%,var(--card));
  border:1px solid color-mix(in srgb,var(--brand) 35%,var(--border));
  border-radius:12px;padding:11px 15px;font-size:13.5px;margin-top:14px}
details{margin-top:12px}
summary{cursor:pointer;color:var(--brand);font-size:13.5px}
footer{color:var(--text2);font-size:12.5px;margin-top:46px;border-top:1px solid var(--border);padding-top:14px}
</style>
</head>
<body>
<div class="wrap">
<nav class="topbar">
  <a href="#climate">Climate</a><a href="#aq">Air quality</a>
  <a href="#planner">Test planner</a><a href="#ml">ML forecast</a>
  <a href="#rcmodel">RC model</a>
  <button id="themeBtn">◐ theme</button>
</nav>
<header class="hero">
  <h1>Bangkok <span class="accent">weather &amp; AQI</span> analytics</h1>
  <p>Open-Meteo + CAMS · <span id="span"></span> · <span id="nd"></span> days weather,
     <span id="na"></span> days air quality · generated <span id="gen"></span></p>
</header>

<div class="cards" id="kpis"></div>

<h2 id="climate">🌡 Climate</h2>
<span class="note">Window:</span>
<div class="seg" id="winSeg"></div>
<div class="panel">
  <div class="legend"><span><span class="ln" style="border-color:#D85A30"></span>Mean temp (14-d roll)</span>
  <span><span class="sw" style="background:#D85A30;opacity:.25"></span>Min–max band</span></div>
  <svg id="tempChart" width="100%" viewBox="0 0 1040 300"></svg>
</div>
<div class="grid2" style="margin-top:14px">
  <div class="panel">
    <div class="legend"><span><span class="ln" style="border-color:#378ADD"></span>Relative humidity (%)</span>
    <span><span class="ln dash" style="border-color:#EF4444"></span>RC penalty ~70 %</span></div>
    <svg id="rhChart" width="100%" viewBox="0 0 520 240"></svg>
  </div>
  <div class="panel">
    <div class="legend"><span><span class="sw" style="background:#0D9373"></span>Mean monthly rainfall (mm)</span></div>
    <svg id="rainChart" width="100%" viewBox="0 0 520 240"></svg>
  </div>
</div>

<h2 id="aq">🌫 Air quality</h2>
<div class="panel">
  <div class="legend"><span><span class="ln" style="border-color:#9b9b9b"></span>PM2.5 daily</span>
  <span><span class="ln" style="border-color:#7F77DD"></span>30-d rolling</span>
  <span><span class="ln dash" style="border-color:#0D9373"></span>WHO 15</span>
  <span><span class="ln dash" style="border-color:#EF4444"></span>Thai NAAQS 37.5</span></div>
  <svg id="pmChart" width="100%" viewBox="0 0 1040 300"></svg>
</div>
<div class="grid2" style="margin-top:14px">
  <div class="panel"><h3 style="margin-top:2px">Mean PM2.5 by month (µg/m³)</h3>
    <svg id="pmMonth" width="100%" viewBox="0 0 520 230"></svg></div>
  <div class="panel"><h3 style="margin-top:2px">Year × month heatmap</h3>
    <svg id="pmHeat" width="100%" viewBox="0 0 520 230"></svg></div>
</div>
<div class="panel" style="margin-top:14px">
  <h3 style="margin-top:2px">Days above guidelines</h3>
  <table id="exceedTable"></table>
</div>

<h2 id="planner">🧪 RC outdoor-test planner</h2>
<p class="note" style="margin:0 0 8px">Each day scored 0–100 for radiative-cooling field
testing: solar 40 % · PM2.5 haze 30 % · cloud 30 %; rain &gt; 1 mm collapses the score.
<b style="color:#0D9373">VALID ≥ 70</b> · <b style="color:#D97706">MARGINAL 40–70</b> ·
<b style="color:#EF4444">INVALID &lt; 40</b></p>
<div class="panel">
  <svg id="validScatter" width="100%" viewBox="0 0 1040 280"></svg>
</div>
<div class="grid2" style="margin-top:14px">
  <div class="panel"><h3 style="margin-top:2px">% VALID days by month</h3>
    <svg id="validMonth" width="100%" viewBox="0 0 520 230"></svg></div>
  <div class="panel"><h3 style="margin-top:2px">Best campaign months</h3>
    <div id="bestMonths" style="font-size:14px;padding:6px 2px"></div></div>
</div>

<h2 id="ml">🤖 ML forecast — honest evaluation</h2>
<div class="cards" id="mlKpis"></div>
<div class="panel">
  <div class="legend"><span><span class="ln" style="border-color:#5F5E5A"></span>Actual PM2.5</span>
  <span><span class="ln dash" style="border-color:#7F77DD"></span>Random forest</span></div>
  <svg id="mlChart" width="100%" viewBox="0 0 1040 280"></svg>
  <p class="note" id="mlPeriod"></p>
</div>
<div class="grid2" style="margin-top:14px">
  <div class="panel"><h3 style="margin-top:2px">Go / no-go probability vs reality</h3>
    <div class="legend"><span><span class="sw" style="background:#9FE1CB"></span>Actual VALID day</span>
    <span><span class="ln dash" style="border-color:#0F6E56"></span>P(valid tomorrow)</span></div>
    <svg id="probChart" width="100%" viewBox="0 0 520 230"></svg></div>
  <div class="panel"><h3 style="margin-top:2px">Permutation importance (top 8)</h3>
    <svg id="impChart" width="100%" viewBox="0 0 520 230"></svg></div>
</div>

<h2 id="rcmodel">❄ RC monthly performance model</h2>
<div class="seg" id="metricSeg"></div>
<div class="chips" id="matChips"></div>
<div class="panel"><svg id="rcLine" width="100%" viewBox="0 0 1040 330"></svg>
  <p class="note" id="rcNote"></p></div>
<h3>Noon energy budget</h3>
<div id="monthChips"></div>
<div class="panel"><svg id="budgetChart" width="100%" viewBox="0 0 1040 300"></svg>
  <p class="note">Orange = solar absorbed · green = net thermal radiation out (capped
  ~22–31 W/m² by the humid sky — daytime ranking is decided by solar reflectance).</p></div>
<div class="warnBox"><b>Bangkok humidity penalty:</b> night cooling is only
<b id="hcPct"></b> of dry-climate (RH 30 %) performance, and the ideal selective
emitter loses to its broadband twin in all 24 scenarios.</div>

<footer>P4 portfolio project · data: Open-Meteo historical archive + CAMS air quality ·
models: scikit-learn (src/ml_forecast.py) + two-band RC energy balance
(src/rc_monthly_model.py) · single-file report, no external assets.</footer>
</div>

<script>
const D = __DATA__;
const MO=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const CLSC=["#EF4444","#D97706","#0D9373"];
const $=id=>document.getElementById(id);
const fmt=(v,d=1)=>v==null?"—":Number(v).toFixed(d);
$("gen").textContent=D.generated; $("span").textContent=D.kpi.span;
$("nd").textContent=D.kpi.nDays; $("na").textContent=D.kpi.nAq;
$("hcPct").textContent=Math.round(D.rc.hc*100)+" %";

$("kpis").innerHTML=[
 ["Latest mean temp",fmt(D.kpi.tm)+" °C",D.kpi.date],
 ["Humidity",fmt(D.kpi.rh,0)+" %",D.kpi.date],
 ["PM2.5 / US AQI",fmt(D.kpi.pm)+" / "+fmt(D.kpi.aq,0),(D.kpi.pm>D.who?"above":"below")+" WHO guideline"],
 ["Valid RC-test days (30d)",D.kpi.valid30,"of last 30 days"],
 ["Best test window","Mar–Apr","50–55 % valid days"],
].map(([l,v,s])=>`<div class="card"><div class="lbl">${l}</div><div class="val">${v}</div><div class="sub">${s}</div></div>`).join("");

function sliceIdx(){
  const n=D.daily.d.length;
  if(state.win==="all") return 0;
  const last=new Date(D.daily.d[n-1]);
  const cut=new Date(last); cut.setMonth(cut.getMonth()-state.win);
  const c=cut.toISOString().slice(0,10);
  let i=D.daily.d.findIndex(x=>x>=c); return i<0?0:i;
}
const state={win:"all",metric:"night_p",month:"1",
  on:Object.fromEntries(D.rc.mats.map(m=>[m,m!=="Bare concrete"]))};

const wins=[["6m",6],["1y",12],["3y",36],["All","all"]];
wins.forEach(([n,v])=>{
  const b=document.createElement("button"); b.textContent=n; b.dataset.v=v;
  if(String(v)===String(state.win))b.classList.add("on");
  b.onclick=()=>{state.win=v;
    $("winSeg").querySelectorAll("button").forEach(x=>x.classList.toggle("on",x.dataset.v==String(v)));
    drawClimate(); drawPM();};
  $("winSeg").appendChild(b);
});

function xticks(dates,L,W){
  const n=dates.length, out=[];
  const span=(new Date(dates[n-1])-new Date(dates[0]))/2592e6;
  let prev="";
  dates.forEach((d,i)=>{
    const ym=d.slice(0,7), mo=+d.slice(5,7);
    if(ym!==prev && (span>26?mo===1:(span>9?[1,4,7,10].includes(mo):true))){
      out.push([L+i*(W)/(n-1), span>26?d.slice(0,4):MO[mo-1]+" "+d.slice(2,4)]); prev=ym;}
  });
  return out;
}

function lineFrame(svg,lo,hi,L,T,W,H,dates,fdec){
  let s=""; const y=v=>T+(hi-v)*H/(hi-lo);
  for(let g=0;g<=4;g++){const v=lo+g*(hi-lo)/4, yy=y(v);
    s+=`<line class="axis" x1="${L}" y1="${yy}" x2="${L+W}" y2="${yy}"/>`+
       `<text x="${L-7}" y="${yy+4}" text-anchor="end">${fmt(v,fdec)}</text>`;}
  if(dates) xticks(dates,L,W).forEach(([xx,lab])=>
    s+=`<text x="${xx}" y="${T+H+18}" text-anchor="middle">${lab}</text>`);
  return [s,y];
}
const path=(arr,x,y)=>{let p="",pen=false;
  arr.forEach((v,i)=>{if(v==null){pen=false;return;}
    p+=(pen?"L":"M")+x(i).toFixed(1)+" "+y(v).toFixed(1); pen=true;});
  return p;}

function drawClimate(){
  const i0=sliceIdx(), d=D.daily, dates=d.d.slice(i0);
  const tx=d.tx.slice(i0), tn=d.tn.slice(i0), tm=d.tm.slice(i0);
  const roll=tm.map((_,i)=>{const a=tm.slice(Math.max(0,i-13),i+1).filter(v=>v!=null);
    return a.length?a.reduce((p,c)=>p+c,0)/a.length:null;});
  const L=46,T=12,W=976,H=240, lo=Math.min(...tn.filter(v=>v!=null))-1,
        hi=Math.max(...tx.filter(v=>v!=null))+1;
  const x=i=>L+i*W/(dates.length-1);
  let [s,y]=lineFrame($("tempChart"),lo,hi,L,T,W,H,dates,0);
  let band="";
  tx.forEach((v,i)=>{if(v!=null)band+=(band?"L":"M")+x(i).toFixed(1)+" "+y(v).toFixed(1);});
  for(let i=tn.length-1;i>=0;i--) if(tn[i]!=null) band+="L"+x(i).toFixed(1)+" "+y(tn[i]).toFixed(1);
  s+=`<path d="${band}Z" fill="#D85A30" opacity=".14"/>`+
     `<path d="${path(roll,x,y)}" fill="none" stroke="#D85A30" stroke-width="2.2"/>`;
  $("tempChart").innerHTML=s;

  const rh=d.rh.slice(i0);
  const L2=42,W2=460,H2=180;
  let [s2,y2]=lineFrame($("rhChart"),35,100,L2,12,W2,H2,dates,0);
  const x2=i=>L2+i*W2/(dates.length-1);
  s2+=`<line x1="${L2}" y1="${y2(70)}" x2="${L2+W2}" y2="${y2(70)}" stroke="#EF4444" stroke-dasharray="5 4"/>`+
      `<path d="${path(rh,x2,y2)}" fill="none" stroke="#378ADD" stroke-width="1.6"/>`;
  $("rhChart").innerHTML=s2;

  monthBars($("rainChart"),D.monthly.rain,"#0D9373",0);
}

function monthBars(svg,vals,color,dec,suffix=""){
  const L=42,T=12,W=460,H=180, hi=Math.max(...vals.filter(v=>v!=null))*1.12||1;
  let [s,y]=lineFrame(svg,0,hi,L,T,W,H,null,dec);
  const bw=W/12*0.62;
  vals.forEach((v,i)=>{if(v==null)return;
    const xx=L+(i+0.5)*W/12;
    s+=`<rect x="${xx-bw/2}" y="${y(v)}" width="${bw}" height="${T+H-y(v)}" rx="3" fill="${color}"><title>${MO[i]}: ${fmt(v,dec)}${suffix}</title></rect>`+
       `<text x="${xx}" y="${T+H+16}" text-anchor="middle">${MO[i]}</text>`;});
  svg.innerHTML=s;
}

function drawPM(){
  const i0=sliceIdx(), d=D.daily;
  const idx=[]; for(let i=i0;i<d.d.length;i++) if(d.pm[i]!=null) idx.push(i);
  if(!idx.length){$("pmChart").innerHTML="";return;}
  const dates=idx.map(i=>d.d[i]), pm=idx.map(i=>d.pm[i]), pr=idx.map(i=>d.pr[i]);
  const L=46,T=12,W=976,H=240, hi=Math.max(...pm)*1.06;
  const x=i=>L+i*W/(dates.length-1);
  let [s,y]=lineFrame($("pmChart"),0,hi,L,T,W,H,dates,0);
  s+=`<path d="${path(pm,x,y)}" fill="none" stroke="#9b9b9b" stroke-width="1" opacity=".6"/>`+
     `<path d="${path(pr,x,y)}" fill="none" stroke="#7F77DD" stroke-width="2.4"/>`+
     `<line x1="${L}" y1="${y(D.who)}" x2="${L+W}" y2="${y(D.who)}" stroke="#0D9373" stroke-dasharray="4 4"/>`+
     `<line x1="${L}" y1="${y(D.thai)}" x2="${L+W}" y2="${y(D.thai)}" stroke="#EF4444" stroke-dasharray="5 4"/>`;
  $("pmChart").innerHTML=s;
}

monthBars($("pmMonth"),D.monthly.pm,"#D97706",0," µg/m³");

(function heat(){
  const g=D.heat.grid, ys=D.heat.years;
  const vals=g.flat().filter(v=>v!=null), lo=Math.min(...vals), hi=Math.max(...vals);
  const L=52,T=10,W=440,H=170, ch=H/ys.length, cw=W/12;
  let s="";
  ys.forEach((yr,r)=>{
    s+=`<text x="${L-8}" y="${T+r*ch+ch/2+4}" text-anchor="end">${yr}</text>`;
    g[r].forEach((v,c)=>{
      if(v==null)return;
      const t=(v-lo)/(hi-lo), col=`hsl(${45-t*45} 85% ${62-t*22}%)`;
      s+=`<rect x="${L+c*cw+1}" y="${T+r*ch+1}" width="${cw-2}" height="${ch-2}" rx="3" fill="${col}"><title>${yr} ${MO[c]}: ${fmt(v)} µg/m³</title></rect>`;
      if(v>=D.thai) s+=`<text x="${L+c*cw+cw/2}" y="${T+r*ch+ch/2+4}" text-anchor="middle" fill="#fff" font-size="10">${Math.round(v)}</text>`;});
  });
  MO.forEach((m,c)=>s+=`<text x="${L+c*cw+cw/2}" y="${T+H+16}" text-anchor="middle">${m}</text>`);
  $("pmHeat").innerHTML=s;
})();

$("exceedTable").innerHTML="<tr><th>Year</th><th>AQ days</th><th>&gt; WHO 15</th><th>&gt; Thai 37.5</th><th>% WHO</th><th>% Thai</th></tr>"+
  D.exceed.map(e=>`<tr><td>${e.y}</td><td>${e.n}</td><td>${e.who}</td><td>${e.thai}</td><td>${e.pwho} %</td><td>${e.pthai} %</td></tr>`).join("");

(function validity(){
  const d=D.daily, idx=[];
  for(let i=0;i<d.d.length;i++) if(d.tv[i]!=null&&d.tc[i]>=0) idx.push(i);
  const L=46,T=12,W=976,H=220;
  const x=k=>L+k*W/(idx.length-1);
  let [s,y]=lineFrame($("validScatter"),0,100,L,T,W,H,idx.map(i=>d.d[i]),0);
  s+=`<line x1="${L}" y1="${y(70)}" x2="${L+W}" y2="${y(70)}" stroke="#0D9373" stroke-dasharray="4 4"/>`+
     `<line x1="${L}" y1="${y(40)}" x2="${L+W}" y2="${y(40)}" stroke="#D97706" stroke-dasharray="4 4"/>`;
  idx.forEach((i,k)=>{
    s+=`<circle cx="${x(k).toFixed(1)}" cy="${y(d.tv[i]).toFixed(1)}" r="2.1" fill="${CLSC[d.tc[i]]}" opacity=".65"><title>${d.d[i]}: ${d.tv[i]}</title></circle>`;});
  $("validScatter").innerHTML=s;

  monthBars($("validMonth"),D.monthly.valid,"#0D9373",0," %");
  const ranked=D.monthly.valid.map((v,i)=>[v,MO[i]]).sort((a,b)=>b[0]-a[0]).slice(0,3);
  $("bestMonths").innerHTML=ranked.map(([v,m],i)=>
    `<div style="margin:6px 0"><b style="color:#0D9373">${i+1}. ${m}</b> — ${fmt(v,0)} % of days valid</div>`).join("")+
    `<p class="note">Jan–Feb skies are dry but burning-season haze (PM2.5 ≈ 45–49 µg/m³)
    suppresses validity; the sweet spot is haze decay + pre-monsoon sun.</p>`;
})();

const M=D.ml.metrics;
$("mlKpis").innerHTML=[
 ["Persistence MAE",fmt(M.persistence_mae,2)+" µg/m³","baseline: tomorrow = today"],
 ["Best model MAE",fmt(M.best_mae,2)+" µg/m³",M.best_name+" — baseline wins"],
 ["VALID-day classifier AUC",fmt(M.auc,2),"with forecast-weather features"],
 ["Classifier F1",fmt(M.f1,2),"vs naive "+fmt(M.naive_f1,2)],
].map(([l,v,s])=>`<div class="card"><div class="lbl">${l}</div><div class="val">${v}</div><div class="sub">${s}</div></div>`).join("");
$("mlPeriod").textContent="Held-out test period (chronological split): "+M.period;

(function mlChart(){
  const m=D.ml, L=46,T=12,W=976,H=220;
  const hi=Math.max(...m.act,...m.mod)*1.06;
  const x=i=>L+i*W/(m.d.length-1);
  let [s,y]=lineFrame($("mlChart"),0,hi,L,T,W,H,m.d,0);
  s+=`<path d="${path(m.act,x,y)}" fill="none" stroke="#5F5E5A" stroke-width="1.8"/>`+
     `<path d="${path(m.mod,x,y)}" fill="none" stroke="#7F77DD" stroke-width="1.8" stroke-dasharray="6 4"/>`;
  $("mlChart").innerHTML=s;

  const L2=40,W2=460,H2=170;
  let [s2,y2]=lineFrame($("probChart"),0,1,L2,12,W2,H2,m.d,1);
  const x2=i=>L2+i*W2/(m.d.length-1);
  m.vact.forEach((v,i)=>{if(v)s2+=`<rect x="${x2(i)-1}" y="${y2(1)}" width="2" height="${H2}" fill="#9FE1CB" opacity=".5"/>`;});
  s2+=`<path d="${path(m.proba,x2,y2)}" fill="none" stroke="#0F6E56" stroke-width="1.8" stroke-dasharray="5 3"/>`;
  $("probChart").innerHTML=s2;

  const imp=D.ml.imp, hi3=Math.max(...imp.map(r=>r.importance));
  const L3=170,W3=320, rh3=170/imp.length;
  let s3="";
  imp.forEach((r,i)=>{
    const w=W3*r.importance/hi3, yy=14+i*rh3;
    s3+=`<text x="${L3-8}" y="${yy+rh3/2+3}" text-anchor="end">${r.feature}</text>`+
        `<rect x="${L3}" y="${yy}" width="${Math.max(w,2)}" height="${rh3-7}" rx="3" fill="#7F77DD"><title>${r.feature}: ${r.importance}</title></rect>`+
        `<text x="${L3+Math.max(w,2)+6}" y="${yy+rh3/2+3}">${r.importance}</text>`;});
  $("impChart").innerHTML=s3;
})();

Object.entries(D.rc.metrics).forEach(([k,m])=>{
  const b=document.createElement("button");
  b.textContent=m.label.split("(")[0].trim(); b.dataset.k=k;
  if(k===state.metric)b.classList.add("on");
  b.onclick=()=>{state.metric=k;
    $("metricSeg").querySelectorAll("button").forEach(x=>x.classList.toggle("on",x.dataset.k===k));
    drawRC();};
  $("metricSeg").appendChild(b);
});
D.rc.mats.forEach(m=>{
  const c=document.createElement("div");
  c.className="chip"+(state.on[m]?" on":""); c.style.color=D.rc.colors[m];
  c.innerHTML=`<span class="dot" style="background:${D.rc.colors[m]}"></span>${m}`;
  c.onclick=()=>{state.on[m]=!state.on[m];c.classList.toggle("on",state.on[m]);drawRC();drawBudget();};
  $("matChips").appendChild(c);
});
MO.forEach((n,i)=>{
  const c=document.createElement("span");
  c.className="mchip"+((String(i+1))===state.month?" on":"");
  c.textContent=n; c.dataset.m=String(i+1);
  c.onclick=()=>{state.month=String(i+1);
    $("monthChips").querySelectorAll(".mchip").forEach(x=>x.classList.toggle("on",x.dataset.m===state.month));
    drawBudget();};
  $("monthChips").appendChild(c);
});

function drawRC(){
  const m=D.rc.metrics[state.metric], mats=D.rc.mats.filter(x=>state.on[x]);
  const svg=$("rcLine"), all=mats.flatMap(x=>m.data[x]);
  if(!all.length){svg.innerHTML="";return;}
  let lo=Math.min(...all),hi=Math.max(...all);
  if(m.zero){lo=Math.min(lo,0);hi=Math.max(hi,0);}
  const pad=(hi-lo)*.08||1; lo-=pad;hi+=pad;
  const L=52,T=14,W=952,H=270;
  const x=i=>L+i*W/11, y=v=>T+(hi-v)*H/(hi-lo);
  let s="";
  for(let g=0;g<=5;g++){const v=lo+g*(hi-lo)/5,yy=y(v);
    s+=`<line class="axis" x1="${L}" y1="${yy}" x2="${L+W}" y2="${yy}"/>`+
       `<text x="${L-8}" y="${yy+4}" text-anchor="end">${fmt(v,Math.abs(hi-lo)<8?1:0)}</text>`;}
  if(m.zero&&lo<0&&hi>0)
    s+=`<line x1="${L}" y1="${y(0)}" x2="${L+W}" y2="${y(0)}" stroke="var(--text2)" stroke-dasharray="5 4"/>`;
  MO.forEach((n,i)=>s+=`<text x="${x(i)}" y="${T+H+18}" text-anchor="middle">${n}</text>`);
  mats.forEach(mt=>{
    const pts=m.data[mt].map((v,i)=>`${x(i)},${y(v).toFixed(1)}`).join(" ");
    s+=`<polyline points="${pts}" fill="none" stroke="${D.rc.colors[mt]}" stroke-width="2.4"/>`;
    m.data[mt].forEach((v,i)=>s+=`<circle cx="${x(i)}" cy="${y(v).toFixed(1)}" r="3.2" fill="${D.rc.colors[mt]}"><title>${mt} — ${MO[i]}: ${fmt(v)}</title></circle>`);});
  svg.innerHTML=s;
  $("rcNote").textContent=m.label;
}

function drawBudget(){
  const bud=D.rc.budget[state.month];
  const mats=D.rc.mats.filter(m=>state.on[m]||m==="Bare concrete").filter(m=>bud[m])
    .sort((a,b)=>bud[b].abs-bud[a].abs);
  const maxA=Math.max(...mats.map(m=>bud[m].abs),1), maxR=Math.max(...mats.map(m=>bud[m].rad),1);
  const W=1040,L=270, zero=L+(W-L-40)*maxA/(maxA+maxR+60);
  const rh=Math.min(34,250/mats.length);
  let s=`<line class="axis" x1="${zero}" y1="6" x2="${zero}" y2="${16+mats.length*rh}"/>`;
  mats.forEach((m,i)=>{
    const yy=12+i*rh,b=bud[m];
    const wA=(zero-L-6)*b.abs/maxA, wR=(W-zero-46)*b.rad/maxR;
    s+=`<text x="${L-8}" y="${yy+rh/2}" text-anchor="end">${m}</text>`+
       `<rect x="${zero-wA}" y="${yy}" width="${wA}" height="${rh-8}" rx="3" fill="#D97706" opacity=".85"><title>${m}: solar ${fmt(b.abs,0)} W/m²</title></rect>`+
       `<rect x="${zero+2}" y="${yy}" width="${Math.max(wR,2)}" height="${rh-8}" rx="3" fill="#0D9373"><title>${m}: radiated ${fmt(b.rad,0)} W/m²</title></rect>`+
       `<text x="${zero-wA-6}" y="${yy+rh/2}" text-anchor="end">${fmt(b.abs,0)}</text>`+
       `<text x="${zero+Math.max(wR,2)+6}" y="${yy+rh/2}">${fmt(b.rad,0)}</text>`;});
  $("budgetChart").setAttribute("viewBox",`0 0 1040 ${24+mats.length*rh}`);
  $("budgetChart").innerHTML=s;
}

$("themeBtn").onclick=()=>{const b=document.body.parentElement;
  b.dataset.theme=b.dataset.theme==="dark"?"":"dark";};
if(matchMedia("(prefers-color-scheme: dark)").matches)
  document.body.parentElement.dataset.theme="dark";

drawClimate(); drawPM(); drawRC(); drawBudget();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
