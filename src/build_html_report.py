# -*- coding: utf-8 -*-
"""Self-contained interactive HTML dashboard for the RC monthly model.

Follows the project dashboard v2 style (see storage/research/
fly-ash-bottom-ash-review/build_dashboard_v2.py): single file, embedded
JSON, restrained palette with consistent color meaning, clickable filters,
dark/light toggle, zero external assets (charts are hand-built SVG).

Output: wiki/_reports/research/rc-monthly-model-dashboard.html
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent.parent          # project root
REPO = SRC.parent.parent                              # MisterX-AI root
OUT_DIR = REPO / "wiki" / "_reports" / "research"
OUT = OUT_DIR / "rc-monthly-model-dashboard.html"

COLORS = {
    "Bare concrete": "#888780",
    "White TiO2 paint": "#E0A82E",
    "Coating C": "#D85A30",
    "PVDF-HFP porous": "#378ADD",
    "BaSO4 ultra-white": "#7F77DD",
    "Ideal selective emitter": "#1D9E75",
    "Ideal broadband emitter": "#D4537E",
}


def main():
    rcm = pd.read_csv(SRC / "data" / "rc_monthly_model.csv")
    summary = json.loads((SRC / "data" / "rc_monthly_summary.json").read_text(encoding="utf-8"))
    mats = [m for m in COLORS if m in set(rcm["material"])]

    def series(scen, col):
        return {m: [float(v) for v in
                    rcm[(rcm.material == m) & (rcm.scenario == scen)]
                    .sort_values("month")[col]] for m in mats}

    budget = {}
    for month in range(1, 13):
        sub = rcm[(rcm.scenario == "day") & (rcm.month == month)]
        budget[str(month)] = {
            r["material"]: {"abs": float(r["P_solar_abs_Wm2"]),
                            "rad": float(r["P_rad_net_Wm2"])}
            for _, r in sub.iterrows()
        }

    clim = summary["climatology"]
    data = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "mats": mats,
        "colors": COLORS,
        "metrics": {
            "night_p": {"label": "Night cooling power (W/m²)", "zero": False,
                        "data": series("night", "P_cool_at_ambient_Wm2")},
            "day_p": {"label": "Noon cooling power (W/m², + = net cooling)",
                      "zero": True, "data": series("day", "P_cool_at_ambient_Wm2")},
            "day_dt": {"label": "Noon equilibrium ΔT (°C, + = below ambient)",
                       "zero": True, "data": series("day", "dT_eq_C")},
        },
        "budget": budget,
        "clim": clim,
        "rows": rcm.to_dict("records"),
        "kpi": {
            "nightRange": summary["coating_c_night_P_cool_range_Wm2"],
            "hc": summary["coating_c_mean_humidity_correction_night"],
        },
    }
    data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")

    html = HTML_TEMPLATE.replace("__DATA__", data_json)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".html.tmp")
    tmp.write_text(html, encoding="utf-8")
    tmp.replace(OUT)
    print(f"Saved → {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bangkok RC Performance — Monthly Model</title>
<style>
:root{
  --brand:#0D9373; --warn:#D97706; --bad:#EF4444;
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
body{margin:0;background:var(--bg);color:var(--text);
  font:15px/1.6 Inter,-apple-system,"Segoe UI",system-ui,sans-serif;
  font-variant-numeric:tabular-nums}
.wrap{max-width:1060px;margin:0 auto;padding:0 20px 70px}
header.hero{display:flex;justify-content:space-between;align-items:flex-end;
  padding:38px 0 14px}
.hero h1{font-size:27px;font-weight:800;letter-spacing:-.02em;margin:0}
.hero h1 .accent{color:var(--brand)}
.hero p{color:var(--text2);margin:5px 0 0;font-size:14px}
#themeBtn{background:var(--chip);border:1px solid var(--border);color:var(--text);
  border-radius:8px;padding:6px 12px;cursor:pointer;font-size:13px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
  gap:12px;margin:14px 0 26px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;
  padding:14px 16px;box-shadow:var(--shadow)}
.card .lbl{font-size:12.5px;color:var(--text2)}
.card .val{font-size:23px;font-weight:700;margin-top:2px}
.card .sub{font-size:12px;color:var(--text2)}
h2{font-size:17px;font-weight:700;margin:30px 0 10px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}
.chip{display:flex;align-items:center;gap:6px;background:var(--chip);
  border:1px solid var(--border);border-radius:999px;padding:5px 12px;
  font-size:13px;cursor:pointer;user-select:none;opacity:.45}
.chip.on{opacity:1;border-color:currentColor}
.chip .dot{width:10px;height:10px;border-radius:3px}
.seg{display:inline-flex;background:var(--chip);border:1px solid var(--border);
  border-radius:10px;overflow:hidden;margin:4px 0 10px}
.seg button{background:none;border:none;color:var(--text2);padding:7px 14px;
  font-size:13px;cursor:pointer}
.seg button.on{background:var(--card);color:var(--text);font-weight:600;
  box-shadow:var(--shadow)}
svg text{fill:var(--text2);font-size:11.5px}
svg .axis{stroke:var(--border)}
.panel{background:var(--card);border:1px solid var(--border);border-radius:12px;
  padding:16px;box-shadow:var(--shadow)}
.note{font-size:13px;color:var(--text2);margin-top:8px}
.mchips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 12px}
.mchip{background:var(--chip);border:1px solid var(--border);border-radius:8px;
  padding:4px 10px;font-size:12.5px;cursor:pointer}
.mchip.on{background:var(--brand);color:#fff;border-color:var(--brand)}
details{margin-top:14px}
summary{cursor:pointer;color:var(--brand);font-size:14px}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:10px}
th,td{border-bottom:1px solid var(--border);padding:6px 8px;text-align:right}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
th{color:var(--text2);font-weight:600;position:sticky;top:0;background:var(--card)}
.tableBox{max-height:420px;overflow:auto;border:1px solid var(--border);
  border-radius:10px;margin-top:10px}
.warnBox{background:color-mix(in srgb,var(--warn) 10%,var(--card));
  border:1px solid color-mix(in srgb,var(--warn) 35%,var(--border));
  border-radius:12px;padding:12px 16px;font-size:14px;margin-top:22px}
</style>
</head>
<body>
<div class="wrap">
<header class="hero">
  <div>
    <h1>Bangkok <span class="accent">radiative cooling</span> — monthly model</h1>
    <p>Two-band energy balance driven by 2019–2026 Open-Meteo climatology ·
       generated <span id="gen"></span></p>
  </div>
  <button id="themeBtn">◐ theme</button>
</header>

<div class="cards" id="kpis"></div>

<h2>Material performance by month</h2>
<div class="seg" id="metricSeg"></div>
<div class="chips" id="matChips"></div>
<div class="panel"><svg id="lineChart" width="100%" height="360"
  viewBox="0 0 1000 360" preserveAspectRatio="xMidYMid meet"></svg>
  <p class="note" id="lineNote"></p></div>

<h2>Noon energy budget</h2>
<div class="mchips" id="monthChips"></div>
<div class="panel"><svg id="budgetChart" width="100%" height="320"
  viewBox="0 0 1000 320" preserveAspectRatio="xMidYMid meet"></svg>
  <p class="note">Orange = solar heat absorbed · green = net thermal radiation out.
  The radiative term is capped at ~22–31 W/m² by the humid sky — daytime ranking
  is decided almost entirely by solar reflectance.</p></div>

<div class="warnBox"><b>Bangkok humidity penalty:</b> night cooling is only
<b id="hcPct"></b> of dry-climate (RH 30 %) performance. The humid sky also makes
the ideal <i>selective</i> emitter lose to its <i>broadband</i> twin in all 24
scenarios — spend the design budget on solar reflectance, not selectivity.</div>

<details><summary>Model assumptions</summary>
<p class="note">8–13 µm window sky emissivity from RH + cloud, anchored to 0.84 at
RH 80 % (project baseline); outside-window sky ε = 0.97; h = 5.5 W/m²K; insulated
surface (no conduction); monthly-mean weather; noon = half-sine peak of mean daily
irradiation; day T = mean monthly T<sub>max</sub>, night T = mean T<sub>min</sub>.
Equilibrium solved with Brent's method. Source:
projects/bangkok_weather_aqi/src/rc_monthly_model.py</p></details>

<details open><summary>Full model table (filtered by selected materials)</summary>
<div class="tableBox"><table id="dataTable"></table></div></details>
</div>

<script>
const D = __DATA__;
const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const state = {metric:"night_p", month:"1",
  on:Object.fromEntries(D.mats.map(m=>[m, m!=="Bare concrete"]))};

document.getElementById("gen").textContent = D.generated;
document.getElementById("hcPct").textContent = Math.round(D.kpi.hc*100)+" %";

const fmt = (v,d=1)=>Number(v).toFixed(d);
document.getElementById("kpis").innerHTML = [
 ["Night cooling (Coating C)", fmt(D.kpi.nightRange[0],0)+"–"+fmt(D.kpi.nightRange[1],0)+" W/m²","best Dec–Mar, worst Jul–Sep"],
 ["vs dry climate", Math.round(D.kpi.hc*100)+" %","of night cooling power"],
 ["Noon subambient", "R ≳ 0.95","BaSO4-class only, Nov–Apr"],
 ["Selectivity cost", "−8 W/m²","broadband beats selective 24/24"],
].map(([l,v,s])=>`<div class="card"><div class="lbl">${l}</div>
  <div class="val">${v}</div><div class="sub">${s}</div></div>`).join("");

const seg = document.getElementById("metricSeg");
Object.entries(D.metrics).forEach(([k,m])=>{
  const b=document.createElement("button");
  b.textContent=m.label.split("(")[0].trim();
  b.dataset.k=k; if(k===state.metric) b.classList.add("on");
  b.onclick=()=>{state.metric=k;
    seg.querySelectorAll("button").forEach(x=>x.classList.toggle("on",x.dataset.k===k));
    drawLine(); drawTable();};
  seg.appendChild(b);
});

const chips=document.getElementById("matChips");
D.mats.forEach(m=>{
  const c=document.createElement("div");
  c.className="chip"+(state.on[m]?" on":""); c.style.color=D.colors[m];
  c.innerHTML=`<span class="dot" style="background:${D.colors[m]}"></span>${m}`;
  c.onclick=()=>{state.on[m]=!state.on[m]; c.classList.toggle("on",state.on[m]);
    drawLine(); drawBudget(); drawTable();};
  chips.appendChild(c);
});

const mchips=document.getElementById("monthChips");
MONTHS.forEach((name,i)=>{
  const c=document.createElement("div");
  c.className="mchip"+((String(i+1))===state.month?" on":"");
  c.textContent=name; c.dataset.m=String(i+1);
  c.onclick=()=>{state.month=String(i+1);
    mchips.querySelectorAll(".mchip").forEach(x=>
      x.classList.toggle("on",x.dataset.m===state.month));
    drawBudget();};
  mchips.appendChild(c);
});

function activeMats(){return D.mats.filter(m=>state.on[m]);}

function drawLine(){
  const M=D.metrics[state.metric], mats=activeMats();
  const svg=document.getElementById("lineChart");
  const all=mats.flatMap(m=>M.data[m]);
  if(!all.length){svg.innerHTML="";return;}
  let lo=Math.min(...all), hi=Math.max(...all);
  if(M.zero){lo=Math.min(lo,0);hi=Math.max(hi,0);}
  const pad=(hi-lo)*0.08||1; lo-=pad; hi+=pad;
  const L=56,R=16,T=14,B=34,W=1000,H=360;
  const x=i=>L+i*(W-L-R)/11, y=v=>T+(hi-v)*(H-T-B)/(hi-lo);
  let s="";
  const step=(hi-lo)/5;
  for(let g=0;g<=5;g++){const v=lo+g*step, yy=y(v);
    s+=`<line class="axis" x1="${L}" y1="${yy}" x2="${W-R}" y2="${yy}"/>`+
       `<text x="${L-8}" y="${yy+4}" text-anchor="end">${fmt(v,Math.abs(hi-lo)<8?1:0)}</text>`;}
  if(M.zero && lo<0 && hi>0)
    s+=`<line x1="${L}" y1="${y(0)}" x2="${W-R}" y2="${y(0)}" stroke="var(--text2)" stroke-dasharray="5 4"/>`;
  MONTHS.forEach((n,i)=>s+=`<text x="${x(i)}" y="${H-10}" text-anchor="middle">${n}</text>`);
  mats.forEach(m=>{
    const pts=M.data[m].map((v,i)=>`${x(i)},${y(v)}`).join(" ");
    s+=`<polyline points="${pts}" fill="none" stroke="${D.colors[m]}" stroke-width="2.5"/>`;
    M.data[m].forEach((v,i)=>{
      s+=`<circle cx="${x(i)}" cy="${y(v)}" r="3.4" fill="${D.colors[m]}">`+
         `<title>${m} — ${MONTHS[i]}: ${fmt(v)}</title></circle>`;});
  });
  svg.innerHTML=s;
  document.getElementById("lineNote").textContent=M.label;
}

function drawBudget(){
  const mats=activeMats().concat(state.on["Bare concrete"]?[]:["Bare concrete"]);
  const bud=D.budget[state.month];
  const rows=mats.filter(m=>bud[m]).sort((a,b)=>bud[b].abs-bud[a].abs);
  const maxAbs=Math.max(...rows.map(m=>bud[m].abs),1);
  const maxRad=Math.max(...rows.map(m=>bud[m].rad),1);
  const W=1000,L=270,zero=L+(W-L-30)*maxAbs/(maxAbs+maxRad+60);
  const rh=Math.min(34,(300-30)/rows.length), svg=document.getElementById("budgetChart");
  let s=`<line class="axis" x1="${zero}" y1="8" x2="${zero}" y2="${20+rows.length*rh}"/>`;
  rows.forEach((m,i)=>{
    const yy=16+i*rh, b=bud[m];
    const wAbs=(zero-L-6)*b.abs/maxAbs, wRad=(W-zero-40)*b.rad/maxRad;
    s+=`<text x="${L-8}" y="${yy+rh/2}" text-anchor="end">${m}</text>`+
       `<rect x="${zero-wAbs}" y="${yy}" width="${wAbs}" height="${rh-9}" rx="3" fill="#D97706" opacity=".85"><title>${m}: solar absorbed ${fmt(b.abs,0)} W/m²</title></rect>`+
       `<rect x="${zero+2}" y="${yy}" width="${Math.max(wRad,2)}" height="${rh-9}" rx="3" fill="#0D9373"><title>${m}: radiated ${fmt(b.rad,0)} W/m²</title></rect>`+
       `<text x="${zero-wAbs-6}" y="${yy+rh/2}" text-anchor="end" font-size="11">${fmt(b.abs,0)}</text>`+
       `<text x="${zero+Math.max(wRad,2)+6}" y="${yy+rh/2}">${fmt(b.rad,0)}</text>`;
  });
  svg.setAttribute("viewBox",`0 0 1000 ${30+rows.length*rh}`);
  svg.innerHTML=s;
}

function drawTable(){
  const mats=new Set(activeMats());
  const cols=["month","material","scenario","T_amb_C","G_Wm2","P_solar_abs_Wm2",
    "P_rad_net_Wm2","P_cool_at_ambient_Wm2","dT_eq_C","T_surface_eq_C","humidity_correction"];
  const head="<tr>"+cols.map(c=>`<th>${c.replaceAll("_"," ")}</th>`).join("")+"</tr>";
  const body=D.rows.filter(r=>mats.has(r.material))
    .map(r=>"<tr>"+cols.map(c=>`<td>${r[c]==null?"—":r[c]}</td>`).join("")+"</tr>").join("");
  document.getElementById("dataTable").innerHTML=head+body;
}

document.getElementById("themeBtn").onclick=()=>{
  const b=document.body.parentElement;
  b.dataset.theme=b.dataset.theme==="dark"?"":"dark";};
if(matchMedia("(prefers-color-scheme: dark)").matches)
  document.body.parentElement.dataset.theme="dark";

drawLine(); drawBudget(); drawTable();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
