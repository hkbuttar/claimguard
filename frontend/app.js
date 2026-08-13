const api = async (path, options = {}) => {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!response.ok) { const body = await response.json(); throw new Error(body.detail || `Request failed (${response.status})`); }
  return response.json();
};
const money = value => new Intl.NumberFormat("en-US", { style: "currency", currency: "EUR", maximumFractionDigits: 0 }).format(value ?? 0);
const number = value => new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value ?? 0);
const pct = value => `${number((value ?? 0) * 100)}%`;
const titles = { overview:"Portfolio overview", explorer:"Policy risk explorer", models:"Actuarial vs. ML lab", tail:"Tail-risk lab", portfolio:"Portfolio risk", bonus:"Bonus-Malus analysis" };

document.querySelectorAll(".nav-item").forEach(button => button.addEventListener("click", () => {
  document.querySelectorAll(".nav-item,.view").forEach(item => item.classList.remove("active"));
  button.classList.add("active"); document.getElementById(button.dataset.view).classList.add("active");
  document.getElementById("page-title").textContent = titles[button.dataset.view];
}));

const metrics = (target, items) => document.getElementById(target).innerHTML = items.map(([label,value,note]) => `<article class="metric"><small>${label}</small><strong>${value}</strong>${note ? `<em>${note}</em>`:""}</article>`).join("");
const bars = (target, rows, labelKey, valueKey, formatter = number) => {
  const max = Math.max(...rows.map(row => Number(row[valueKey]) || 0), 1);
  document.getElementById(target).innerHTML = rows.map(row => `<div class="bar-row"><span>${row[labelKey]}</span><div class="bar-track"><div class="bar-fill" style="width:${100 * Number(row[valueKey]) / max}%"></div></div><strong>${formatter(row[valueKey])}</strong></div>`).join("");
};
const table = (target, rows, columns) => {
  if (!rows.length) return;
  document.getElementById(target).innerHTML = `<table><thead><tr>${columns.map(([key,label]) => `<th>${label}</th>`).join("")}</tr></thead><tbody>${rows.map(row => `<tr>${columns.map(([key,,format]) => `<td>${format ? format(row[key]) : row[key] ?? "—"}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
};

async function loadOverview() {
  const data = await api("/portfolio/summary"); const p = data.portfolio; const s = data.severity; const c = data.loss_concentration;
  metrics("overview-metrics", [["Policies",number(p.policies),"earned exposures"],["Recorded claims",number(p.recorded_claims),"frequency target"],["Total exposure",number(p.exposure),"policy-years"],["Total linked loss",money(p.linked_loss),"severity records"],["Claim frequency",number(p.frequency),"per policy-year"],["Mean severity",money(s.mean),"conditional on claim"],["Pure premium",money(p.linked_loss / p.exposure),"loss per exposure"],["Largest claim",money(s.maximum),"observed maximum"]]);
  const rows = c.map(row => ({label:`Top ${number(row.TopClaimShare * 100)}%`,value:row.LossShare})); bars("concentration-bars",rows,"label","value",pct);
  const top = rows.at(-1); document.getElementById("tail-insight").textContent = `${top.label} of claims accounts for ${pct(top.value)} of linked losses. Average-error metrics alone cannot describe this concentration, so ClaimGuard separates expected loss from tail scenarios.`;
}

async function loadModels() {
  const [rows,deciles] = await Promise.all([api("/models/benchmark"),api("/portfolio/risk-deciles")]);
  const task = document.getElementById("model-task"); const tasks = [...new Set(rows.map(row => row.Task))]; task.innerHTML = tasks.map(value => `<option>${value}</option>`).join("");
  const render = () => table("benchmark-table",rows.filter(row => row.Task === task.value),[["Metric","Metric"],["TraditionalModel","Traditional model"],["TraditionalValue","Value",number],["MLModel","ML model"],["MLValue","Value",number],["Winner","Winner"]]); task.addEventListener("change",render); render();
  const model = document.getElementById("decile-model"); const renderDeciles = () => bars("decile-chart",deciles[model.value],"RiskDecile","ObservedLossCost",money); model.addEventListener("change",renderDeciles); renderDeciles();
}

async function loadTail() {
  const [data,quantiles] = await Promise.all([api("/tail-risk"),api("/tail-risk/quantiles")]); const g = data.gpd;
  metrics("tail-metrics",[["EVT threshold",money(g.threshold),`${number(g.exceedances)} exceedances`],["Tail probability",pct(g.exceedance_probability),"above threshold"],["GPD shape",number(g.shape),g.finite_variance ? "finite variance":"infinite fitted variance"],["Expected tail claim",money(g.expected_claim_given_exceedance),"conditional exceedance"]]);
  table("quantile-table",quantiles,[["Probability","Quantile",pct],["Empirical","Empirical",money],["Gamma","Gamma",money],["Lognormal","Lognormal",money],["EVT","EVT",money]]);
  document.getElementById("gpd-details").innerHTML = [["Shape",number(g.shape)],["Scale",money(g.scale)],["Mean finite",g.finite_mean ? "Yes":"No"],["Variance finite",g.finite_variance ? "Yes":"No"]].map(([a,b]) => `<div><span>${a}</span><strong>${b}</strong></div>`).join("");
}

async function loadPortfolio() {
  const [stress,segments] = await Promise.all([api("/portfolio/stress-test"),api("/portfolio/segments")]); const full = stress.full_tail;
  metrics("stress-metrics",[["Expected aggregate loss",money(full.mean),`${number(stress.simulations)} simulations`],["95% VaR",money(full.var_95),"annual aggregate"],["99% VaR",money(full.var_99),"annual aggregate"],["99% expected shortfall",money(full.expected_shortfall_99),"tail average"]]);
  table("segment-table",segments,[["RiskSegment","Segment"],["Policies","Policies",number],["PredictedAnnualFrequency","Frequency",number],["ExpectedSeverity","Severity",money],["AnnualPurePremium","Pure premium",money],["LargeLossProbability","Tail probability",pct],["ObservedLossCost","Observed loss cost",money]]);
}

async function loadBonus() {
  const [data,observed] = await Promise.all([api("/bonus-malus"),api("/bonus-malus/observed")]);
  const sample = observed.filter((_,index) => index % Math.max(1,Math.floor(observed.length/15)) === 0); bars("bonus-chart",sample,"BonusMalusBand","PurePremium",money);
  const frequency = data.frequency; document.getElementById("bonus-insight").textContent = `A 10-point increase corresponds to a ${number((frequency.relativity_per_10_points - 1) * 100)}% modeled frequency relativity. The relationship supports risk separation, while the held-out evidence should still be interpreted as predictive rather than causal.`;
}

document.getElementById("policy-form").addEventListener("submit", async event => {
  event.preventDefault(); const form = new FormData(event.currentTarget); const payload = Object.fromEntries(form.entries());
  ["exposure","vehicle_power","vehicle_age","driver_age","bonus_malus","density"].forEach(key => payload[key] = Number(payload[key]));
  const error = document.getElementById("score-error"); error.textContent = "";
  try { const r = await api("/policy/score",{method:"POST",body:JSON.stringify(payload)}); document.getElementById("empty-score").hidden=true; const target=document.getElementById("score-result"); target.hidden=false; target.innerHTML=`<div class="score-head"><div><p class="eyebrow">ClaimGuard score</p><div class="score-number">${number(r.claimguard_score)}</div></div><span class="pill">${r.overall_risk} RISK</span></div><h2>${r.risk_segment}</h2><div class="flow"><div><small>Frequency</small><strong>${number(r.expected_claims_per_year)}</strong></div><div><small>Severity</small><strong>${money(r.expected_claim_severity)}</strong></div><div><small>Annual loss</small><strong>${money(r.expected_annual_loss)}</strong></div><div><small>Tail risk</small><strong>${pct(r.large_loss_probability)}</strong></div></div><div class="risk-lines"><div class="risk-line"><span>Exposure-period loss</span><strong>${money(r.expected_loss_for_exposure)}</strong></div><div class="risk-line"><span>Tail percentile</span><strong>${number(r.tail_risk_percentile)}th</strong></div><div class="risk-line"><span>Frequency / severity</span><strong>${r.frequency_risk} / ${r.severity_risk}</strong></div></div><div class="drivers">${r.primary_risk_drivers.map(driver=>`<span>${driver}</span>`).join("")}</div>`; } catch (reason) { error.textContent=reason.message; }
});

async function boot() {
  try { await api("/health"); document.querySelector(".status").classList.add("online"); document.getElementById("api-status").textContent="Models online";
    const brands=["B1","B2","B3","B4","B5","B6","B10","B11","B12","B13","B14"];
    document.getElementById("vehicle-brand").innerHTML=brands.map(value=>`<option>${value}</option>`).join("");
    const regions=["R11","R21","R22","R23","R24","R25","R26","R31","R41","R42","R43","R52","R53","R54","R72","R73","R74","R82","R83","R91","R93","R94"];
    document.getElementById("region").innerHTML=regions.map(value=>`<option>${value}</option>`).join("");
    await Promise.all([loadOverview(),loadModels(),loadTail(),loadPortfolio(),loadBonus()]);
  } catch (reason) { document.getElementById("api-status").textContent="API unavailable"; console.error(reason); }
}
boot();
