/* SmishGuard smishing classifier — UI layer over the JSON API.
 *
 * Every value that originates from an SMS is written with textContent, never
 * innerHTML: the input to this tool is by definition attacker-authored text.
 */
"use strict";

const LABELS = {
  legit:                  { text: "Legitimate",           severity: "safe" },
  phishing_credential:    { text: "Phishing · credential", severity: "danger" },
  phishing_reversal_scam: { text: "Reversal scam",        severity: "danger" },
  fake_agent:             { text: "Fake agent",           severity: "danger" },
  prize_scam:             { text: "Prize scam",           severity: "danger" },
  account_takeover:       { text: "Account takeover",     severity: "danger" },
  other_fraud:            { text: "Other fraud",          severity: "danger" },
};

const $ = (id) => document.getElementById(id);
const meta = {
  low_confidence_threshold: 0.6,
  risk_fraud_threshold: 0.5,
  risk_legit_threshold: 0.4,
  risk_block_threshold: 0.8,
  max_batch_rows: 500,
};

const pct = (n) => `${(n * 100).toFixed(0)}%`;
const signed = (n) => `${n >= 0 ? "+" : "−"}${Math.abs(n).toFixed(2)}`;
const describe = (label) => LABELS[label] || { text: label, severity: "danger" };

/* ——— Tabs ——————————————————————————————————————————————— */
const tabs = [
  { tab: $("tab-single"), panel: $("panel-single") },
  { tab: $("tab-batch"),  panel: $("panel-batch") },
];

tabs.forEach(({ tab }, i) => {
  tab.addEventListener("click", () => selectTab(i));
});

function selectTab(active) {
  tabs.forEach(({ tab, panel }, i) => {
    tab.setAttribute("aria-selected", String(i === active));
    panel.hidden = i !== active;
  });
  clearError();
}

/* ——— Pending state — spinner is delay-shown so fast runs never flash it —— */
function setPending(button, pending) {
  if (!pending) {
    clearTimeout(button._spinTimer);
    button.disabled = false;
    button.textContent = button.dataset.idleLabel || button.textContent;
    return;
  }
  button.dataset.idleLabel = button.textContent;
  button.disabled = true;
  button._spinTimer = setTimeout(() => {
    button.replaceChildren(
      Object.assign(document.createElement("span"), { className: "spinner" }),
      document.createTextNode("Working…"),
    );
  }, 150);
}

function clearError() {
  $("form-error").replaceChildren();
}

function showError(message) {
  const p = document.createElement("p");
  p.className = "error-note";
  p.textContent = message;
  $("form-error").replaceChildren(p);
}

async function postJSON(url, options) {
  const resp = await fetch(url, options);
  if (!resp.ok) {
    let detail = `Request failed (${resp.status}).`;
    try {
      const body = await resp.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail) && body.detail[0]?.msg) detail = body.detail[0].msg;
    } catch { /* non-JSON error body — keep the status message */ }
    throw new Error(detail);
  }
  return resp.json();
}

function showState(id) {
  ["state-empty", "state-single", "state-batch"].forEach((s) => { $(s).hidden = s !== id; });
  const band = $("results");
  band.classList.remove("is-in");
  void band.offsetWidth;          // restart the reveal transition
  band.classList.add("reveal", "is-in");
}

/* ——— Single message ————————————————————————————————————— */
$("single-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = $("sms-text");
  const text = input.value.trim();
  clearError();

  if (!text) {
    input.setAttribute("aria-invalid", "true");
    input.focus();
    showError("Enter an SMS message to classify.");
    return;
  }
  input.removeAttribute("aria-invalid");

  const button = $("single-submit");
  setPending(button, true);
  try {
    renderSingle(await postJSON("/api/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    }));
  } catch (err) {
    showError(err.message);
  } finally {
    setPending(button, false);
  }
});

/* Risk (P of any fraud class) is the headline — a scam that splits its mass
 * across several fraud classes has low argmax confidence but high risk, and a
 * security tool must not read as unsure in that case. `risk` alone gates
 * this verdict; `confidence` (the class argmax) never does — see
 * renderUncertainty, which surfaces it as a separate, non-overriding note.
 *
 * Two bands above the legit line, not one: "block" (>=0.8) is reserved for
 * risk the model is genuinely confident about, so it doesn't read with the
 * same urgency as a message that only just crossed the 0.5 F1-optimal
 * cutoff into "warn". */
function riskVerdict(risk) {
  if (risk == null) return { text: "Unscored", severity: "warn" };
  if (risk >= meta.risk_block_threshold) return { text: "High-confidence fraud", severity: "danger" };
  if (risk >= meta.risk_fraud_threshold) return { text: "Likely fraud", severity: "danger" };
  if (risk <= meta.risk_legit_threshold) return { text: "Likely legitimate", severity: "safe" };
  return { text: "Inconclusive", severity: "warn" };
}

function renderSingle(result) {
  const verdict = riskVerdict(result.risk);

  $("verdict").dataset.severity = verdict.severity;
  $("verdict-risk").textContent = verdict.text;
  $("verdict-riskpct").textContent = result.risk == null ? "n/a" : pct(result.risk);
  $("verdict-label").textContent = describe(result.label).text;
  $("verdict-conf").textContent = result.confidence == null ? "" : `(${pct(result.confidence)})`;
  $("verdict-meta").textContent = `${result.risk_signals.length} signals checked`;

  renderUncertainty(result.risk, result.confidence);
  renderLlmOpinion(result.llm_opinion);
  renderSignals(result.risk_signals);
  renderTokens(result.top_tokens);
  setGauge(result.risk, verdict.severity);
  renderMeter(result.risk_signals);
  showState("state-single");

  // #results itself isn't a live region — re-rendering the full readout on
  // every classify would read the whole panel aloud. This is the one line SR
  // users actually hear.
  $("a11y-status").textContent =
    `${verdict.text}, ${result.risk == null ? "risk unscored" : pct(result.risk) + " risk"}.`;
}

function renderUncertainty(risk, confidence) {
  const host = $("uncertain-note");
  host.replaceChildren();
  if (risk == null) return;

  const riskIsAmbiguous = risk > meta.risk_legit_threshold && risk < meta.risk_fraud_threshold;

  // Fraud is settled, only the category is split — say so plainly rather than
  // firing an alarm that would undersell a confident detection.
  if (!riskIsAmbiguous && confidence != null && confidence < meta.low_confidence_threshold) {
    const note = document.createElement("p");
    note.className = "readout__note";
    note.style.marginBottom = "var(--space-lg)";
    note.textContent =
      `Risk is decisive at ${pct(risk)}, but the fraud category is split across several ` +
      "classes — read the label as the closest match, not a firm classification.";
    host.append(note);
    return;
  }
  if (!riskIsAmbiguous) return;

  const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  icon.setAttribute("width", "16");
  icon.setAttribute("height", "16");
  icon.setAttribute("viewBox", "0 0 16 16");
  icon.setAttribute("fill", "none");
  icon.setAttribute("stroke", "currentColor");
  icon.setAttribute("stroke-width", "1.5");
  icon.setAttribute("stroke-linecap", "round");
  icon.setAttribute("aria-hidden", "true");
  icon.innerHTML = '<path d="M8 1.9 1.2 13.6h13.6L8 1.9Z"/><path d="M8 6.2v3.4"/><path d="M8 11.6h.01"/>';

  const body = document.createElement("p");
  body.style.margin = "0";
  const lead = document.createElement("b");
  lead.textContent = "Inconclusive. ";
  body.append(
    lead,
    document.createTextNode(
      `Risk sits at ${pct(risk)} — the model cannot separate this from a legitimate message. ` +
      "Treat it as suspicious: do not click links or share your PIN, and verify directly " +
      "with the institution using a number or channel you already trust.",
    ),
  );

  const wrap = document.createElement("div");
  wrap.className = "uncertain";
  wrap.append(icon, body);
  host.replaceChildren(wrap);
}

/* Second opinion from an LLM, only present when the local model's risk was
 * inconclusive AND a Groq key is configured server-side. Rendered as a
 * clearly separate block — never merged into the primary verdict — because
 * it's not mechanistically explainable the way the weighted signals are. */
function renderLlmOpinion(opinion) {
  const host = $("llm-opinion");
  host.replaceChildren();
  if (!opinion) return;

  const wrap = document.createElement("div");
  wrap.className = "llm-opinion";
  wrap.dataset.verdict = opinion.verdict;

  const head = document.createElement("p");
  head.className = "llm-opinion__head";
  const badge = document.createElement("span");
  badge.className = "mono-label";
  badge.textContent = "AI second opinion";
  const verdict = document.createElement("b");
  verdict.textContent = `${opinion.verdict === "fraud" ? "Likely fraud" : "Likely legitimate"} `;
  const conf = document.createElement("span");
  conf.textContent = `(${pct(opinion.confidence)} confidence)`;
  head.append(badge, document.createTextNode(" "), verdict, conf);

  const reasoning = document.createElement("p");
  reasoning.className = "llm-opinion__reasoning";
  reasoning.textContent = opinion.reasoning;

  wrap.append(head, reasoning);
  host.append(wrap);
}

function renderSignals(signals) {
  const list = $("signals-list");
  list.replaceChildren();

  for (const signal of signals) {
    const row = document.createElement("li");
    row.className = "signal";
    row.dataset.present = String(signal.present);

    const mark = document.createElement("span");
    mark.className = "signal__mark";

    const label = document.createElement("span");
    label.className = "signal__label";
    label.textContent = signal.label;

    const weight = document.createElement("span");
    weight.className = "signal__weight";
    weight.textContent = signed(signal.weight);

    const state = document.createElement("span");
    state.className = "visually-hidden";
    state.textContent = signal.present ? " — fired" : " — not found";
    label.append(state);

    row.append(mark, label, weight);

    if (signal.note) {
      const note = document.createElement("span");
      note.className = "signal__note";
      note.textContent = signal.note;
      row.append(note);
    }
    list.append(row);
  }
}

function renderTokens(tokens) {
  const list = $("tokens-list");
  list.replaceChildren();

  if (!tokens.length) {
    const empty = document.createElement("li");
    empty.className = "token-row";
    empty.textContent = "No non-zero feature contributions.";
    list.append(empty);
    return;
  }

  for (const token of tokens) {
    const row = document.createElement("li");
    row.className = "token-row";

    const name = document.createElement("span");
    name.className = "token-row__name";
    name.textContent = token.token;

    const value = document.createElement("span");
    value.className = "token-row__val";
    value.dataset.sign = token.contribution >= 0 ? "pos" : "neg";
    value.textContent = signed(token.contribution);

    row.append(name, value);
    list.append(row);
  }
}

/* ——— CSV batch ——————————————————————————————————————————— */
$("batch-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const picker = $("csv-file");
  clearError();

  if (!picker.files.length) {
    showError("Choose a CSV file to classify.");
    picker.focus();
    return;
  }

  const payload = new FormData();
  payload.append("file", picker.files[0]);

  const button = $("batch-submit");
  setPending(button, true);
  try {
    renderBatch(await postJSON("/api/classify/batch", { method: "POST", body: payload }));
  } catch (err) {
    showError(err.message);
  } finally {
    setPending(button, false);
  }
});

function renderBatch(rows) {
  const body = $("batch-rows");
  body.replaceChildren();

  for (const row of rows) {
    const verdict = riskVerdict(row.risk);
    const tr = document.createElement("tr");

    const message = document.createElement("td");
    message.className = "cell-text";
    message.textContent = row.text;

    const risk = document.createElement("td");
    risk.dataset.severity = verdict.severity;
    risk.textContent = verdict.text;

    const riskPct = document.createElement("td");
    riskPct.className = "num";
    riskPct.textContent = row.risk == null ? "—" : pct(row.risk);

    const label = document.createElement("td");
    label.className = "cell-muted";
    label.textContent = describe(row.label).text;

    tr.append(message, risk, riskPct, label);
    body.append(tr);
  }

  const flagged = rows.filter((r) => r.risk != null && r.risk >= meta.risk_fraud_threshold).length;
  $("batch-meta").textContent = `${rows.length} messages · ${flagged} at or above ${pct(meta.risk_fraud_threshold)} risk`;
  setGauge(null, null);
  $("meter-state").textContent = `${rows.length} rows · ${flagged} flagged`;
  showState("state-batch");

  $("a11y-status").textContent = `${rows.length} messages classified, ${flagged} flagged.`;
}

/* ——— Boot ———————————————————————————————————————————————— */
fetch("/api/meta")
  .then((r) => (r.ok ? r.json() : null))
  .then((data) => {
    if (!data) return;
    Object.assign(meta, data);
    $("max-rows").textContent = String(data.max_batch_rows);
    $("llm-disclosure").hidden = !data.llm_review_available;
  })
  .catch(() => { /* defaults already cover it */ });

/* ——— Apparatus: the risk dial ——————————————————————————————
 * The dial is the readout, not decoration: --risk drives the sweep, the
 * needle and the halo, and severity recolours all three. Geometry is a 270°
 * arc centred at (100,100), r=90, so value v sits at -135° + v·270°.
 */
const DIAL_SWEEP_DEG = 270;
const DIAL_START_DEG = -135;
const DIAL_ARC_LEN = 424.115;          // 2*pi*90 * (270/360), matches the path in index.html

function dialPoint(value, radius) {
  const rad = ((DIAL_START_DEG + value * DIAL_SWEEP_DEG) * Math.PI) / 180;
  return [100 + radius * Math.sin(rad), 100 - radius * Math.cos(rad)];
}

(function buildDialTicks() {
  const host = $("dial-ticks");
  if (!host) return;
  for (let i = 0; i <= 10; i++) {
    const v = i / 10;
    const major = v === 0 || v === 1;
    const threshold = v === 0.5;                 // the F1-optimal fraud line
    const [x1, y1] = dialPoint(v, major || threshold ? 76 : 80);
    const [x2, y2] = dialPoint(v, 88);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", x1.toFixed(2));
    line.setAttribute("y1", y1.toFixed(2));
    line.setAttribute("x2", x2.toFixed(2));
    line.setAttribute("y2", y2.toFixed(2));
    if (major) line.classList.add("is-major");
    if (threshold) line.classList.add("is-threshold");
    host.append(line);
  }
})();

/* The needle angle, arc length and halo are written as resolved values rather
 * than driven from a --risk custom property in calc(): an unregistered custom
 * property inside calc() does not reliably invalidate, and registering it via
 * @property did not fix it here either. Concrete values transition correctly
 * and have no browser caveat. */
function setGauge(risk, severity) {
  const dial = $("dial");
  if (!dial) return;
  const v = risk == null ? 0 : risk;

  $("dial-needle").style.transform =
    `translateY(-100%) rotate(${(DIAL_START_DEG + v * DIAL_SWEEP_DEG).toFixed(2)}deg)`;
  $("dial-sweep").style.strokeDashoffset = (DIAL_ARC_LEN * (1 - v)).toFixed(2);
  dial.querySelector(".dial__glow").style.opacity = (0.25 + v * 0.75).toFixed(3);

  if (severity) dial.dataset.severity = severity;
  else delete dial.dataset.severity;
  $("dial-value").textContent = risk == null ? "—" : pct(risk);
}

/* ——— Meter strip: one tick per engineered signal ————————————
 * At rest the strip is flat and labelled "awaiting input" — an invented
 * envelope here would be a decorative readout, which is the thing this
 * whole page argues against. Heights come from |weight| once a real
 * classification lands.
 */
const METER_SIGNALS = 20;

(function buildMeter() {
  const host = $("meter-bars");
  if (!host) return;
  for (let i = 0; i < METER_SIGNALS; i++) {
    host.append(Object.assign(document.createElement("span"), { style: "--h: 12%" }));
  }
})();

function renderMeter(signals) {
  const host = $("meter-bars");
  if (!host) return;
  const peak = Math.max(...signals.map((s) => Math.abs(s.weight)), 0.01);
  const bars = host.children;

  // The strip is a fixed 20-tick instrument; a model with a different feature
  // count still renders, it just fills fewer ticks.
  for (let i = 0; i < bars.length; i++) {
    const signal = signals[i];
    if (!signal) {
      bars[i].style.setProperty("--h", "12%");
      delete bars[i].dataset.fired;
      continue;
    }
    const height = 12 + (Math.abs(signal.weight) / peak) * 88;
    bars[i].style.setProperty("--h", `${height.toFixed(1)}%`);
    bars[i].dataset.fired = String(signal.present);
  }
  const fired = signals.filter((s) => s.present).length;
  $("meter-state").textContent = `fired · ${fired} / ${signals.length}`;
}

/* ——— Palette drop ————————————————————————————————————————
 * Day Foundry is the default at every entry point; night is opt-in and
 * persisted. The OS preference deliberately does not decide - the page is
 * specified light-first, so an unset choice always means day.
 */
(function dropToggle() {
  const button = $("drop-toggle");
  if (!button) return;

  const current = () => document.documentElement.dataset.drop || "day";

  function sync() {
    const now = current();
    const next = now === "night" ? "day" : "night";
    button.setAttribute("aria-pressed", String(now === "night"));
    button.setAttribute("aria-label", `Switch to ${next} foundry`);
    $("drop-toggle-text").textContent = next;
  }

  button.addEventListener("click", () => {
    const next = current() === "night" ? "day" : "night";
    document.documentElement.dataset.drop = next;
    try { localStorage.setItem("sg-drop", next); } catch { /* private mode — session only */ }
    sync();
  });

  sync();
})();
