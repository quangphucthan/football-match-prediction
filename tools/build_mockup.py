"""Builds the clickable mockup, injecting real model output so no number is invented."""
import base64, json, pathlib

HERE = pathlib.Path(__file__).parent
data = json.loads((HERE / "mock.json").read_text())


def b64(name):
    return base64.b64encode((HERE / "fonts" / name).read_bytes()).decode()

HTML = """<title>Match Predictor — interface prototype</title>
<style>
  /* Fonts are inlined as data URIs, not <link>ed. The Artifact CSP blocks every
     external host, so a Google Fonts or Font Awesome CDN link would silently
     fall back to system faces. Roboto and Roboto Mono are variable fonts, so one
     file each covers every weight. Font Awesome is subsetted to the four glyphs
     actually used here: 204 KB of full solid down to 1.4 KB. */
  @font-face {
    font-family: 'Roboto'; font-style: normal; font-weight: 100 900;
    font-display: swap; src: url(data:font/woff2;base64,__ROBOTO__) format('woff2');
  }
  @font-face {
    font-family: 'Roboto Mono'; font-style: normal; font-weight: 100 700;
    font-display: swap; src: url(data:font/woff2;base64,__ROBOTO_MONO__) format('woff2');
  }
  @font-face {
    font-family: 'Font Awesome 6 Free'; font-style: normal; font-weight: 900;
    font-display: block; src: url(data:font/woff2;base64,__FA__) format('woff2');
  }
  .fa { font-family: 'Font Awesome 6 Free'; font-weight: 900; font-style: normal;
        line-height: 1; -webkit-font-smoothing: antialiased; display: inline-block; }

  :root {
    color-scheme: light dark;
    --font-sans: 'Roboto', system-ui, -apple-system, sans-serif;
    --font-mono: 'Roboto Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
    --bg:        #F1F3EF;
    --panel:     #FFFFFF;
    --inset:     #E7EBE5;
    --rule:      #D2D9CE;
    --ink:       #101A16;
    --ink-dim:   #4A574F;
    --ink-faint: #7C8981;
    --accent:    #9A6B10;
    --accent-bg: #F6E6C4;
    /* Home/away are fixed semantic tokens, NOT the teams' flag colours. Flag
       colours in the dataset are mostly #FFFFFF / #000000 / red, which are
       neither legible on both grounds nor distinguishable from each other.
       The real colour survives as the identity swatch beside each name. */
    --home:      #A8690E;
    --away:      #2E6B8C;
    --shadow:    0 1px 2px rgba(16,26,22,.06), 0 8px 24px -12px rgba(16,26,22,.18);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg:        #0E1714;
      --panel:     #16211D;
      --inset:     #1D2A25;
      --rule:      #2A3A33;
      --ink:       #E8EFEA;
      --ink-dim:   #9DAEA5;
      --ink-faint: #6C7F76;
      --accent:    #E0A33F;
      --accent-bg: #2E2415;
      --home:      #E0A33F;
      --away:      #74B4D4;
      --shadow:    0 1px 2px rgba(0,0,0,.4), 0 8px 24px -12px rgba(0,0,0,.6);
    }
  }
  :root[data-theme="light"] {
    --bg:#F1F3EF; --panel:#FFFFFF; --inset:#E7EBE5; --rule:#D2D9CE;
    --ink:#101A16; --ink-dim:#4A574F; --ink-faint:#7C8981;
    --accent:#9A6B10; --accent-bg:#F6E6C4; --home:#A8690E; --away:#2E6B8C;
    --shadow:0 1px 2px rgba(16,26,22,.06), 0 8px 24px -12px rgba(16,26,22,.18);
  }
  :root[data-theme="dark"] {
    --bg:#0E1714; --panel:#16211D; --inset:#1D2A25; --rule:#2A3A33;
    --ink:#E8EFEA; --ink-dim:#9DAEA5; --ink-faint:#6C7F76;
    --accent:#E0A33F; --accent-bg:#2E2415; --home:#E0A33F; --away:#74B4D4;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -12px rgba(0,0,0,.6);
  }

  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font-family: var(--font-sans);
    font-size: 15px; line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }
  .mono { font-family: var(--font-mono); }
  .num  { font-variant-numeric: tabular-nums; font-family: var(--font-mono); }

  .wrap { max-width: 1080px; margin: 0 auto; padding: 32px 20px 72px; }

  /* ---- masthead ---- */
  .masthead { display:flex; align-items:baseline; justify-content:space-between; gap:16px; flex-wrap:wrap;
              border-bottom:1px solid var(--rule); padding-bottom:14px; margin-bottom:24px; }
  .masthead h1 { font-size:19px; font-weight:640; letter-spacing:-.02em; margin:0; }
  .masthead h1 span { color: var(--ink-faint); font-weight:440; }
  .provenance { font-size:11.5px; letter-spacing:.07em; text-transform:uppercase; color:var(--ink-faint); }

  /* ---- fixture bar ---- */
  .fixture { background:var(--panel); border:1px solid var(--rule); border-radius:4px;
             box-shadow:var(--shadow); padding:18px; margin-bottom:10px; }
  .picks { display:grid; grid-template-columns:1fr auto 1fr; gap:14px; align-items:end; }
  .field { display:flex; flex-direction:column; gap:6px; min-width:0; }
  label.cap { font-size:10.5px; letter-spacing:.12em; text-transform:uppercase; color:var(--ink-faint);
              font-family:var(--font-mono); }
  .inputrow { display:flex; align-items:center; gap:9px; background:var(--inset);
              border:1px solid var(--rule); border-radius:3px; padding:0 10px; }
  .inputrow:focus-within { border-color:var(--accent); box-shadow:0 0 0 3px var(--accent-bg); }
  /* Bordered so pure-white and pure-black flag colours still read as a swatch. */
  .swatch { width:11px; height:18px; border-radius:1px; flex:none; background:var(--ink-faint);
            border:1px solid var(--rule); transition:background .18s ease; }
  .inputrow input { flex:1; min-width:0; background:none; border:none; outline:none; color:var(--ink);
                    font:inherit; font-size:15px; padding:9px 0; }
  .masthead h1 .mark { color:var(--accent); font-size:16px; margin-right:2px; }
  .opt-icon { color:var(--ink-faint); font-size:12px; }

  /* Replaces a dead "V" label with a control that does something. */
  .swap { background:var(--inset); border:1px solid var(--rule); border-radius:3px; color:var(--ink-dim);
          cursor:pointer; padding:0; width:34px; height:36px; margin-bottom:1px;
          display:grid; place-items:center; font-size:12px; transition:all .15s ease; }
  .swap:hover { border-color:var(--accent); color:var(--ink); }
  .swap:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
  .swap:active { transform:scale(.94); }

  .opts { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:14px;
          padding-top:14px; border-top:1px solid var(--rule); }
  select { background:var(--inset); border:1px solid var(--rule); border-radius:3px; color:var(--ink);
           font:inherit; font-size:13.5px; padding:7px 9px; }
  select:focus-visible { outline:2px solid var(--accent); outline-offset:1px; }
  .toggle { display:flex; align-items:center; gap:7px; font-size:13.5px; color:var(--ink-dim); cursor:pointer; }
  .toggle input { accent-color:var(--accent); width:15px; height:15px; }
  .spacer { flex:1; }
  .hint { font-size:12px; color:var(--ink-faint); }

  /* ---- fixture shortcuts ---- */
  .shortcuts { display:flex; gap:7px; flex-wrap:wrap; margin-bottom:26px; }
  .chip { background:var(--panel); border:1px solid var(--rule); border-radius:99px; color:var(--ink-dim);
          font-size:12.5px; padding:5px 12px; cursor:pointer; font-family:inherit; transition:all .15s ease; }
  .chip:hover { border-color:var(--accent); color:var(--ink); }
  .chip[aria-pressed="true"] { background:var(--accent-bg); border-color:var(--accent); color:var(--ink); }
  .chip:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }

  /* ---- verdict ---- */
  .verdict { background:var(--panel); border:1px solid var(--rule); border-radius:4px;
             box-shadow:var(--shadow); padding:22px; margin-bottom:14px; }
  .odds { display:grid; grid-template-columns:1fr 1fr 1fr; gap:2px; margin-bottom:14px; }
  .odds div { text-align:center; }
  .odds .pct { font-size:34px; font-weight:600; letter-spacing:-.03em; line-height:1.05;
               font-variant-numeric:tabular-nums; font-family:var(--font-mono); }
  .odds .who { font-size:11px; letter-spacing:.11em; text-transform:uppercase; color:var(--ink-faint);
               margin-top:3px; font-family:var(--font-mono); }
  .odds .home .pct { color:var(--home); }
  .odds .away .pct { color:var(--away); }

  .bar { display:flex; height:9px; border-radius:2px; overflow:hidden; background:var(--inset); }
  .bar i { display:block; transition:width .5s cubic-bezier(.4,0,.2,1); }
  .bar .h { background:var(--home); }
  .bar .d { background:var(--ink-faint); }
  .bar .a { background:var(--away); }

  .xg { display:flex; align-items:baseline; justify-content:center; gap:12px; margin-top:16px;
        padding-top:15px; border-top:1px solid var(--rule); }
  .xg b { font-size:22px; font-weight:600; font-variant-numeric:tabular-nums;
          font-family:var(--font-mono); }
  .xg .lab { font-size:11px; letter-spacing:.11em; text-transform:uppercase; color:var(--ink-faint);
             font-family:var(--font-mono); }

  /* ---- panels ---- */
  /* start, not stretch: panels size to their content rather than padding out to
     match the tallest neighbour, which left dead space under the short ones. */
  .panels { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; align-items:start; }
  .panel { background:var(--panel); border:1px solid var(--rule); border-radius:4px;
           box-shadow:var(--shadow); padding:18px; min-width:0; }
  .p-matrix  { grid-column:span 7; }
  .p-scores  { grid-column:span 5; }
  .p-markets { grid-column:span 5; }
  .p-history { grid-column:span 7; }
  @media (max-width:820px) { .panel { grid-column:1 / -1 !important; } }

  h2.ph { font-size:11px; letter-spacing:.12em; text-transform:uppercase; color:var(--ink-faint);
          margin:0 0 14px; font-weight:500; font-family:var(--font-mono); }

  /* ---- matrix ---- */
  .matrixscroll { overflow-x:auto; }
  .matrix { display:grid; gap:2px; font-variant-numeric:tabular-nums; }
  .matrix .cell { aspect-ratio:1; border-radius:2px; display:grid; place-items:center;
                  font-size:9.5px; color:var(--ink-dim); position:relative;
                  font-family:var(--font-mono); }
  .matrix .cell.peak { outline:1.5px solid var(--accent); outline-offset:-1.5px; color:var(--ink); font-weight:600; }
  .matrix .ax { font-size:10px; color:var(--ink-faint); display:grid; place-items:center;
                font-family:var(--font-mono); }
  .axtitle { font-size:10px; letter-spacing:.09em; text-transform:uppercase; color:var(--ink-faint);
             font-family:var(--font-mono); }
  .matrixlegend { display:flex; justify-content:space-between; align-items:center; margin-top:11px;
                  font-size:11px; color:var(--ink-faint); }
  .key { display:flex; align-items:center; gap:6px; }
  .key i { width:9px; height:9px; border-radius:1px; display:block; }

  /* ---- scorelines ---- */
  .sl { display:flex; flex-direction:column; gap:8px; }
  .sl .row { display:grid; grid-template-columns:46px 1fr 46px; align-items:center; gap:10px; }
  .sl .s { font-size:14px; font-weight:600; font-variant-numeric:tabular-nums;
           font-family:var(--font-mono); }
  .sl .track { height:6px; background:var(--inset); border-radius:2px; overflow:hidden; }
  .sl .track i { display:block; height:100%; transition:width .5s cubic-bezier(.4,0,.2,1); }
  .sl .p { font-size:12.5px; color:var(--ink-dim); text-align:right; font-variant-numeric:tabular-nums;
           font-family:var(--font-mono); }

  /* ---- markets ---- */
  .mk { display:flex; flex-direction:column; gap:11px; }
  .mk .row { display:grid; grid-template-columns:1fr 62px 40px; align-items:center; gap:10px; font-size:13.5px; }
  .mk .track { height:5px; background:var(--inset); border-radius:2px; overflow:hidden; }
  .mk .track i { display:block; height:100%; background:var(--ink-dim); transition:width .5s cubic-bezier(.4,0,.2,1); }
  .mk .v { text-align:right; font-size:12.5px; color:var(--ink-dim); font-variant-numeric:tabular-nums;
           font-family:var(--font-mono); }

  /* ---- history ---- */
  .hsplit { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
  @media (max-width:560px) { .hsplit { grid-template-columns:1fr; } }
  .h2hbar { display:flex; height:7px; border-radius:2px; overflow:hidden; background:var(--inset); margin:4px 0 9px; }
  .h2hcount { font-size:12.5px; color:var(--ink-dim); display:flex; justify-content:space-between; }
  .played { font-size:12px; color:var(--ink-faint); margin-bottom:10px; }
  .recent { display:flex; flex-direction:column; gap:5px; font-size:12.5px; }
  .recent .r { display:grid; grid-template-columns:auto 1fr auto; gap:10px; color:var(--ink-dim);
               font-variant-numeric:tabular-nums; }
  .recent .r > span:first-child { white-space:nowrap; font-family:var(--font-mono);
                                  font-size:11.5px; color:var(--ink-faint); }
  .recent .r .sc { font-weight:600; color:var(--ink); font-family:var(--font-mono); }
  .formrow { display:flex; align-items:center; gap:9px; margin-bottom:9px; }
  .formrow .nm { font-size:12.5px; color:var(--ink-dim); flex:1; }
  .dots { display:flex; gap:3px; }
  .dot { width:19px; height:19px; border-radius:2px; display:grid; place-items:center; font-size:10px;
         font-weight:600; color:var(--bg); font-family:var(--font-mono); }
  .dot.W { background:#3E8E5A; } .dot.D { background:var(--ink-faint); } .dot.L { background:#A8503F; }

  .empty { background:var(--panel); border:1px dashed var(--rule); border-radius:4px; padding:34px 20px;
           text-align:center; color:var(--ink-faint); font-size:13.5px; }

  .footnote { margin-top:26px; padding-top:16px; border-top:1px solid var(--rule);
              font-size:12px; color:var(--ink-faint); line-height:1.6; }
  .footnote strong { color:var(--ink-dim); font-weight:560; }

  @media (prefers-reduced-motion: reduce) { * { transition:none !important; } }
</style>

<div class="wrap">
  <div class="masthead">
    <h1><i class="fa mark" aria-hidden="true">&#xf1e3;</i> Match Predictor <span>— interface prototype</span></h1>
    <div class="provenance">25,328 internationals · 2000–2026</div>
  </div>

  <div class="fixture">
    <div class="picks">
      <div class="field">
        <label class="cap" for="home">Home</label>
        <div class="inputrow"><i class="swatch" id="hsw"></i><input id="home" list="teams" autocomplete="off"></div>
      </div>
      <button class="swap" id="swap" title="Swap home and away" aria-label="Swap home and away">
        <i class="fa" aria-hidden="true">&#xf362;</i>
      </button>
      <div class="field">
        <label class="cap" for="away">Away</label>
        <div class="inputrow"><i class="swatch" id="asw"></i><input id="away" list="teams" autocomplete="off"></div>
      </div>
    </div>
    <datalist id="teams"></datalist>
    <div class="opts">
      <i class="fa opt-icon" aria-hidden="true">&#xf091;</i>
      <select id="trn" aria-label="Competition"></select>
      <label class="toggle">
        <input type="checkbox" id="neu">
        <i class="fa opt-icon" aria-hidden="true">&#xf3c5;</i> Neutral venue
      </label>
      <span class="spacer"></span>
      <span class="hint" id="hint"></span>
    </div>
  </div>

  <div class="shortcuts" id="chips"></div>
  <div id="out"></div>

  <div class="footnote">
    <strong>Prototype.</strong> Every number below is real output from the fitted model, not sample data —
    but only the four fixtures above are precomputed here. The live app predicts any of 237 teams.
    Cards, possession and shot data are absent from the dataset, so nothing here shows them.
  </div>
</div>

<script>
const DATA = __DATA__;
const T = Object.fromEntries(DATA.teams.map(t => [t.name, t]));
// Chips show the four forward fixtures; predictions also hold each reverse so the
// swap button lands on real output rather than the empty state.
const FIXTURES = DATA.fixtures;
let cur = FIXTURES[0];

const pct = v => (v * 100).toFixed(1);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

document.getElementById("teams").innerHTML =
  DATA.teams.map(t => `<option value="${esc(t.name)}">`).join("");
document.getElementById("trn").innerHTML =
  DATA.tournaments.map(t => `<option>${esc(t)}</option>`).join("");
document.getElementById("chips").innerHTML =
  FIXTURES.map(([h, a], i) => `<button class="chip" data-i="${i}">${esc(h)} v ${esc(a)}</button>`).join("");

// Only for the identity swatch. Data encoding uses --home / --away.
function teamColour(name) { return (T[name] && T[name].color) || "#7C8981"; }

function matrix(grid) {
  // Window the grid to where the probability actually is: lopsided fixtures sit
  // well outside 0-5 (Brazil v San Marino peaks at 6-0). Grow the square until it
  // holds 95% of the mass -- a peak-plus-two rule left a fifth of it off-screen.
  let pi = 0, pj = 0, peak = 0;
  grid.forEach((row, i) => row.forEach((p, j) => { if (p > peak) { peak = p; pi = i; pj = j; } }));
  const covered = k => {
    let s = 0;
    for (let i = 0; i <= k; i++) for (let j = 0; j <= k; j++) s += grid[i][j];
    return s;
  };
  let n = Math.max(5, Math.max(pi, pj));
  while (n < 10 && covered(n) < 0.95) n++;
  const cells = [`<div class="ax"></div>`];
  for (let j = 0; j <= n; j++) cells.push(`<div class="ax">${j}</div>`);
  for (let i = 0; i <= n; i++) {
    cells.push(`<div class="ax">${i}</div>`);
    for (let j = 0; j <= n; j++) {
      const p = grid[i][j];
      const tint = i > j ? "var(--home)" : i < j ? "var(--away)" : "var(--ink-faint)";
      const a = Math.min(1, Math.pow(p / peak, 0.55));
      const isPeak = i === pi && j === pj;
      cells.push(
        `<div class="cell${isPeak ? " peak" : ""}" style="background:color-mix(in srgb, ${tint} ${(a * 82).toFixed(1)}%, var(--inset))" title="${i}-${j} · ${pct(p)}%">` +
        (p >= 0.02 ? pct(p) : "") + `</div>`
      );
    }
  }
  // Cap the cell size: left to 1fr the cells ballooned to ~110px and the panel
  // towered over everything beside it.
  return `<div class="matrixscroll"><div class="matrix" style="grid-template-columns:20px repeat(${n + 1},minmax(26px,40px));justify-content:start">${cells.join("")}</div></div>`;
}

function render(home, away) {
  const p = DATA.predictions[home + "|" + away];
  const out = document.getElementById("out");
  document.getElementById("hsw").style.background = teamColour(home);
  document.getElementById("asw").style.background = teamColour(away);

  if (!p) {
    out.innerHTML = `<div class="empty">No precomputed numbers for <strong>${esc(home)} v ${esc(away)}</strong>.<br>Pick one of the four fixtures above.</div>`;
    return;
  }

  const o = p.outcome, m = p.markets, h = p.h2h;
  const topP = p.scorelines[0].p;
  const mkRow = (lab, v) =>
    `<div class="row"><span>${lab}</span><span class="track"><i style="width:${v * 100}%"></i></span><span class="v">${pct(v)}%</span></div>`;
  const dots = f => `<span class="dots">${f.map(r => `<i class="dot ${r}">${r}</i>`).join("")}</span>`;

  out.innerHTML = `
  <div class="verdict">
    <div class="odds">
      <div class="home"><div class="pct">${pct(o.home)}%</div><div class="who">${esc(home)}</div></div>
      <div><div class="pct">${pct(o.draw)}%</div><div class="who">Draw</div></div>
      <div class="away"><div class="pct">${pct(o.away)}%</div><div class="who">${esc(away)}</div></div>
    </div>
    <div class="bar">
      <i class="h" style="width:${o.home * 100}%"></i>
      <i class="d" style="width:${o.draw * 100}%"></i>
      <i class="a" style="width:${o.away * 100}%"></i>
    </div>
    <div class="xg">
      <span class="lab">Expected goals</span>
      <b style="color:var(--home)">${p.expected_goals.home.toFixed(2)}</b>
      <span class="lab">—</span>
      <b style="color:var(--away)">${p.expected_goals.away.toFixed(2)}</b>
    </div>
  </div>

  <div class="panels">
    <div class="panel p-matrix">
      <h2 class="ph">Scoreline probability</h2>
      ${matrix(p.grid)}
      <div class="matrixlegend">
        <span class="axtitle">↓ ${esc(home)} &nbsp;·&nbsp; → ${esc(away)}</span>
        <span class="key">
          <i style="background:var(--home)"></i> home
          <i style="background:var(--ink-faint)"></i> draw
          <i style="background:var(--away)"></i> away
        </span>
      </div>
    </div>

    <div class="panel p-scores">
      <h2 class="ph">Most likely scores</h2>
      <div class="sl">
        ${p.scorelines.map(s => {
          const [i, j] = s.score.split("-").map(Number);
          const c = i > j ? "var(--home)" : i < j ? "var(--away)" : "var(--ink-faint)";
          return `<div class="row"><span class="s">${s.score}</span>
            <span class="track"><i style="width:${(s.p / topP) * 100}%;background:${c}"></i></span>
            <span class="p">${pct(s.p)}%</span></div>`;
        }).join("")}
      </div>
    </div>

    <div class="panel p-markets">
      <h2 class="ph">Goal markets</h2>
      <div class="mk">
        ${mkRow("Over 1.5 goals", m.over_1_5)}
        ${mkRow("Over 2.5 goals", m.over_2_5)}
        ${mkRow("Over 3.5 goals", m.over_3_5)}
        ${mkRow("Both teams score", m.btts)}
        ${mkRow(esc(home) + " clean sheet", m.home_clean_sheet)}
        ${mkRow(esc(away) + " clean sheet", m.away_clean_sheet)}
      </div>
    </div>

    <div class="panel p-history">
      <h2 class="ph">History</h2>
      <div class="hsplit">
        <div>
          <div class="played">Head to head · ${h.played} meeting${h.played === 1 ? "" : "s"} since 2000</div>
          ${h.played ? `
            <div class="h2hbar">
              <i style="width:${(h.home_wins / h.played) * 100}%;background:var(--home)"></i>
              <i style="width:${(h.draws / h.played) * 100}%;background:var(--ink-faint)"></i>
              <i style="width:${(h.away_wins / h.played) * 100}%;background:var(--away)"></i>
            </div>
            <div class="h2hcount">
              <span>${esc(home)} ${h.home_wins}</span>
              <span>${h.draws} drawn</span>
              <span>${esc(away)} ${h.away_wins}</span>
            </div>
            <div class="recent" style="margin-top:12px">
              ${h.recent.map(r => `<div class="r"><span>${r.date}</span><span>${esc(r.home)} v ${esc(r.away)}</span><span class="sc">${r.score}</span></div>`).join("")}
            </div>` : `<div class="hint">Never played since 2000.</div>`}
        </div>
        <div>
          <div class="played">Recent form · newest first</div>
          <div class="formrow"><span class="nm">${esc(home)}</span>${dots(p.form.home)}</div>
          <div class="formrow"><span class="nm">${esc(away)}</span>${dots(p.form.away)}</div>
        </div>
      </div>
    </div>
  </div>`;
}

function select(home, away) {
  cur = [home, away];
  document.getElementById("home").value = home;
  document.getElementById("away").value = away;
  document.querySelectorAll(".chip").forEach((c, i) =>
    c.setAttribute("aria-pressed", FIXTURES[i][0] === home && FIXTURES[i][1] === away));
  const known = !!DATA.predictions[home + "|" + away];
  document.getElementById("hint").textContent = known
    ? "Live model output" : "Not precomputed in this prototype";
  render(home, away);
}

document.getElementById("swap").addEventListener("click", () => select(cur[1], cur[0]));

document.getElementById("chips").addEventListener("click", e => {
  const b = e.target.closest(".chip");
  if (b) select(...FIXTURES[+b.dataset.i]);
});
["home", "away"].forEach(id =>
  document.getElementById(id).addEventListener("change", () =>
    select(document.getElementById("home").value, document.getElementById("away").value)));

// Venue and competition are wired in the real app; the prototype's four fixtures
// are precomputed at the default settings, so changing them here only annotates.
["neu", "trn"].forEach(id => document.getElementById(id).addEventListener("change", () => {
  document.getElementById("hint").textContent = "Prototype: re-predicts in the live app";
}));

select(...cur);
</script>
"""

out = (HTML
       .replace("__ROBOTO__", b64("roboto.woff2"))
       .replace("__ROBOTO_MONO__", b64("robotomono.woff2"))
       .replace("__FA__", b64("fa-subset.woff2"))
       .replace("__DATA__", json.dumps(data, separators=(",", ":"))))
assert "__" not in out.split("<script>")[0], "unreplaced placeholder in markup"
(HERE / "mockup.html").write_text(out)
print(f"wrote mockup.html ({len(out):,} bytes)")
