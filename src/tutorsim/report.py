"""Per-set aggregate rate math for tutorsim benchmark results.

Verbatim port of _archive/benchmark/core/score.py: score_scenarios + all helpers.

Lucy's three-axis scoring (proposed 2026-06-15):

  scaffolding_did_rate    yes_count / (count where gold says scaffolding-appropriate). Higher better.
  rigor_did_rate          yes_count / (count where gold says rigor-appropriate).      Higher better.
  overscaffold_rate       count where any overscaffold_decomposed facet was emitted / total. Lower better.

The collapsed action label decomposes via _ACTION_LABEL_TO_DIMENSIONS:
  both        -> scaffolding=yes, rigor=yes
  scaffolding -> scaffolding=yes, rigor=no
  rigor       -> scaffolding=no,  rigor=yes
  neither     -> scaffolding=no,  rigor=no

Field mapping from archive (dict-of-dicts) to new API (Scenario + Annotation objects):
  archive: scenario["detection"]["situation_label_agg"]
  new:     scenario.dimension  (= scenario.rubric["gold"], set at build time)

  archive: _action_label_for_scenario(ann) -> ann["annotations"][i]["action_label"]
  new:     annotation.action_label  (first and only Annotation per scenario)

  archive: _has_pos_result(ann) -> ann["annotations"][i]["result_label"] in {"pos"}
  new:     annotation.result_label  (str "pos" | "neg" | "unclear" | "no_evidence")

  archive: _has_overscaffold(ann) -> ann["annotations"][i]["overscaffold_decomposed"] non-empty
  new:     annotation.overscaffold_decomposed  (non-empty list means over-scaffold detected)

  archive: "overscaffold_decomposed" key presence in ANY annotation
  new:     always present on Annotation dataclass (field always exists); treat as always available.

No module-level SDK imports.
"""
# ---------------------------------------------------------------------------
# Verbatim port of _archive/benchmark/core/score.py constants + helpers
# ---------------------------------------------------------------------------

_ACTION_LABEL_TO_DIMENSIONS = {
    "both":        ("yes", "yes"),
    "scaffolding": ("yes", "no"),
    "rigor":       ("no",  "yes"),
    "neither":     ("no",  "no"),
}


def _to_dims(label):
    return _ACTION_LABEL_TO_DIMENSIONS.get(label)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def aggregate(scenarios: list, annotations: list) -> dict:
    """Compute Lucy's three-axis scoring + outcome+ rate.

    Verbatim port of score_scenarios from _archive/benchmark/core/score.py.

    Args:
        scenarios: list of tutorsim.scenarios.Scenario objects (canonical order).
        annotations: list of tutorsim.scoring.Annotation objects, parallel to scenarios.

    Returns:
        {
          "n_scenarios": int,
          "scaffolding_did": {"n_yes": int, "n_total": int, "rate": float|None},
          "rigor_did":       {"n_yes": int, "n_total": int, "rate": float|None},
          "overscaffold":    {"n_yes": int, "n_total": int, "rate": float|None, "available": bool},
          "outcome_pos_rate": float,
          "scaffold_calibrated": {"n_clean_yes": int, "n_overscaffold": int, "n_total": int, "score": float|None},
          "rigor_calibrated":    {"n_clean_yes": int, "n_total": int, "score": float|None},
        }

    scaffolding_did.rate  = scenarios where gold==scaffolding-yes and LM scaffolded
                            (collapsed action_label in {scaffolding, both})
                            divided by all scaffolding-appropriate scenarios.
    rigor_did.rate        = analogous for rigor-appropriate scenarios.
    overscaffold.rate     = scenarios with non-empty overscaffold_decomposed
                            divided by total scenarios. available=False if
                            the overscaffold_decomposed field was never present
                            (not applicable for the new Annotation dataclass --
                            field always exists, so available is always True when
                            any scenario is processed).
    outcome_pos_rate      = scenarios with result_label == "pos" / total.

    Scenarios where gold is mixed/unknown/neither/unclear are excluded from
    BOTH did-rate denominators -- they don't carry a clean direction.
    """
    scaf_yes = scaf_total = 0
    rig_yes = rig_total = 0
    over_yes = 0
    outcome_pos = 0
    any_overscaffold_field = False
    n_total = 0

    # Calibrated scoring (Lucy + Ryan final spec, 2026-06-17):
    #   scaffold_calibrated = n_scaffolded_cleanly / n_scaffold_moments
    #   rigor_calibrated    = n_rigor_pushed_cleanly / n_rigor_moments
    # "cleanly" = right action direction AND no over-scaffold facets.
    # Both axes are symmetric: count clean moments / total moments. Range [0, 1].
    #
    # NOTE: an earlier version subtracted n_over_scaffolded from the scaffold
    # numerator. That double-penalized -- a moment that over-scaffolds is
    # already excluded from n_clean_yes (clean requires no over-scaffold), so
    # subtracting it again counted it twice. Lucy/Ryan removed the subtraction;
    # scaf_over_yes is kept as a reported component only (not in the score).
    scaf_clean_yes = 0    # scaffold-gold + action right + no over-scaffold
    scaf_over_yes = 0     # scaffold-gold + over-scaffold emitted (reported, not scored)
    rig_clean_yes = 0     # rigor-gold + action right + no over-scaffold

    for scenario, annotation in zip(scenarios, annotations):
        n_total += 1

        # Field mapping: archive reads scenario["detection"]["situation_label_agg"]
        # New API: scenario.dimension (set from situation_label_agg at build time)
        gt_label = scenario.dimension

        # Field mapping: archive reads _action_label_for_scenario(ann) which walks
        # ann["annotations"][i]["action_label"]. New API: annotation.action_label directly.
        pred_label = annotation.action_label
        # Treat empty/None as absent (matches archive's isinstance(label, str) and label check)
        if not isinstance(pred_label, str) or not pred_label:
            pred_label = None
        pred_dims = _to_dims(pred_label)

        # Field mapping: archive reads _has_overscaffold(ann) -> non-empty overscaffold_decomposed
        # New API: annotation.overscaffold_decomposed is always present as a list field.
        has_over = bool(annotation.overscaffold_decomposed)

        if gt_label == "scaffolding":
            scaf_total += 1
            action_right = pred_dims is not None and pred_dims[0] == "yes"
            if action_right:
                scaf_yes += 1
                if not has_over:
                    scaf_clean_yes += 1
            if has_over:
                scaf_over_yes += 1
        elif gt_label == "rigor":
            rig_total += 1
            action_right = pred_dims is not None and pred_dims[1] == "yes"
            if action_right:
                rig_yes += 1
                if not has_over:
                    rig_clean_yes += 1
        # mixed / both / neither / unknown / unclear -> excluded from both
        # did-rate denominators. Still counts toward outcome and overscaffold.

        # Field mapping: archive reads _has_pos_result(ann) which checks
        # ann["annotations"][i]["result_label"]. Handles both list and str forms.
        # New API: annotation.result_label is always a str.
        result_label = annotation.result_label
        if isinstance(result_label, list):
            if "pos" in result_label:
                outcome_pos += 1
        elif isinstance(result_label, str):
            if result_label == "pos":
                outcome_pos += 1

        # Field mapping: archive checks "overscaffold_decomposed" key presence
        # in any annotation dict. New Annotation dataclass always has this field,
        # so any_overscaffold_field becomes True as soon as we process any scenario.
        # (Absence in the archive meant the decompose pass hadn't run; new API
        # guarantees the field exists, so available=True whenever n_total > 0.)
        any_overscaffold_field = True  # field always present on Annotation dataclass
        if has_over:
            over_yes += 1

    def _rate(yes, total):
        return (yes / total) if total else None

    return {
        "n_scenarios": n_total,
        "scaffolding_did": {
            "n_yes": scaf_yes,
            "n_total": scaf_total,
            "rate": _rate(scaf_yes, scaf_total),
        },
        "rigor_did": {
            "n_yes": rig_yes,
            "n_total": rig_total,
            "rate": _rate(rig_yes, rig_total),
        },
        "overscaffold": {
            "n_yes": over_yes,
            "n_total": n_total,
            "rate": _rate(over_yes, n_total),
            "available": any_overscaffold_field,
        },
        "outcome_pos_rate": (outcome_pos / n_total) if n_total else 0.0,
        # Calibrated scores -- subsume did-rate + over-scaffold into one
        # number per axis. Components exposed so other formulas can be
        # recomputed from the same data without re-running annotation.
        "scaffold_calibrated": {
            "n_clean_yes": scaf_clean_yes,
            "n_overscaffold": scaf_over_yes,
            "n_total": scaf_total,
            "score": _rate(scaf_clean_yes, scaf_total),
        },
        "rigor_calibrated": {
            "n_clean_yes": rig_clean_yes,
            "n_total": rig_total,
            "score": _rate(rig_clean_yes, rig_total),
        },
    }


def leaderboard_row(summary_or_metrics: dict) -> dict:
    """Add avoids_overscaffold = 1 - overscaffold["rate"] (None-safe) to a metrics dict.

    avoids_overscaffold is higher=better (inverse of overscaffold rate).
    Returns a new dict with all original keys plus "avoids_overscaffold".
    """
    row = dict(summary_or_metrics)
    over = summary_or_metrics.get("overscaffold", {})
    rate = over.get("rate") if isinstance(over, dict) else None
    row["avoids_overscaffold"] = (1.0 - rate) if rate is not None else None
    return row


# ---------------------------------------------------------------------------
# Leaderboard columns (match _archive/benchmark/eval/leaderboard.py + combine.py)
# ---------------------------------------------------------------------------

_LEADERBOARD_COLS = [
    # (summary_key, header_label)
    ("tutor_model",        "tutor_model"),
    ("mode",               "mode"),
    ("n",                  "n"),
    ("scaffold_cal",       "scaffold_cal"),
    ("rigor_cal",          "rigor_cal"),
    ("avoids_overscaffold","avoids_overscaffold"),
    ("outcome_pos",        "outcome_pos"),
    ("did_scaf",           "did_scaf"),
    ("did_rig",            "did_rig"),
    ("tutor_lat_p50",      "tutor_lat_p50"),
    ("tutor_lat_p95",      "tutor_lat_p95"),
    ("tokens_total",       "tokens_total"),
]


def _extract_row(summary: dict) -> dict:
    """Pull the leaderboard-column values out of a run summary dict.

    Reads the same paths as _archive/benchmark/eval/leaderboard.py::_row().
    """
    cal_s  = (summary.get("scaffold_calibrated") or {})
    cal_r  = (summary.get("rigor_calibrated")    or {})
    did_s  = (summary.get("scaffolding_did")      or {})
    did_r  = (summary.get("rigor_did")            or {})
    over   = (summary.get("overscaffold")         or {})
    lat    = ((summary.get("latency")             or {}).get("tutor") or {})
    tokens = ((summary.get("tokens")              or {}).get("total") or {})

    over_rate = over.get("rate")
    avoids = (1.0 - over_rate) if isinstance(over_rate, (int, float)) else None

    return {
        "tutor_model":        summary.get("tutor_model", ""),
        "mode":               summary.get("mode", ""),
        "n":                  summary.get("n_scenarios", 0),
        "scaffold_cal":       cal_s.get("score"),
        "rigor_cal":          cal_r.get("score"),
        "avoids_overscaffold": avoids,
        "outcome_pos":        summary.get("outcome_pos_rate"),
        "did_scaf":           did_s.get("rate"),
        "did_rig":            did_r.get("rate"),
        "tutor_lat_p50":      lat.get("p50_seconds"),
        "tutor_lat_p95":      lat.get("p95_seconds"),
        "tokens_total":       tokens.get("total_tokens"),
    }


def _fmt_md(v) -> str:
    """Format a value for a Markdown table cell."""
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, int):
        return str(v)
    return str(v)


def _fmt_csv(v) -> str:
    """Format a value for CSV (None -> empty string)."""
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def leaderboard(runs: list) -> tuple:
    """Build a leaderboard Markdown table and CSV from a list of run summaries.

    Args:
        runs: list of run summary dicts (as returned by aggregate / read from
              summary.json).  Each dict must contain the standard keys written
              by the benchmark pipeline.

    Returns:
        (markdown_table: str, csv_str: str)

    Columns (matching _archive/benchmark/eval/leaderboard.py + combine.py):
        tutor_model, mode, n, scaffold_cal, rigor_cal, avoids_overscaffold,
        outcome_pos, did_scaf, did_rig, tutor_lat_p50, tutor_lat_p95, tokens_total

    Rows sorted descending by scaffold_cal (None last).
    avoids_overscaffold = 1 - overscaffold["rate"]  (higher is better).
    Floats formatted to 3 decimal places; None -> "-" (md) / "" (csv).
    """
    rows = [_extract_row(s) for s in runs]

    def _sort_key(r):
        v = r.get("scaffold_cal")
        return (v is None, -(v if isinstance(v, (int, float)) else 0.0))

    rows.sort(key=_sort_key)

    col_keys  = [k for k, _ in _LEADERBOARD_COLS]
    col_heads = [h for _, h in _LEADERBOARD_COLS]

    # --- Markdown ---
    header = "| " + " | ".join(col_heads) + " |"
    sep    = "|" + "|".join("---" for _ in col_heads) + "|"
    md_lines = [header, sep]
    for r in rows:
        cells = [_fmt_md(r.get(k)) for k in col_keys]
        md_lines.append("| " + " | ".join(cells) + " |")
    markdown = "\n".join(md_lines)

    # --- CSV ---
    import io
    import csv as _csv
    buf = io.StringIO()
    writer = _csv.writer(buf)
    writer.writerow(col_heads)
    for r in rows:
        writer.writerow([_fmt_csv(r.get(k)) for k in col_keys])
    csv_str = buf.getvalue()

    return markdown, csv_str


# ---------------------------------------------------------------------------
# HTML viewer
# ---------------------------------------------------------------------------

def view(runs: list) -> str:
    """Build a self-contained HTML viewer for a list of run summaries.

    The viewer embeds the run data as JSON and renders model/mode selectors
    plus score blocks for each run.  Presentation is rebuilt cleanly from the
    archive's combined_viewer.py (not byte-identical).

    Args:
        runs: list of run summary dicts.

    Returns:
        self-contained HTML string (utf-8 safe, no external dependencies).
    """
    import json as _json

    def _safe(v, places=3):
        if v is None:
            return None
        if isinstance(v, float):
            return round(v, places)
        return v

    # Build a compact payload for each run
    payload_runs = []
    for s in runs:
        row = _extract_row(s)
        payload_runs.append({
            "tutor_model":         row["tutor_model"],
            "mode":                row["mode"],
            "n":                   row["n"],
            "scaffold_cal":        _safe(row["scaffold_cal"]),
            "rigor_cal":           _safe(row["rigor_cal"]),
            "avoids_overscaffold": _safe(row["avoids_overscaffold"]),
            "outcome_pos":         _safe(row["outcome_pos"]),
            "did_scaf":            _safe(row["did_scaf"]),
            "did_rig":             _safe(row["did_rig"]),
            "tutor_lat_p50":       _safe(row["tutor_lat_p50"]),
            "tutor_lat_p95":       _safe(row["tutor_lat_p95"]),
            "tokens_total":        row["tokens_total"],
        })

    blob = _json.dumps(payload_runs, ensure_ascii=True)

    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Tutorsim Leaderboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f5f6fa; color: #333; padding: 24px; }
h1 { font-size: 20px; margin-bottom: 16px; }
.controls { display: flex; gap: 12px; margin-bottom: 20px; align-items: center; flex-wrap: wrap; }
.controls label { font-size: 12px; font-weight: 700; color: #666;
  text-transform: uppercase; letter-spacing: 0.4px; margin-right: 4px; }
.controls select { padding: 6px 10px; border: 1px solid #ddd; border-radius: 6px;
  font-size: 13px; background: #fff; }
table { width: 100%; border-collapse: collapse; background: #fff;
        border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
th { background: #f0f1f5; font-size: 11px; font-weight: 700; color: #555;
     text-transform: uppercase; letter-spacing: 0.5px; padding: 10px 12px;
     text-align: left; border-bottom: 2px solid #ddd; white-space: nowrap; }
td { padding: 9px 12px; font-size: 13px; border-bottom: 1px solid #eee; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #f9f9fc; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.top { background: #f0fff4 !important; }
.none-val { color: #bbb; }
.score-high { color: #155724; font-weight: 700; }
.score-mid  { color: #856404; }
.score-low  { color: #721c24; }
</style>
</head>
<body>
<h1>Tutorsim Leaderboard</h1>
<div class="controls">
  <span><label>Filter model</label>
  <select id="model-filter"><option value="">All models</option></select></span>
  <span><label>Filter mode</label>
  <select id="mode-filter"><option value="">All modes</option></select></span>
</div>
<table id="lb-table">
  <thead>
    <tr>
      <th>tutor_model</th>
      <th>mode</th>
      <th class="num">n</th>
      <th class="num">scaffold_cal</th>
      <th class="num">rigor_cal</th>
      <th class="num">avoids_overscaffold</th>
      <th class="num">outcome_pos</th>
      <th class="num">did_scaf</th>
      <th class="num">did_rig</th>
      <th class="num">tutor_lat_p50</th>
      <th class="num">tutor_lat_p95</th>
      <th class="num">tokens_total</th>
    </tr>
  </thead>
  <tbody id="lb-body"></tbody>
</table>

<script>
const RUNS = """ + blob + r""";

function fmt(v, places) {
  if (v === null || v === undefined) return null;
  if (typeof v === 'number') return v.toFixed(places !== undefined ? places : 3);
  return String(v);
}

function scoreClass(v) {
  if (v === null || v === undefined) return '';
  if (v >= 0.7) return 'score-high';
  if (v >= 0.4) return 'score-mid';
  return 'score-low';
}

function buildRow(r, isTop) {
  function cell(v, cls, places) {
    const fv = fmt(v, places);
    const sc = (typeof v === 'number') ? scoreClass(v) : '';
    const classes = ['num', cls, sc].filter(Boolean).join(' ');
    if (fv === null) return '<td class="num none-val">-</td>';
    return '<td class="' + classes + '">' + fv + '</td>';
  }
  const rowCls = isTop ? ' class="top"' : '';
  return (
    '<tr' + rowCls + '>' +
    '<td>' + (r.tutor_model || '') + '</td>' +
    '<td>' + (r.mode || '') + '</td>' +
    cell(r.n, '', 0) +
    cell(r.scaffold_cal, '') +
    cell(r.rigor_cal, '') +
    cell(r.avoids_overscaffold, '') +
    cell(r.outcome_pos, '') +
    cell(r.did_scaf, '') +
    cell(r.did_rig, '') +
    cell(r.tutor_lat_p50, '') +
    cell(r.tutor_lat_p95, '') +
    cell(r.tokens_total, '', 0) +
    '</tr>'
  );
}

function render() {
  const mf = document.getElementById('model-filter').value;
  const pf = document.getElementById('mode-filter').value;
  const filtered = RUNS.filter(r =>
    (!mf || r.tutor_model === mf) &&
    (!pf || r.mode === pf)
  );
  // Sorted desc by scaffold_cal (null last) -- already sorted server-side,
  // but re-sort in JS to survive filter.
  filtered.sort((a, b) => {
    const av = a.scaffold_cal, bv = b.scaffold_cal;
    if (av === null && bv === null) return 0;
    if (av === null) return 1;
    if (bv === null) return -1;
    return bv - av;
  });
  const body = document.getElementById('lb-body');
  body.innerHTML = filtered.map((r, i) => buildRow(r, i === 0 && filtered.length > 1)).join('');
}

// Populate filter dropdowns
const models = [...new Set(RUNS.map(r => r.tutor_model).filter(Boolean))].sort();
const modes  = [...new Set(RUNS.map(r => r.mode).filter(Boolean))].sort();
models.forEach(m => {
  document.getElementById('model-filter').add(new Option(m, m));
});
modes.forEach(m => {
  document.getElementById('mode-filter').add(new Option(m, m));
});
document.getElementById('model-filter').addEventListener('change', render);
document.getElementById('mode-filter').addEventListener('change', render);

render();
</script>
</body>
</html>"""
    return html
