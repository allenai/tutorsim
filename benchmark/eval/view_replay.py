"""Build an HTML viewer that compares the ORIGINAL real transcript to the AI-REPLAYED
continuation, side by side, for each benchmark scenario.

Three-column layout per scenario:
  LEFT   - Original transcript (full): turns 1..cut_turn shared, then the real
           human tutor / student turns from the recording.
  CENTER - Replayed transcript (full): turns 1..cut_turn shared, then the AI
           tutor + synthetic student turns from the cached exchange.
  RIGHT  - Annotations for the replayed continuation.

Usage:
    python -m benchmark.eval.view_replay --version dyn_smoke_v12_2026_06_09 --profile anthropic
"""

import argparse
import html
import json

from annotator.core.config import get_valid_styles
from annotator.core.storage import (
    load_transcript, load_benchmark_result,
    list_benchmark_result_files, get_benchmark_result_path,
)


def _classify_turn(turn_number: int, cut_turn: int) -> str:
    if turn_number <= cut_turn:
        return "prefix"
    return "post_cut"


def load_data(version: str, profile: str):
    scenarios_raw = load_benchmark_result(version, "scenarios.json")
    if not scenarios_raw:
        raise FileNotFoundError(f"No scenarios.json found for version {version}")

    # Exchanges
    exchanges = {}
    for fname in list_benchmark_result_files(version, "exchanges", profile):
        data = load_benchmark_result(version, "exchanges", profile, fname)
        if data:
            exchanges[fname.replace(".json", "")] = data

    # Annotations -- support any style dir present (v12, generous, balanced, demanding, ...).
    style_annotations: dict[str, dict] = {}
    try:
        ann_root = get_benchmark_result_path(version, "annotations", profile)
        candidate_styles = []
        if ann_root and ann_root.exists():
            candidate_styles = [p.name for p in ann_root.iterdir() if p.is_dir()]
    except Exception:
        candidate_styles = list(get_valid_styles())

    for style in candidate_styles:
        files = list_benchmark_result_files(version, "annotations", profile, style)
        if not files:
            continue
        style_annotations[style] = {}
        for fname in files:
            data = load_benchmark_result(version, "annotations", profile, style, fname)
            if data:
                style_annotations[style][fname.replace(".json", "")] = data

    transcripts_cache: dict = {}
    scenarios = []
    for s in scenarios_raw:
        scenario_id = s["scenario_id"]
        conv_id = s["conv_id"]
        cut_turn = s["cut_turn"]
        mode = s.get("mode", "human")
        detection = s.get("detection") or {}

        if conv_id not in transcripts_cache:
            transcripts_cache[conv_id] = load_transcript(conv_id)
        transcript_data = transcripts_cache.get(conv_id)
        if not transcript_data:
            continue

        original_turns = [
            {
                "turn_number": t["turn_number"],
                "role": t["role"],
                "text": t["text"],
                "kind": _classify_turn(t["turn_number"], cut_turn),
            }
            for t in transcript_data["turns"]
        ]

        exchange = exchanges.get(scenario_id) or {}
        generated_turns = exchange.get("generated_turns", []) or []

        replayed_turns = [
            t for t in original_turns if t["turn_number"] <= cut_turn
        ]
        for gt in generated_turns:
            replayed_turns.append({
                "turn_number": gt["turn_number"],
                "role": gt["role"],
                "text": gt["text"],
                "kind": "ai_generated",
            })

        per_style: dict[str, list[dict]] = {}
        for style, style_data in style_annotations.items():
            scenario_ann = style_data.get(scenario_id) or {}
            results = scenario_ann.get("results", {}).get(scenario_id, {})
            per_style[style] = results.get("annotations", []) or []

        scenarios.append({
            "scenario_id": scenario_id,
            "conv_id": conv_id,
            "cut_turn": cut_turn,
            "mode": mode,
            "detection": {
                "turn_start": detection.get("turn_start"),
                "turn_end": detection.get("turn_end"),
                "annotation_type": detection.get("annotation_type"),
                "situation_label_agg": detection.get("situation_label_agg"),
                "situation": detection.get("situation"),
                "chosen_cut_turn": detection.get("chosen_cut_turn"),
                "cut_votes": detection.get("cut_votes"),
                "cluster_size": detection.get("cluster_size"),
            },
            "original_turns": original_turns,
            "replayed_turns": replayed_turns,
            "style_annotations": per_style,
            "tutor_model": exchange.get("tutor_model", profile),
        })

    return scenarios


def escape(text):
    return html.escape(str(text)) if text else ""


def build_html(scenarios: list, version: str, profile: str) -> str:
    data_json = json.dumps(scenarios, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Replay Viewer -- {escape(version)} / {escape(profile)}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f5f6fa; color: #333; }}

.header {{
  background: #fff; border-bottom: 1px solid #e0e0e0; padding: 12px 24px;
  display: flex; align-items: center; gap: 16px; position: sticky; top: 0; z-index: 100;
  flex-wrap: wrap;
}}
.header h1 {{ font-size: 18px; }}
.header select {{
  padding: 8px 12px; border: 1px solid #ddd; border-radius: 6px;
  font-size: 14px; min-width: 360px; background: #fff;
}}
.header .info {{ font-size: 13px; color: #666; display: flex; gap: 16px; flex-wrap: wrap; }}
.header .info .tag {{
  background: #eef; color: #336; padding: 2px 8px; border-radius: 4px; font-size: 12px;
  font-weight: 600;
}}
.header .info .tag.human {{ background: #e3f2fd; color: #1565c0; }}
.header .info .tag.detected {{ background: #e8f5e9; color: #2e7d32; }}
.header .info .tag.agg-scaffolding {{ background: #e3f2fd; color: #0d47a1; border: 1px solid #90caf9; }}
.header .info .tag.agg-rigor {{ background: #fff3e0; color: #e65100; border: 1px solid #ffb74d; }}

.col.original h2 .agg-badge {{
  display: inline-block; font-size: 11px; font-weight: 700; padding: 2px 8px;
  margin-left: 10px; border-radius: 10px; vertical-align: middle;
  text-transform: uppercase; letter-spacing: 0.5px;
}}
.col.original h2 .agg-badge.scaffolding {{ background: #e3f2fd; color: #0d47a1; border: 1px solid #90caf9; }}
.col.original h2 .agg-badge.rigor {{ background: #fff3e0; color: #e65100; border: 1px solid #ffb74d; }}

.legend {{
  display: flex; gap: 14px; font-size: 12px; align-items: center; margin-left: auto;
}}
.legend-item {{ display: flex; align-items: center; gap: 4px; }}
.legend-swatch {{ width: 14px; height: 14px; border-radius: 3px; border: 1px solid rgba(0,0,0,0.1); }}

.main {{
  display: grid;
  grid-template-columns: 1fr 1fr 380px;
  gap: 0;
  height: calc(100vh - 56px);
  overflow: hidden;
}}
.col {{ overflow-y: auto; padding: 16px 20px; background: #fafbfc; border-right: 1px solid #e6e6ea; }}
.col.right {{ background: #fff; border-right: none; }}
.col h2 {{
  font-size: 13px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase;
  color: #555; padding-bottom: 8px; border-bottom: 2px solid #e0e0e0; margin-bottom: 10px;
  position: sticky; top: 0; background: #fafbfc; z-index: 1;
}}
.col.right h2 {{ background: #fff; }}
.col.original h2 {{ color: #1565c0; border-bottom-color: #bbdefb; }}
.col.replayed h2 {{ color: #6a1b9a; border-bottom-color: #ce93d8; }}

.cut-marker {{
  display: flex; align-items: center; gap: 8px; margin: 8px 0; color: #9c27b0;
  font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;
}}
.cut-marker::before, .cut-marker::after {{
  content: ''; flex: 1; height: 2px; background: #ce93d8;
}}

.turn {{
  display: flex; gap: 8px; margin-bottom: 4px; padding: 6px 10px;
  border-radius: 4px; font-size: 13px; line-height: 1.45;
  border-left: 3px solid transparent;
}}
.turn .turn-num {{ color: #999; font-size: 11px; min-width: 30px; text-align: right; padding-top: 2px; }}
.turn .role {{ font-weight: 600; min-width: 64px; font-size: 12px; padding-top: 2px; }}
.turn .role.tutor {{ color: #2c5282; }}
.turn .role.student {{ color: #276749; }}
.turn .role.system {{ color: #6c757d; font-style: italic; }}
.turn .text {{ flex: 1; white-space: pre-wrap; }}

.turn.prefix {{ background: transparent; }}
.turn.post_cut {{ background: #e3f2fd; border-left-color: #1976d2; }}
.turn.ai_generated {{ background: #f3e5f5; border-left-color: #9c27b0; }}
.turn.system {{ background: #f0f0f3; border-left-color: #9e9e9e; color: #555; font-style: italic; }}

/* Annotation cards */
.ann-style-section {{ margin-bottom: 16px; }}
.ann-style-title {{
  font-size: 12px; font-weight: 700; color: #444; padding: 6px 0;
  border-bottom: 1px solid #eee; margin-bottom: 8px; text-transform: uppercase;
  letter-spacing: 0.5px;
}}
.ann-card {{
  border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px;
  margin-bottom: 8px; background: #fff; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}}
.ann-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; flex-wrap: wrap; }}
.ann-badge {{
  font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 10px;
  text-transform: uppercase; letter-spacing: 0.5px;
}}
.ann-badge.scaffolding {{ background: #e3f2fd; color: #1565c0; }}
.ann-badge.rapport {{ background: #f3e5f5; color: #7b1fa2; }}
.ann-turns {{ font-size: 11px; color: #888; }}

.ann-field {{ margin-top: 4px; }}
.ann-field-label {{ font-size: 10px; font-weight: 700; color: #777; text-transform: uppercase; letter-spacing: 0.4px; }}
.ann-field-value {{ font-size: 12px; color: #333; margin-top: 2px; line-height: 1.4; white-space: pre-wrap; }}

.effectiveness {{
  display: inline-block; font-size: 10px; font-weight: 700; padding: 2px 8px;
  border-radius: 10px; margin-top: 6px;
}}
.effectiveness.effective {{ background: #d4edda; color: #155724; }}
.effectiveness.partial {{ background: #fff3cd; color: #856404; }}
.effectiveness.ineffective {{ background: #f8d7da; color: #721c24; }}
.effectiveness.unclear {{ background: #e2e3e5; color: #383d41; }}

.detection-box {{
  background: #fffde7; border: 1px solid #ffe082; border-radius: 6px;
  padding: 8px 10px; font-size: 12px; color: #5d4037; margin-bottom: 12px;
}}
.detection-box .label {{ font-weight: 700; }}

.empty {{ color: #999; font-style: italic; padding: 16px; text-align: center; }}
</style>
</head>
<body>
<div class="header">
  <h1>Replay Viewer</h1>
  <select id="scenario-select" onchange="selectScenario(this.value)">
    <option value="">Select a scenario...</option>
  </select>
  <div class="info" id="scenario-info"></div>
  <div class="legend">
    <div class="legend-item"><div class="legend-swatch" style="background:transparent;border:1px solid #ccc"></div> Shared prefix</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#e3f2fd;border-left:3px solid #1976d2"></div> Real continuation</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#f3e5f5;border-left:3px solid #9c27b0"></div> AI continuation</div>
  </div>
</div>

<div class="main">
  <div class="col original">
    <h2>Original Transcript <span id="agg-badge"></span></h2>
    <div id="col-original"></div>
  </div>
  <div class="col replayed">
    <h2>AI Replay</h2>
    <div id="col-replayed"></div>
  </div>
  <div class="col right">
    <h2>Annotations</h2>
    <div id="col-annotations"></div>
  </div>
</div>

<script>
const DATA = {data_json};

const sel = document.getElementById('scenario-select');
DATA.forEach((s, i) => {{
  const opt = document.createElement('option');
  opt.value = i;
  const label = s.scenario_id.length > 60 ? s.scenario_id.substring(0, 57) + '...' : s.scenario_id;
  opt.textContent = (i + 1) + '. ' + label;
  sel.appendChild(opt);
}});
if (DATA.length > 0) {{ sel.value = 0; selectScenario(0); }}

function escapeHtml(s) {{
  const div = document.createElement('div');
  div.textContent = (s == null ? '' : String(s));
  return div.innerHTML;
}}

function selectScenario(idx) {{
  if (idx === '' || idx == null) return;
  const s = DATA[idx];

  const info = document.getElementById('scenario-info');
  const modeClass = s.mode || 'human';
  const agg = (s.detection || {{}}).situation_label_agg || '';
  const aggTag = agg ? ('<span class="tag agg-' + agg + '">' + escapeHtml(agg) + '-appropriate</span>') : '';
  info.innerHTML =
    '<span class="tag ' + modeClass + '">' + escapeHtml(s.mode) + '</span>' +
    aggTag +
    '<span>Cut: turn ' + s.cut_turn + '</span>' +
    '<span>Conv: ' + escapeHtml(s.conv_id.substring(s.conv_id.length - 36)) + '</span>' +
    '<span>Tutor: ' + escapeHtml(s.tutor_model) + '</span>';

  const aggBadge = document.getElementById('agg-badge');
  aggBadge.innerHTML = agg
    ? '<span class="agg-badge ' + agg + '">' + escapeHtml(agg) + '-appropriate</span>'
    : '';

  document.getElementById('col-original').innerHTML = renderTurns(s.original_turns, s.cut_turn);
  document.getElementById('col-replayed').innerHTML = renderTurns(s.replayed_turns, s.cut_turn);
  document.getElementById('col-annotations').innerHTML = renderAnnotations(s);
}}

function renderTurns(turns, cut) {{
  let h = '';
  let inserted = false;
  for (const t of turns) {{
    if (!inserted && t.turn_number > cut) {{
      h += '<div class="cut-marker">After cut (turn ' + cut + ')</div>';
      inserted = true;
    }}
    const role = (t.role || '').toLowerCase();
    const roleClass = role === 'tutor' ? 'tutor' : (role === 'system' ? 'system' : 'student');
    let kindClass = t.kind;
    if (role === 'system') kindClass = 'system';
    h += '<div class="turn ' + kindClass + '">' +
         '<span class="turn-num">' + t.turn_number + '</span>' +
         '<span class="role ' + roleClass + '">' + escapeHtml(t.role) + '</span>' +
         '<span class="text">' + escapeHtml(t.text) + '</span>' +
         '</div>';
  }}
  return h || '<div class="empty">No turns.</div>';
}}

function renderAnnotations(s) {{
  const det = s.detection || {{}};
  let h = '';
  if (det.turn_start != null) {{
    h += '<div class="detection-box">' +
         '<div><span class="label">Moment:</span> turns ' + (det.turn_start || '?') + '&ndash;' + (det.turn_end || '?') + '</div>';
    if (det.situation_label_agg) {{
      h += '<div><span class="label">Type:</span> ' + escapeHtml(det.situation_label_agg) + '</div>';
    }}
    if (det.chosen_cut_turn != null && det.chosen_cut_turn !== s.cut_turn) {{
      h += '<div><span class="label">Chosen cut:</span> ' + det.chosen_cut_turn + ' &rarr; adjusted to ' + s.cut_turn + '</div>';
    }}
    if (det.cut_votes) {{
      const votes = Object.entries(det.cut_votes).map(([k, v]) => k + ':' + v).join(', ');
      h += '<div><span class="label">Votes:</span> ' + escapeHtml(votes) + ' (cluster=' + (det.cluster_size || '?') + ')</div>';
    }}
    h += '</div>';
  }}

  const styles = Object.keys(s.style_annotations || {{}});
  if (styles.length === 0) {{
    h += '<div class="empty">No annotations.</div>';
    return h;
  }}

  for (const style of styles) {{
    const anns = s.style_annotations[style] || [];
    h += '<div class="ann-style-section">';
    h += '<div class="ann-style-title">' + escapeHtml(style) + ' (' + anns.length + ')</div>';
    if (anns.length === 0) {{
      h += '<div class="empty">No annotations for this style.</div>';
    }} else {{
      for (const a of anns) {{
        const type = a.annotation_type || 'scaffolding';
        const label = a.strategy_label || a.effectiveness || 'unclear';
        h += '<div class="ann-card">';
        h += '<div class="ann-header">';
        h += '<span class="ann-badge ' + type + '">' + escapeHtml(type) + '</span>';
        h += '<span class="ann-turns">Turns ' + (a.turn_start || '?') + '&ndash;' + (a.turn_end || '?') + '</span>';
        h += '</div>';
        h += renderField('Situation', a.situation);
        h += renderField('Action', a.action);
        h += renderField('Result', a.result);
        h += '<span class="effectiveness ' + label + '">' + escapeHtml(label) + '</span>';
        h += '</div>';
      }}
    }}
    h += '</div>';
  }}
  return h;
}}

function renderField(label, value) {{
  if (!value) return '';
  return '<div class="ann-field">' +
         '<div class="ann-field-label">' + escapeHtml(label) + '</div>' +
         '<div class="ann-field-value">' + escapeHtml(value) + '</div>' +
         '</div>';
}}
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--profile", default="anthropic")
    ap.add_argument("--out", default=None,
                    help="Output HTML path (default: results/benchmark/<version>/viewer_replay_<profile>.html)")
    args = ap.parse_args()

    scenarios = load_data(args.version, args.profile)
    print(f"Loaded {len(scenarios)} scenarios for {args.profile}")
    out = build_html(scenarios, args.version, args.profile)

    if args.out:
        out_path = args.out
    else:
        from annotator.core.storage import get_benchmark_result_path
        base = get_benchmark_result_path(args.version)
        out_path = str(base / f"viewer_replay_{args.profile}.html")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
