"""
Build a self-contained HTML viewer for benchmark pipeline results.

Generates a single HTML file with:
- Three-panel layout: Ground Truth (left) | Transcript (center) | AI Annotations (right)
- AI-generated turns highlighted with purple/violet tint
- Annotation cards grouped by annotator style (Generous / Balanced / Demanding)
- Scenario selector with metadata (conv_id, mode, cut point)

Usage:
    python -m benchmark.eval.view --version v1 --profile gemini
"""

import argparse
import json
import html
from pathlib import Path

from annotator.core.storage import (
    load_transcript, load_benchmark_result,
    list_benchmark_result_files, get_benchmark_result_path,
)

REPO_ROOT = Path(__file__).parent.parent.parent
BENCHMARK_RESULTS_DIR = REPO_ROOT / "results" / "benchmark"


def load_data(version: str, profile: str):
    """Load scenarios, exchanges, annotations, transcripts, and ground truth."""
    # Load scenarios
    scenarios_raw = load_benchmark_result(version, "scenarios.json")
    if not scenarios_raw:
        raise FileNotFoundError(f"No scenarios.json found for version {version}")

    # Load exchanges for this profile
    exchanges = {}
    exchange_files = list_benchmark_result_files(version, "exchanges", profile)
    for fname in exchange_files:
        data = load_benchmark_result(version, "exchanges", profile, fname)
        if data:
            exchanges[fname.replace(".json", "")] = data

    # Load annotations for each style
    style_annotations = {}  # {style: {scenario_id: data}}
    for style in ("generous", "balanced", "demanding"):
        ann_files = list_benchmark_result_files(version, "annotations", profile, style)
        if ann_files:
            style_annotations[style] = {}
            for fname in ann_files:
                data = load_benchmark_result(version, "annotations", profile, style, fname)
                if data:
                    style_annotations[style][fname.replace(".json", "")] = data

    # Load transcripts per conv_id (cache)
    transcripts_cache = {}

    scenarios = []
    for s in scenarios_raw:
        scenario_id = s["scenario_id"]
        conv_id = s["conv_id"]
        cut_turn = s["cut_turn"]
        mode = s["mode"]
        detection = s.get("detection")

        # Load transcript if not cached
        if conv_id not in transcripts_cache:
            transcripts_cache[conv_id] = load_transcript(conv_id)

        transcript_data = transcripts_cache.get(conv_id)
        if not transcript_data:
            continue

        # Get exchange for this scenario
        exchange = exchanges.get(scenario_id)
        generated_turns = exchange.get("generated_turns", []) if exchange else []
        generated_turn_numbers = {t["turn_number"] for t in generated_turns}

        # Build full turn list: original turns up to cut_turn + generated turns
        all_turns = []
        for turn in transcript_data["turns"]:
            if turn["turn_number"] <= cut_turn:
                all_turns.append({
                    "turn_number": turn["turn_number"],
                    "role": turn["role"],
                    "text": turn["text"],
                    "is_generated": False,
                })
        for gt in generated_turns:
            all_turns.append({
                "turn_number": gt["turn_number"],
                "role": gt["role"],
                "text": gt["text"],
                "is_generated": True,
            })

        # Collect annotations per style
        per_style = {}
        for style, style_data in style_annotations.items():
            scenario_ann = style_data.get(scenario_id)
            if not scenario_ann:
                per_style[style] = []
                continue
            results = scenario_ann.get("results", {})
            # Annotations are keyed by scenario_id (annotator_bridge remaps conv_id)
            conv_results = results.get(scenario_id, {})
            anns = conv_results.get("annotations", [])
            per_style[style] = anns

        # Detection info (what triggered this scenario)
        detection_info = [detection] if detection else []

        scenarios.append({
            "scenario_id": scenario_id,
            "conv_id": conv_id,
            "cut_turn": cut_turn,
            "mode": mode,
            "turns": all_turns,
            "generated_turn_numbers": sorted(generated_turn_numbers),
            "style_annotations": per_style,
            "detection_info": detection_info,
            "tutor_model": exchange.get("tutor_model", profile) if exchange else profile,
        })

    return scenarios


def escape(text: str) -> str:
    return html.escape(str(text)) if text else ""


def build_html(scenarios: list, version: str, profile: str) -> str:
    """Generate the full HTML document."""
    data_json = json.dumps(scenarios, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Benchmark Viewer -- {escape(version)} / {escape(profile)}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f6fa; color: #333; }}

.header {{
  background: #fff; border-bottom: 1px solid #e0e0e0; padding: 12px 24px;
  display: flex; align-items: center; gap: 16px; position: sticky; top: 0; z-index: 100;
  flex-wrap: wrap;
}}
.header h1 {{ font-size: 18px; color: #333; white-space: nowrap; }}
.header select {{
  padding: 8px 12px; border: 1px solid #ddd; border-radius: 6px;
  font-size: 14px; min-width: 340px; background: #fff;
}}
.header .info {{
  font-size: 13px; color: #666; display: flex; gap: 16px; align-items: center; flex-wrap: wrap;
}}
.header .info .tag {{
  background: #eef; color: #336; padding: 2px 8px; border-radius: 4px; font-size: 12px;
  font-weight: 600;
}}
.header .info .tag.key_moment {{ background: #e8f5e9; color: #2e7d32; }}
.header .info .tag.random {{ background: #fff3e0; color: #e65100; }}

.legend {{
  display: flex; gap: 14px; font-size: 12px; align-items: center; margin-left: auto;
}}
.legend-item {{ display: flex; align-items: center; gap: 4px; }}
.legend-swatch {{
  width: 14px; height: 14px; border-radius: 3px; border: 1px solid rgba(0,0,0,0.1);
}}

.main {{
  display: flex; height: calc(100vh - 56px); overflow: hidden;
}}

.sidebar {{
  width: 340px; min-width: 280px; overflow-y: auto; padding: 16px;
  background: #fff; border-right: 1px solid #e0e0e0;
}}
.sidebar.right {{ border-right: none; border-left: 1px solid #e0e0e0; }}
.sidebar h3 {{
  font-size: 14px; color: #555; margin-bottom: 12px; padding-bottom: 8px;
  border-bottom: 2px solid #e0e0e0; position: sticky; top: 0; background: #fff; z-index: 1;
}}
.sidebar h3.ground-truth {{ border-bottom-color: #4a90d9; }}
.sidebar h3.ai-ann {{ border-bottom-color: #9c27b0; }}

.style-group {{
  margin-bottom: 16px;
}}
.style-header {{
  font-size: 13px; font-weight: 600; color: #555; padding: 6px 0 6px 0;
  border-bottom: 1px solid #eee; margin-bottom: 8px; text-transform: uppercase;
  letter-spacing: 0.5px;
}}
.style-header.generous {{ color: #2e7d32; border-bottom-color: #c8e6c9; }}
.style-header.balanced {{ color: #1565c0; border-bottom-color: #bbdefb; }}
.style-header.demanding {{ color: #c62828; border-bottom-color: #ffcdd2; }}

.transcript {{
  flex: 1; overflow-y: auto; padding: 16px 20px; background: #fafbfc;
}}

.cut-marker {{
  display: flex; align-items: center; gap: 8px; margin: 8px 0; color: #9c27b0;
  font-size: 12px; font-weight: 600;
}}
.cut-marker::before, .cut-marker::after {{
  content: ''; flex: 1; height: 2px; background: #ce93d8;
}}

.turn {{
  display: flex; gap: 8px; margin-bottom: 4px; padding: 6px 10px;
  border-radius: 4px; transition: background 0.15s; font-size: 14px; line-height: 1.5;
  border-left: 3px solid transparent;
}}
.turn .turn-num {{ color: #999; font-size: 12px; min-width: 32px; text-align: right; padding-top: 2px; }}
.turn .role {{ font-weight: 600; min-width: 70px; font-size: 13px; padding-top: 2px; }}
.turn .role.tutor {{ color: #2c5282; }}
.turn .role.student {{ color: #276749; }}
.turn .text {{ flex: 1; }}

.turn.original {{ background: transparent; }}
.turn.generated {{
  background: #f3e5f5; border-left: 3px solid #9c27b0;
}}
.turn.highlighted {{ background: #fff3cd !important; box-shadow: inset 0 0 0 2px #ffc107; }}

.ann-card {{
  border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px;
  margin-bottom: 10px; background: #fff; cursor: pointer;
  transition: all 0.15s; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}
.ann-card:hover {{
  box-shadow: 0 3px 10px rgba(0,0,0,0.12); transform: translateY(-1px);
}}
.ann-card.active {{ border-color: #ffc107; background: #fffde7; }}

.ann-header {{
  display: flex; align-items: center; gap: 8px; margin-bottom: 8px;
}}
.ann-badge {{
  font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 10px;
  text-transform: uppercase; letter-spacing: 0.5px;
}}
.ann-badge.scaffolding {{ background: #e3f2fd; color: #1565c0; }}
.ann-badge.rapport {{ background: #f3e5f5; color: #7b1fa2; }}
.ann-turns {{ font-size: 12px; color: #888; }}
.ann-annotator {{ font-size: 11px; color: #999; margin-left: auto; }}

.ann-field {{ margin-bottom: 6px; }}
.ann-field-label {{ font-size: 11px; font-weight: 600; color: #666; text-transform: uppercase; letter-spacing: 0.3px; }}
.ann-field-value {{ font-size: 13px; color: #333; margin-top: 2px; line-height: 1.4; }}

.effectiveness {{
  display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 8px;
  border-radius: 10px; margin-top: 4px;
}}
.effectiveness.effective {{ background: #d4edda; color: #155724; }}
.effectiveness.partial {{ background: #fff3cd; color: #856404; }}
.effectiveness.ineffective {{ background: #f8d7da; color: #721c24; }}
.effectiveness.unclear {{ background: #e2e3e5; color: #383d41; }}

.empty {{ color: #999; font-style: italic; padding: 20px; text-align: center; }}
</style>
</head>
<body>

<div class="header">
  <h1>Benchmark Viewer</h1>
  <select id="scenario-select" onchange="selectScenario(this.value)">
    <option value="">Select a scenario...</option>
  </select>
  <div class="info" id="scenario-info"></div>
  <div class="legend">
    <div class="legend-item"><div class="legend-swatch" style="background:transparent; border: 1px solid #ccc"></div> Original</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#f3e5f5; border-left: 3px solid #9c27b0"></div> AI Generated</div>
  </div>
</div>

<div class="main">
  <div class="sidebar" id="gt-sidebar">
    <h3 class="ground-truth">Ground Truth</h3>
    <div id="gt-cards"></div>
  </div>
  <div class="transcript" id="transcript"></div>
  <div class="sidebar right" id="ai-sidebar">
    <h3 class="ai-ann">AI Annotations</h3>
    <div id="ai-cards"></div>
  </div>
</div>

<script>
const DATA = {data_json};

const select = document.getElementById('scenario-select');
DATA.forEach((s, i) => {{
  const opt = document.createElement('option');
  opt.value = i;
  const label = s.scenario_id.length > 60 ? s.scenario_id.substring(0, 57) + '...' : s.scenario_id;
  opt.textContent = label;
  select.appendChild(opt);
}});

let activeCard = null;
let highlightedTurns = [];

function selectScenario(idx) {{
  if (idx === '') return;
  const s = DATA[idx];

  // Update info bar
  const info = document.getElementById('scenario-info');
  const modeClass = s.mode === 'key_moment' ? 'key_moment' : 'random';
  info.innerHTML =
    '<span class="tag ' + modeClass + '">' + escapeHtml(s.mode) + '</span>' +
    '<span>Conv: ' + escapeHtml(s.conv_id.substring(0, 40)) + '</span>' +
    '<span>Cut: turn ' + s.cut_turn + '</span>' +
    '<span>Model: ' + escapeHtml(s.tutor_model) + '</span>';

  renderTranscript(s);
  renderGroundTruth(s);
  renderAIAnnotations(s);
}}

function renderTranscript(s) {{
  const container = document.getElementById('transcript');
  const genSet = new Set(s.generated_turn_numbers);

  let html = '';
  let pastCut = false;

  s.turns.forEach(turn => {{
    const n = turn.turn_number;
    const isGen = turn.is_generated;

    // Insert cut marker before first generated turn
    if (isGen && !pastCut) {{
      pastCut = true;
      html += '<div class="cut-marker">AI-GENERATED TURNS BELOW (cut at turn ' + s.cut_turn + ')</div>';
    }}

    const bgClass = isGen ? 'generated' : 'original';
    const role = turn.role.toLowerCase();
    const roleClass = role === 'tutor' ? 'tutor' : 'student';

    html += '<div class="turn ' + bgClass + '" id="turn-' + n + '">';
    html += '<span class="turn-num">' + n + '</span>';
    html += '<span class="role ' + roleClass + '">' + escapeHtml(turn.role) + '</span>';
    html += '<span class="text">' + escapeHtml(turn.text) + '</span>';
    html += '</div>';
  }});

  container.innerHTML = html;
}}

function renderGroundTruth(s) {{
  const container = document.getElementById('gt-cards');
  const anns = s.detection_info;

  if (!anns || anns.length === 0) {{
    container.innerHTML = '<div class="empty">No ground truth annotations' +
      (s.mode === 'random' ? ' (random scenario)' : '') + '</div>';
    return;
  }}

  let html = '';
  anns.forEach((ann, i) => {{
    const cardId = 'gt-card-' + i;
    const type = ann.annotation_type || 'unknown';
    const label = ann.strategy_label || ann.effectiveness || 'unclear';

    html += '<div class="ann-card" id="' + cardId + '" ';
    html += 'onmouseenter="highlightTurns(' + (ann.turn_start || 0) + ',' + (ann.turn_end || 0) + ',\\'' + cardId + '\\')" ';
    html += 'onmouseleave="clearHighlight()" ';
    html += 'onclick="scrollToTurn(' + (ann.turn_start || 0) + ')">';

    html += '<div class="ann-header">';
    html += '<span class="ann-badge ' + type + '">' + escapeHtml(type) + '</span>';
    html += '<span class="ann-turns">Turns ' + (ann.turn_start || '?') + '-' + (ann.turn_end || '?') + '</span>';
    if (ann.annotator_id) {{
      html += '<span class="ann-annotator">' + escapeHtml(ann.annotator_id) + '</span>';
    }}
    html += '</div>';

    html += field('Situation', ann.situation);
    html += field('Action', ann.action);
    html += field('Result', ann.result);

    html += '<span class="effectiveness ' + label + '">' + escapeHtml(label) + '</span>';
    html += '</div>';
  }});

  container.innerHTML = html;
}}

function renderAIAnnotations(s) {{
  const container = document.getElementById('ai-cards');
  const styles = s.style_annotations;
  const styleOrder = ['generous', 'balanced', 'demanding'];

  // Get all style names, ordered
  const availableStyles = styleOrder.filter(st => st in styles);
  const extraStyles = Object.keys(styles).filter(st => !styleOrder.includes(st)).sort();
  const allStyles = availableStyles.concat(extraStyles);

  if (allStyles.length === 0) {{
    container.innerHTML = '<div class="empty">No AI annotations</div>';
    return;
  }}

  let html = '';
  let cardIdx = 0;

  allStyles.forEach(style => {{
    const anns = styles[style] || [];
    html += '<div class="style-group">';
    html += '<div class="style-header ' + style + '">' + escapeHtml(style) +
      ' (' + anns.length + ')</div>';

    if (anns.length === 0) {{
      html += '<div class="empty">No annotations</div>';
    }}

    anns.forEach((ann, i) => {{
      const cardId = 'ai-card-' + cardIdx;
      cardIdx++;
      const type = ann.annotation_type || 'unknown';
      const label = ann.effectiveness || ann.strategy_label || 'unclear';

      html += '<div class="ann-card" id="' + cardId + '" ';
      html += 'onmouseenter="highlightTurns(' + (ann.turn_start || 0) + ',' + (ann.turn_end || 0) + ',\\'' + cardId + '\\')" ';
      html += 'onmouseleave="clearHighlight()" ';
      html += 'onclick="scrollToTurn(' + (ann.turn_start || 0) + ')">';

      html += '<div class="ann-header">';
      html += '<span class="ann-badge ' + type + '">' + escapeHtml(type) + '</span>';
      html += '<span class="ann-turns">Turns ' + (ann.turn_start || '?') + '-' + (ann.turn_end || '?') + '</span>';
      html += '</div>';

      html += field('Situation', ann.situation);
      html += field('Action', ann.action);
      html += field('Result', ann.result);

      html += '<span class="effectiveness ' + label + '">' + escapeHtml(label) + '</span>';
      html += '</div>';
    }});

    html += '</div>';
  }});

  container.innerHTML = html;
}}

function field(label, value) {{
  if (!value) return '';
  return '<div class="ann-field">' +
    '<div class="ann-field-label">' + label + '</div>' +
    '<div class="ann-field-value">' + escapeHtml(value) + '</div>' +
    '</div>';
}}

function highlightTurns(start, end, cardId) {{
  clearHighlight();
  for (let t = start; t <= end; t++) {{
    const el = document.getElementById('turn-' + t);
    if (el) {{ el.classList.add('highlighted'); highlightedTurns.push(el); }}
  }}
  if (cardId) {{
    const card = document.getElementById(cardId);
    if (card) {{ card.classList.add('active'); activeCard = card; }}
  }}
}}

function clearHighlight() {{
  highlightedTurns.forEach(el => el.classList.remove('highlighted'));
  highlightedTurns = [];
  if (activeCard) {{ activeCard.classList.remove('active'); activeCard = null; }}
}}

function scrollToTurn(turnNum) {{
  const el = document.getElementById('turn-' + turnNum);
  if (el) el.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
}}

function escapeHtml(text) {{
  if (!text) return '';
  const div = document.createElement('div');
  div.textContent = String(text);
  return div.innerHTML;
}}

// Auto-select first scenario
if (DATA.length > 0) {{
  select.value = '0';
  selectScenario('0');
}}
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Build benchmark HTML viewer")
    parser.add_argument("--version", required=True, help="Benchmark version (e.g. v1)")
    parser.add_argument("--profile", required=True,
                        help="Tutor profile to view (e.g. gemini, openai, anthropic)")
    args = parser.parse_args()

    version = args.version
    profile = args.profile

    scenarios = load_data(version, profile)
    print(f"Loaded {len(scenarios)} scenarios for {profile}")

    html_content = build_html(scenarios, version, profile)

    output_dir = get_benchmark_result_path(version)
    output_path = output_dir / f"viewer_{profile}.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Written: {output_path}")
    print(f"Open in browser to view.")


if __name__ == "__main__":
    main()
