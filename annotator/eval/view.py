"""
Build a self-contained HTML comparison view of human vs LLM annotations.

Generates a single HTML file with:
- Three-panel layout: Human annotations | Transcript | LLM annotations
- Color-coded turn highlights (green=both, blue=human-only, orange=LLM-only)
- Annotation cards with Situation/Action/Result + effectiveness labels
- Conversation selector dropdown

Usage:
    python -m annotator.eval.view --version v1
"""

import argparse
import json
from pathlib import Path

from ..core.utils import load_ground_truth, load_split_ids
from ..core.storage import (
    load_annotator_result, annotator_result_exists, load_transcript,
    save_annotator_result, get_annotator_result_path,
)


def find_llm_file(version: str, gold: bool = False, profile: str | None = None):
    """Find the best available LLM results filename for this version.

    Checks profile-suffixed names first, then falls back to unprefixed names
    for backward compatibility with older results.
    """
    profile_suffix = f"_{profile}" if profile else ""
    if gold:
        candidates = [f"annotations_gold{profile_suffix}.json", "annotations_gold.json"]
    else:
        candidates = [f"annotations{profile_suffix}.json", "annotations.json", "outputs.json"]

    for name in candidates:
        if annotator_result_exists(version, name):
            return name, "annotations"

    # Fall back to detections-only
    if annotator_result_exists(version, "detections.json"):
        return "detections.json", "detections"

    return None, None


def load_data(version: str, gold: bool = False, profile: str | None = None):
    """Load ground truth, LLM results, and consolidated transcripts."""
    ground_truth = load_ground_truth()
    train_ids = load_split_ids("train")
    ground_truth["conversations"] = {
        k: v for k, v in ground_truth["conversations"].items() if k in train_ids
    }

    llm_filename, llm_type = find_llm_file(version, gold=gold, profile=profile)
    if not llm_filename:
        raise FileNotFoundError(f"No LLM results found for version {version}")

    llm_data = load_annotator_result(version, llm_filename)
    print(f"Using: {llm_filename} ({llm_type})")

    llm_results = llm_data.get("results", {})

    # Only include conversations with LLM results
    conv_ids = sorted(set(ground_truth["conversations"].keys()) & set(llm_results.keys()))

    conversations = []
    for conv_id in conv_ids:
        conv_data = load_transcript(conv_id)
        if conv_data is None:
            continue

        gt_conv = ground_truth["conversations"][conv_id]
        llm_conv = llm_results[conv_id]

        # Only keep LLM annotations whose type exists in this conversation's ground truth
        human_types = {m.get("annotation_type") for m in gt_conv["key_moments"]}

        # Detections-only files have "detections" key, annotation files have "annotations"
        raw_anns = llm_conv.get("annotations", llm_conv.get("detections", []))
        llm_annotations = []
        for ann in raw_anns:
            if ann.get("annotation_type") not in human_types:
                continue
            ann_copy = dict(ann)
            # Normalize label field -- annotations have "effectiveness", detections don't
            if "effectiveness" in ann_copy:
                ann_copy["strategy_label"] = ann_copy["effectiveness"]
            elif "strategy_label" not in ann_copy:
                ann_copy["strategy_label"] = "n/a"
            llm_annotations.append(ann_copy)

        conversations.append({
            "conversation_id": conv_id,
            "turns": conv_data["turns"],
            "num_turns": conv_data["num_turns"],
            "context": conv_data.get("context", ""),
            "human_annotations": gt_conv["key_moments"],
            "llm_annotations": llm_annotations,
        })

    return conversations


def build_html(conversations: list, version: str) -> str:
    """Generate the full HTML document."""
    conv_json = json.dumps(conversations, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Annotation Comparison -- {version}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f6fa; color: #333; }}

.header {{
  background: #fff; border-bottom: 1px solid #e0e0e0; padding: 16px 24px;
  display: flex; align-items: center; gap: 20px; position: sticky; top: 0; z-index: 100;
}}
.header h1 {{ font-size: 18px; color: #333; white-space: nowrap; }}
.header select {{
  padding: 8px 12px; border: 1px solid #ddd; border-radius: 6px;
  font-size: 14px; min-width: 300px; background: #fff;
}}
.header .stats {{
  font-size: 13px; color: #666; margin-left: auto; white-space: nowrap;
}}

.main {{
  display: flex; height: calc(100vh - 60px); overflow: hidden;
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
.sidebar h3.human {{ border-bottom-color: #4a90d9; }}
.sidebar h3.llm {{ border-bottom-color: #e8963e; }}

.transcript {{
  flex: 1; overflow-y: auto; padding: 16px 20px; background: #fafbfc;
}}

.turn {{
  display: flex; gap: 8px; margin-bottom: 4px; padding: 6px 10px;
  border-radius: 4px; transition: background 0.15s; font-size: 14px; line-height: 1.5;
}}
.turn .turn-num {{ color: #999; font-size: 12px; min-width: 32px; text-align: right; padding-top: 2px; }}
.turn .role {{ font-weight: 600; min-width: 70px; font-size: 13px; padding-top: 2px; }}
.turn .role.tutor {{ color: #2c5282; }}
.turn .role.student {{ color: #276749; }}
.turn .text {{ flex: 1; }}

.turn.bg-both {{ background: #d4edda; }}
.turn.bg-human {{ background: #cce5ff; }}
.turn.bg-llm {{ background: #ffecd2; }}
.turn.bg-none {{ background: transparent; }}
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

.legend {{
  display: flex; gap: 16px; font-size: 12px; align-items: center;
}}
.legend-item {{
  display: flex; align-items: center; gap: 4px;
}}
.legend-swatch {{
  width: 14px; height: 14px; border-radius: 3px; border: 1px solid rgba(0,0,0,0.1);
}}

.empty {{ color: #999; font-style: italic; padding: 20px; text-align: center; }}
</style>
</head>
<body>

<div class="header">
  <h1>Comparison View</h1>
  <select id="conv-select" onchange="selectConversation(this.value)">
    <option value="">Select a conversation...</option>
  </select>
  <div class="legend">
    <div class="legend-item"><div class="legend-swatch" style="background:#d4edda"></div> Both</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#cce5ff"></div> Human only</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#ffecd2"></div> LLM only</div>
  </div>
  <div class="stats" id="stats"></div>
</div>

<div class="main">
  <div class="sidebar" id="human-sidebar">
    <h3 class="human">Human Annotations</h3>
    <div id="human-cards"></div>
  </div>
  <div class="transcript" id="transcript"></div>
  <div class="sidebar right" id="llm-sidebar">
    <h3 class="llm">LLM Annotations</h3>
    <div id="llm-cards"></div>
  </div>
</div>

<script>
const DATA = {conv_json};

// Populate conversation selector
const select = document.getElementById('conv-select');
DATA.forEach((conv, i) => {{
  const opt = document.createElement('option');
  opt.value = i;
  opt.textContent = conv.conversation_id.substring(0, 50) + (conv.conversation_id.length > 50 ? '...' : '');
  select.appendChild(opt);
}});

let activeCard = null;
let highlightedTurns = [];

function turnsForAnnotations(annotations) {{
  const s = new Set();
  annotations.forEach(a => {{
    for (let t = a.turn_start; t <= a.turn_end; t++) s.add(t);
  }});
  return s;
}}

function selectConversation(idx) {{
  if (idx === '') return;
  const conv = DATA[idx];
  renderTranscript(conv);
  renderCards('human-cards', conv.human_annotations, true);
  renderCards('llm-cards', conv.llm_annotations, false);
  document.getElementById('stats').textContent =
    conv.num_turns + ' turns | ' +
    conv.human_annotations.length + ' human | ' +
    conv.llm_annotations.length + ' LLM';
}}

function renderTranscript(conv) {{
  const container = document.getElementById('transcript');
  const humanTurns = turnsForAnnotations(conv.human_annotations);
  const llmTurns = turnsForAnnotations(conv.llm_annotations);

  let html = '';
  conv.turns.forEach(turn => {{
    const n = turn.turn_number;
    const inHuman = humanTurns.has(n);
    const inLLM = llmTurns.has(n);
    let bgClass = 'bg-none';
    if (inHuman && inLLM) bgClass = 'bg-both';
    else if (inHuman) bgClass = 'bg-human';
    else if (inLLM) bgClass = 'bg-llm';

    const role = turn.role.toLowerCase();
    const roleClass = role === 'tutor' ? 'tutor' : 'student';
    const text = escapeHtml(turn.text);

    html += '<div class="turn ' + bgClass + '" id="turn-' + n + '">';
    html += '<span class="turn-num">' + n + '</span>';
    html += '<span class="role ' + roleClass + '">' + turn.role + '</span>';
    html += '<span class="text">' + text + '</span>';
    html += '</div>';
  }});
  container.innerHTML = html;
}}

function renderCards(containerId, annotations, isHuman) {{
  const container = document.getElementById(containerId);
  if (!annotations || annotations.length === 0) {{
    container.innerHTML = '<div class="empty">No annotations</div>';
    return;
  }}

  let html = '';
  annotations.forEach((ann, i) => {{
    const type = ann.annotation_type || 'unknown';
    const label = ann.strategy_label || ann.effectiveness || 'unclear';
    const prefix = isHuman ? 'human' : 'llm';
    const cardId = prefix + '-card-' + i;

    html += '<div class="ann-card" id="' + cardId + '" ';
    html += 'onmouseenter="highlightTurns(' + ann.turn_start + ',' + ann.turn_end + ',\\'' + cardId + '\\')" ';
    html += 'onmouseleave="clearHighlight()" ';
    html += 'onclick="scrollToTurn(' + ann.turn_start + ')">';

    html += '<div class="ann-header">';
    html += '<span class="ann-badge ' + type + '">' + type + '</span>';
    html += '<span class="ann-turns">Turns ' + ann.turn_start + '-' + ann.turn_end + '</span>';
    if (isHuman && ann.annotator_id) {{
      html += '<span class="ann-annotator">' + escapeHtml(ann.annotator_id) + '</span>';
    }}
    html += '</div>';

    html += field('Situation', ann.situation);
    html += field('Action', ann.action);
    html += field('Result', ann.result);

    html += '<span class="effectiveness ' + label + '">' + label + '</span>';
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

// Auto-select first conversation if available
if (DATA.length > 0) {{
  select.value = '0';
  selectConversation('0');
}}
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Build comparison view HTML")
    parser.add_argument("--version", required=True, help="Results version (e.g. v1)")
    parser.add_argument("--gold", action="store_true",
                        help="Use gold truth annotations (annotations_gold.json)")
    parser.add_argument("--profile", default=None,
                        help="Config profile used when generating annotations (e.g. anthropic, gemini)")
    args = parser.parse_args()

    version = args.version
    conversations = load_data(version, gold=args.gold, profile=args.profile)
    print(f"Loaded {len(conversations)} conversations with both human and LLM annotations")

    html_content = build_html(conversations, version)

    suffix = "_gold" if args.gold else ""
    output_dir = get_annotator_result_path(version)
    output_path = output_dir / f"comparison{suffix}.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Written: {output_path}")
    print(f"Open in browser to view.")


if __name__ == "__main__":
    main()
