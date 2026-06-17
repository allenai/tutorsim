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

from annotator.core.storage import (
    load_transcript, load_benchmark_result, _get_backend,
    list_benchmark_result_files, get_benchmark_result_path,
)


def _safe_call(fn, *args, **kwargs):
    """Call fn(...), returning ('', err_str) on exception so the viewer
    can render best-effort even if prompt reconstruction fails."""
    try:
        return fn(*args, **kwargs), None
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"


def _load_config(version: str) -> dict:
    """Read config.json saved by varied_smoke / `python -m benchmark`."""
    data = load_benchmark_result(version, "config.json")
    return data or {}


def _trait_persona_for(conv_id: str, cut_turn: int, trait_mode: str) -> "str | None":
    """Pull the cached trait persona, if any."""
    safe_conv = conv_id.replace("/", "_")
    safe_mode = (trait_mode or "joined-3").replace("/", "_")
    relpath = f"results/benchmark/_trait_cache/{safe_conv}__{cut_turn}__{safe_mode}.json"
    try:
        data = _get_backend().read_json(relpath)
        if data and isinstance(data, dict):
            return data.get("persona")
    except Exception:
        pass
    return None


def _resolved_trait_mode(student_mode: str) -> str:
    if student_mode == "trait":
        return "joined-3"
    return student_mode


def _aggregate_from_annotations(scenarios: list, annotations_by_scenario: dict) -> dict:
    """Compute per-dimension F1 + outcome rate + label distribution for the viewer header."""
    from benchmark.core.score import score_scenarios

    # score_scenarios expects (scenarios_dict_list, annotations_dict_list) aligned.
    scenario_dicts = [
        {"scenario_id": s["scenario_id"], "detection": s["detection"]}
        for s in scenarios
    ]
    ann_dicts = []
    for s in scenarios:
        ann_doc = annotations_by_scenario.get(s["scenario_id"]) or {}
        if "results" in ann_doc:
            scen = (ann_doc.get("results") or {}).get(s["scenario_id"]) or {}
        else:
            scen = ann_doc.get(s["scenario_id"]) or {}
        ann_dicts.append({"annotations": scen.get("annotations", []) or []})

    scores = score_scenarios(scenario_dicts, ann_dicts)

    # Action label distribution
    counts = {"scaffolding": 0, "rigor": 0, "both": 0, "neither": 0, "other": 0}
    for a in ann_dicts:
        for ann in a.get("annotations", []) or []:
            lbl = ann.get("action_label")
            if isinstance(lbl, list):
                lbl = lbl[0] if lbl else None
            if lbl in counts:
                counts[lbl] += 1
            elif lbl:
                counts["other"] += 1
    scores["action_label_counts"] = counts
    return scores


def _classify_turn(turn_number: int, cut_turn: int) -> str:
    if turn_number <= cut_turn:
        return "prefix"
    return "post_cut"


def load_data(version: str, profile: str):
    """Return (scenarios, run_meta).

    `scenarios` is a list of per-scenario dicts (transcripts + annotations +
    reconstructed prompts + trait persona + reference transcript + ended_via).
    `run_meta` carries the config + aggregate scores + label distribution.
    """
    scenarios_raw = load_benchmark_result(version, "scenarios.json")
    if not scenarios_raw:
        raise FileNotFoundError(f"No scenarios.json found for version {version}")

    cfg = _load_config(version)
    tutor_mode = cfg.get("tutor_mode")
    student_mode = cfg.get("student_mode") or "imitate_example"
    prompt_version = cfg.get("prompt_version") or "v5"
    resolved_trait_mode = _resolved_trait_mode(student_mode)

    # Exchanges
    exchanges = {}
    for fname in list_benchmark_result_files(version, "exchanges", profile):
        data = load_benchmark_result(version, "exchanges", profile, fname)
        if data:
            exchanges[fname.replace(".json", "")] = data

    # Annotations are flat per-scenario: annotations/{profile}/{scenario_id}.json
    # (No per-style subdir in the new pipeline.)
    annotations_by_scenario: dict[str, dict] = {}
    try:
        ann_root = get_benchmark_result_path(version, "annotations", profile)
        if ann_root and ann_root.exists():
            for entry in ann_root.iterdir():
                if entry.is_file() and entry.name.endswith(".json"):
                    sid = entry.name[:-5]
                    with open(entry, "r", encoding="utf-8") as f:
                        annotations_by_scenario[sid] = json.load(f)
    except Exception:
        annotations_by_scenario = {}

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

        # Annotation file is saved as {scenario_id: {conversation_id, annotations, ...}}
        # by run_phase2_and_score. Older v12 runs saved {results: {scenario_id: ...}}
        # so support both shapes.
        ann_doc = annotations_by_scenario.get(scenario_id) or {}
        if "results" in ann_doc:
            anns = ((ann_doc.get("results") or {}).get(scenario_id) or {}).get("annotations", []) or []
        else:
            anns = (ann_doc.get(scenario_id) or {}).get("annotations", []) or []

        # --- Reconstruct prompts the models actually saw (best-effort) ---
        from benchmark.core.tutors import build_tutor_system_prompt
        from benchmark.core.students import build_student_system_prompt, is_trait_mode
        from benchmark.core.exchange import _build_reference_transcript
        from annotator.core.annotate import _suggestion_text

        student_context = s.get("student_context") or ""
        transcript_prefix_str = "\n".join(
            f"Turn {t['turn_number']}. {t['role']}: {t['text']}"
            for t in original_turns if t["turn_number"] <= cut_turn
        )

        # Oracle tutor AND oracle student both use the same post-cut reference.
        reference_transcript = None
        if (tutor_mode == "oracle" or student_mode == "oracle") and transcript_data:
            reference_transcript = _build_reference_transcript(
                transcript_data, cut_turn,
            )

        tutor_prompt, tutor_err = _safe_call(
            build_tutor_system_prompt,
            tutor_mode,
            prompt_version=prompt_version,
            student_context=student_context,
            reference_transcript=reference_transcript,
        )

        persona = None
        if is_trait_mode(student_mode):
            persona = _trait_persona_for(conv_id, cut_turn, resolved_trait_mode)
        elif student_mode == "oracle":
            # Oracle student uses the joined-3 persona ('trait' default) plus
            # the in-moment post-cut turns. Read the cached persona if present.
            persona = _trait_persona_for(conv_id, cut_turn, "joined-3")
        student_prompt, student_err = _safe_call(
            build_student_system_prompt,
            student_mode,
            student_context=student_context,
            transcript_prefix=transcript_prefix_str,
            persona=persona,
            reference_transcript=reference_transcript,
        )

        suggestion_text = _suggestion_text(detection.get("situation_label_agg"))

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
            "annotations": anns,
            "tutor_model": exchange.get("tutor_model", profile),
            "ended_via": exchange.get("ended_via", ""),
            "tutor_system_prompt": tutor_prompt,
            "tutor_prompt_error": tutor_err,
            "student_system_prompt": student_prompt,
            "student_prompt_error": student_err,
            "trait_persona": persona,
            "reference_transcript": reference_transcript,
            "annotator_suggestion": suggestion_text,
        })

    aggregate = _aggregate_from_annotations(scenarios, annotations_by_scenario)
    # Merge in latency/tokens/timings from scores/{profile}.json if present.
    # The recomputation above doesn't have access to exchanges, so it can't
    # produce per-call latency stats. The saved scores file does -- it was
    # written by run.py with the full picture.
    scores_doc = load_benchmark_result(version, "scores", f"{profile}.json")
    if scores_doc:
        for key in ("latency", "tokens", "timings"):
            if key in scores_doc:
                aggregate[key] = scores_doc[key]

    run_meta = {
        "version": version,
        "profile": profile,
        "tutor_mode": tutor_mode or "default",
        "student_mode": student_mode,
        "prompt_version": prompt_version,
        "config": cfg,
        "aggregate": aggregate,
    }
    return scenarios, run_meta


def escape(text):
    return html.escape(str(text)) if text else ""


def build_html(scenarios: list, version: str, profile: str, run_meta: dict | None = None) -> str:
    data_json = json.dumps(scenarios, ensure_ascii=False)
    meta_json = json.dumps(run_meta or {}, ensure_ascii=False)

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
.ann-badge.rigor {{ background: #fff3e0; color: #e65100; }}
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

.facet {{ margin-top: 6px; font-size: 12px; line-height: 1.4; }}
.facet-text {{ color: #333; }}
.facet-badge {{
  display: inline-block; font-size: 11px; font-weight: 600;
  padding: 2px 8px; border-radius: 8px; margin-left: 6px;
  vertical-align: middle; letter-spacing: 0;
}}
.facet-badge.scaffolding {{ background:#e3f2fd; color:#0d47a1; }}
.facet-badge.rigor {{ background:#fff3e0; color:#e65100; }}
.facet-badge.neither {{ background:#eceff1; color:#455a64; }}
.facet-badge.both {{ background:#f3e5f5; color:#6a1b9a; }}
.facet-badge.pos {{ background:#d4edda; color:#155724; }}
.facet-badge.neg {{ background:#f8d7da; color:#721c24; }}
.tag.appropriate-yes {{ background:#d4edda; color:#155724; border:1px solid #b1dfbb; }}
.tag.appropriate-no {{ background:#f8d7da; color:#721c24; border:1px solid #f1aeb5; }}
.tag.appropriate-amb {{ background:#e2e3e5; color:#383d41; border:1px solid #c6c8ca; }}

/* Run header (top of page) */
.run-header {{
  background: #fff; border-bottom: 1px solid #e0e0e0; padding: 14px 24px;
}}
.run-header .row {{ display: flex; gap: 24px; flex-wrap: wrap; align-items: baseline; }}
.run-header .version-name {{ font-size: 16px; font-weight: 700; color: #222; }}
.run-header .pill {{
  display: inline-block; font-size: 11px; font-weight: 700; padding: 3px 10px;
  border-radius: 10px; text-transform: uppercase; letter-spacing: 0.5px;
  background: #eef; color: #336; margin-right: 4px;
}}
.run-header .pill.oracle {{ background: #fff3e0; color: #b35900; }}
.run-header .pill.default {{ background: #e3f2fd; color: #1565c0; }}
.run-header .pill.trait {{ background: #f3e5f5; color: #6a1b9a; }}
.run-header .scores {{ display: flex; gap: 18px; font-size: 13px; font-weight: 600; }}
.run-header .score-block .label {{
  font-size: 10px; color: #888; text-transform: uppercase; letter-spacing: 0.5px;
}}
.run-header .score-block .value {{ font-size: 18px; font-weight: 700; color: #222; }}
.run-header .label-dist {{ font-size: 12px; color: #666; }}
.run-header .label-dist .chip {{
  display: inline-block; padding: 2px 7px; margin-right: 4px;
  border-radius: 8px; font-weight: 600; font-size: 11px;
}}
.run-header .label-dist .chip.scaffolding {{ background: #e3f2fd; color: #0d47a1; }}
.run-header .label-dist .chip.rigor {{ background: #fff3e0; color: #e65100; }}
.run-header .label-dist .chip.both {{ background: #f3e5f5; color: #6a1b9a; }}
.run-header .label-dist .chip.neither {{ background: #eceff1; color: #455a64; }}

/* Collapsible reveals (prompts, persona, reference) */
details.reveal {{
  background: #f8f9fb; border: 1px solid #e2e4ea; border-radius: 6px;
  padding: 6px 10px; margin-bottom: 8px;
}}
details.reveal[open] {{ background: #fff; }}
details.reveal summary {{
  cursor: pointer; font-size: 12px; font-weight: 700; color: #4a5568;
  text-transform: uppercase; letter-spacing: 0.5px; outline: none;
}}
details.reveal summary .meta {{
  font-weight: 400; color: #888; text-transform: none; letter-spacing: 0;
  margin-left: 8px; font-size: 11px;
}}
details.reveal pre {{
  margin-top: 8px; padding: 10px; background: #1e293b; color: #e2e8f0;
  font-family: 'SFMono-Regular', Menlo, Consolas, monospace; font-size: 11px;
  line-height: 1.5; border-radius: 4px; white-space: pre-wrap;
  max-height: 400px; overflow-y: auto;
}}
details.reveal.persona pre {{ background: #4a148c; color: #f3e5f5; }}
details.reveal.reference pre {{ background: #663300; color: #ffe0b3; }}
details.reveal.suggestion pre {{ background: #5d4037; color: #ffe0b3; }}

.ended-via-badge {{
  display: inline-block; font-size: 10px; font-weight: 700; padding: 2px 8px;
  border-radius: 10px; text-transform: uppercase; letter-spacing: 0.5px;
  margin-left: 6px;
}}
.ended-via-badge.END {{ background: #d4edda; color: #155724; }}
.ended-via-badge.PROBLEM_CHANGE {{ background: #cfe2ff; color: #084298; }}
.ended-via-badge.NEXT_PROBLEM {{ background: #cfe2ff; color: #084298; }}  /* legacy v5 */
.ended-via-badge.MAX_TURNS {{ background: #fff3cd; color: #856404; }}
</style>
</head>
<body>
<div class="run-header" id="run-header"></div>

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
const META = {meta_json};

function renderRunHeader() {{
  const m = META || {{}};
  const agg = m.aggregate || {{}};
  const scafDid = agg.scaffolding_did || {{}};
  const rigDid = agg.rigor_did || {{}};
  const overscaffold = agg.overscaffold || {{}};
  const lc = agg.action_label_counts || {{}};
  const tutorPill = '<span class="pill ' + escapeHtml(m.tutor_mode || 'default') + '">' +
                    escapeHtml(m.tutor_mode || 'default') + ' tutor</span>';
  const studentPill = '<span class="pill ' + (String(m.student_mode || '').includes('trait') ? 'trait' : '') + '">' +
                      escapeHtml(m.student_mode || '') + ' student</span>';
  const chipLabels = {{
    'scaffolding': 'scaffolding',
    'rigor': 'rigor push',
    'both': 'scaffolding + rigor',
    'neither': 'neither',
  }};
  const chips = ['scaffolding', 'rigor', 'both', 'neither'].map(k =>
    '<span class="chip ' + k + '">' + chipLabels[k] + ': ' + (lc[k] || 0) + '</span>'
  ).join('');
  function _fmtRate(rate, denom) {{
    if (rate == null || denom === 0) return '—';
    return rate.toFixed(2);
  }}
  function _fracChip(yes, total) {{
    return total > 0 ? '<span style="color:#888;font-size:10px;">(' + yes + '/' + total + ')</span>' : '';
  }}
  // Lucy's three axes (2026-06-15): two recall-style "did the tutor do X?"
  // rates + over-scaffold rate. Over-scaffold rate may say "not available"
  // when the decompose_overscaffold step hasn't been run on the annotations.
  // Model-config strip (resolved tutor model + its generate kwargs + the
  // student's kwargs). Helps reviewers see what config produced these
  // annotations without spelunking through config.json.
  const cfg = m.config || {{}};
  const tutorModel = cfg.resolved_tutor_model || cfg.tutor_model || '?';
  const tutorKwargs = cfg.tutor_kwargs || {{}};
  const studentKwargs = cfg.student_kwargs || {{}};
  function _kwargsBadge(kw) {{
    const parts = [];
    if (kw.thinking === true) parts.push('thinking=on');
    else if (kw.thinking === false) parts.push('thinking=off');
    if (kw.effort) parts.push('effort=' + escapeHtml(kw.effort));
    if (kw.reasoning_effort) parts.push('reasoning_effort=' + escapeHtml(kw.reasoning_effort));
    if (kw.thinking_budget !== undefined && kw.thinking_budget !== null) {{
      parts.push('thinking_budget=' + (kw.thinking_budget === -1 ? 'dynamic' : kw.thinking_budget));
    }}
    return parts.length ? parts.join(' · ') : '(no extras)';
  }}
  const latencyBlock = (() => {{
    const lat = (agg.latency || {{}}).tutor || null;
    if (!lat) return '';
    return '<span style="margin-left:18px;color:#666;">tutor latency: ' +
           'mean ' + lat.mean_seconds.toFixed(2) + 's · ' +
           'p50 ' + lat.p50_seconds.toFixed(2) + 's · ' +
           'p95 ' + lat.p95_seconds.toFixed(2) + 's · ' +
           'n=' + lat.n + '</span>';
  }})();
  const tokensBlock = (() => {{
    const tk = (agg.tokens || {{}}).total || null;
    if (!tk) return '';
    return '<span style="margin-left:18px;color:#666;">tokens: ' +
           (tk.total_tokens || 0).toLocaleString() + ' total</span>';
  }})();

  const html = (
    '<div class="row">' +
      '<div class="version-name">' + escapeHtml(m.version || '') + '</div>' +
      tutorPill + studentPill +
      '<div style="margin-left:auto;color:#888;font-size:12px;">prompt_version=' + escapeHtml(m.prompt_version || '') +
        ' · profile=' + escapeHtml(m.profile || '') + '</div>' +
    '</div>' +
    '<div class="row" style="margin-top:4px;font-size:12px;color:#444;">' +
      '<span><b>tutor</b> ' + escapeHtml(tutorModel) + ' · ' + _kwargsBadge(tutorKwargs) + '</span>' +
      '<span style="margin-left:18px;"><b>student</b> ' + escapeHtml(cfg.student_mode || '') + ' · ' + _kwargsBadge(studentKwargs) + '</span>' +
      latencyBlock + tokensBlock +
    '</div>' +
    '<div class="row" style="margin-top:8px;">' +
      '<div class="scores">' +
        '<div class="score-block" title="Of scaffolding-appropriate scenarios, fraction where the AI tutor scaffolded (action_label in scaffolding-or-both). Higher better.">' +
          '<div class="label">did scaffold</div><div class="value">' + _fmtRate(scafDid.rate, scafDid.n_total) + '</div>' +
          '<div style="font-size:10px;color:#888;">' + _fracChip(scafDid.n_yes, scafDid.n_total) + '</div>' +
        '</div>' +
        '<div class="score-block" title="Of rigor-appropriate scenarios, fraction where the AI tutor pushed for rigor (action_label in rigor-or-both). Higher better.">' +
          '<div class="label">did rigor</div><div class="value">' + _fmtRate(rigDid.rate, rigDid.n_total) + '</div>' +
          '<div style="font-size:10px;color:#888;">' + _fracChip(rigDid.n_yes, rigDid.n_total) + '</div>' +
        '</div>' +
        (overscaffold.available
          ? '<div class="score-block" title="Fraction of scenarios where the annotator detected over-scaffolding facets in the tutor\\'s actions. Lower better.">' +
              '<div class="label">over-scaffold</div><div class="value">' + _fmtRate(overscaffold.rate, overscaffold.n_total) + '</div>' +
              '<div style="font-size:10px;color:#888;">' + _fracChip(overscaffold.n_yes, overscaffold.n_total) + '</div>' +
            '</div>'
          : '<div class="score-block" title="The annotations don\\'t carry overscaffold_decomposed -- re-run Phase 2 to populate it."><div class="label">over-scaffold</div><div class="value">—</div><div style="font-size:10px;color:#888;">not run</div></div>') +
        '<div class="score-block" title="Fraction of scenarios where the annotator labeled any result facet as student progressed (pos).">' +
          '<div class="label">outcome+ rate</div><div class="value">' + (agg.outcome_pos_rate != null ? agg.outcome_pos_rate.toFixed(2) : '—') + '</div>' +
        '</div>' +
      '</div>' +
      '<div class="label-dist" style="margin-left:auto;align-self:center;">action labels: ' + chips + '</div>' +
    '</div>'
  );
  document.getElementById('run-header').innerHTML = html;
}}

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

function appropriateClass(agg, actionLabels) {{
  const informative = (agg === 'scaffolding' || agg === 'rigor');
  if (!informative) return 'amb';
  const set = new Set(actionLabels);
  const pred = (agg === 'scaffolding') ? (set.has('scaffolding') || set.has('both'))
                                        : (set.has('rigor') || set.has('both'));
  return pred ? 'yes' : 'no';
}}

const ACTION_LABEL_TEXT = {{
  'scaffolding': 'scaffolding',
  'rigor': 'rigor push',
  'both': 'scaffolding + rigor',
  'neither': 'neither',
  'unclear': 'unclear',
}};
const RESULT_LABEL_TEXT = {{
  'pos': 'student progressed',
  'neg': 'student stuck',
  'no_evidence': 'no evidence',
  'unclear': 'unclear',
}};

function actionBadge(lbl) {{
  if (!lbl) return '';
  const txt = ACTION_LABEL_TEXT[lbl] || lbl;
  return '<span class="facet-badge ' + escapeHtml(lbl) + '">' + escapeHtml(txt) + '</span>';
}}
function resultBadge(lbl) {{
  if (!lbl) return '';
  const txt = RESULT_LABEL_TEXT[lbl] || lbl;
  return '<span class="facet-badge ' + escapeHtml(lbl) + '">' + escapeHtml(txt) + '</span>';
}}

function renderReveal(klass, summary, body, meta) {{
  if (!body) return '';
  const metaHtml = meta ? '<span class="meta">' + escapeHtml(meta) + '</span>' : '';
  return '<details class="reveal ' + klass + '"><summary>' + escapeHtml(summary) + metaHtml + '</summary>' +
         '<pre>' + escapeHtml(body) + '</pre></details>';
}}

function renderAnnotations(s) {{
  const det = s.detection || {{}};
  const anns = s.annotations || [];
  let h = '';

  // Ended-via badge inline at the top
  if (s.ended_via) {{
    h += '<div style="margin-bottom:8px;font-size:11px;color:#666;">terminated via ' +
         '<span class="ended-via-badge ' + escapeHtml(s.ended_via) + '">' + escapeHtml(s.ended_via) + '</span></div>';
  }}

  // Collapsible reveals: prompts + persona + reference + annotator suggestion
  const charsTutor = s.tutor_system_prompt ? s.tutor_system_prompt.length : 0;
  const charsStudent = s.student_system_prompt ? s.student_system_prompt.length : 0;
  h += renderReveal('tutor-prompt', 'Tutor system prompt',
                    s.tutor_system_prompt || (s.tutor_prompt_error || ''),
                    charsTutor ? `${{charsTutor}} chars` : null);
  h += renderReveal('student-prompt', 'Student system prompt',
                    s.student_system_prompt || (s.student_prompt_error || ''),
                    charsStudent ? `${{charsStudent}} chars` : null);
  // Note: trait_persona is already substituted into student_system_prompt
  // (replaces [[PERSONA_DESCRIPTION_HERE]]), and reference_transcript is
  // already substituted into whichever system prompt is the oracle one
  // (tutor's {{reference_transcript}} or student's [[REFERENCE_TRANSCRIPT_HERE]]).
  // We don't show them as separate reveals -- would just duplicate the text.
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
    // What the LM annotator actually saw as {{suggestion}} in the v13 Pass 2
    // prompt -- one of 6 canned sentences keyed off situation_label_agg.
    // This is the only teacher signal that reaches the annotator; teacher
    // free-text situation prose is NOT fed to the LM.
    if (s.annotator_suggestion) {{
      h += '<div style="margin-top:6px;"><span class="label">Hint to annotator LM:</span> <span style="color:#555;font-style:italic;">' + escapeHtml(s.annotator_suggestion) + '</span></div>';
    }}
    h += '</div>';
  }}

  // Flatten action labels across all annotations for verdict.
  // Lucy's structure pipeline emits a single string per annotation; older
  // annotator output emitted a list. Accept both.
  const allActionLabels = [];
  for (const a of anns) {{
    const lbl = a.action_label;
    if (Array.isArray(lbl)) {{
      for (const x of lbl) allActionLabels.push(x);
    }} else if (typeof lbl === 'string' && lbl) {{
      allActionLabels.push(lbl);
    }}
  }}
  const verdict = appropriateClass(det.situation_label_agg, allActionLabels);
  const verdictTag = (verdict === 'yes') ? 'appropriate ✓'
                   : (verdict === 'no') ? 'inappropriate ✗'
                   : 'ambiguous —';

  h += '<div class="ann-style-section">';
  h += '<div class="ann-style-title">verdict: ';
  h += '<span class="tag appropriate-' + verdict + '">' + verdictTag + '</span>';
  h += '</div>';

  if (anns.length === 0) {{
    h += '<div class="empty">No annotations.</div>';
    h += '</div>';
    return h;
  }}

  // Badge reflects the teacher consensus (situation_label_agg) when present
  // -- scaffolding-gold vs rigor-gold moments get different colors so the
  // gold direction is eyeball-able. Falls back to annotation_type for
  // rapport / random scenarios where situation_label_agg is absent.
  const goldDir = (det && det.situation_label_agg) || '';
  const useGold = (goldDir === 'scaffolding' || goldDir === 'rigor');
  for (const a of anns) {{
    const badge = useGold ? goldDir : (a.annotation_type || 'scaffolding');
    h += '<div class="ann-card">';
    h += '<div class="ann-header">';
    h += '<span class="ann-badge ' + badge + '">' + escapeHtml(badge) + '</span>';
    h += '<span class="ann-turns">turns ' + (a.turn_start || '?') + '&ndash;' + (a.turn_end || '?') + '</span>';
    h += '</div>';

    if (a.situation) {{
      h += '<div class="ann-field"><div class="ann-field-label">Situation</div><div class="ann-field-value">' + escapeHtml(a.situation) + '</div></div>';
    }}

    // Action: show prose narrative + overall classification badge + facet bullets.
    // (Lucy's structure pipeline emits ONE label per annotation, not per facet,
    // so facet rows don't carry their own badge.)
    if (a.action) {{
      h += '<div class="ann-field"><div class="ann-field-label">Action</div><div class="ann-field-value">' + escapeHtml(a.action) + '</div></div>';
    }}
    if (a.action_decomposed && a.action_decomposed.length) {{
      h += '<div class="ann-field"><div class="ann-field-label">Action facets</div>';
      const overall = (typeof a.action_label === 'string') ? a.action_label
                    : (Array.isArray(a.action_label) ? a.action_label[0] : '');
      if (overall) h += '<div style="margin:4px 0 6px;">' + actionBadge(overall) + '</div>';
      for (let i = 0; i < a.action_decomposed.length; i++) {{
        h += '<div class="facet"><span class="facet-text">' + escapeHtml(a.action_decomposed[i]) + '</span></div>';
      }}
      h += '</div>';
    }}

    // Over-scaffold facets (from Lucy's decompose_overscaffold step, PR #18).
    // Each facet is a description of an over-scaffolding behavior the annotator
    // detected in the tutor's action. Empty list = no over-scaffolding detected.
    // Missing key = the decompose step wasn't run on this annotation.
    if (Array.isArray(a.overscaffold_decomposed)) {{
      h += '<div class="ann-field"><div class="ann-field-label">Over-scaffold facets</div>';
      if (a.overscaffold_decomposed.length === 0) {{
        h += '<div class="empty" style="font-size:11px;color:#888;">(none detected)</div>';
      }} else {{
        for (const facet of a.overscaffold_decomposed) {{
          h += '<div class="facet" style="border-left:3px solid #ff8a65;padding-left:6px;"><span class="facet-text">' + escapeHtml(facet) + '</span></div>';
        }}
      }}
      h += '</div>';
    }}

    if (a.result) {{
      h += '<div class="ann-field"><div class="ann-field-label">Result</div><div class="ann-field-value">' + escapeHtml(a.result) + '</div></div>';
    }}
    if (a.result_decomposed && a.result_decomposed.length) {{
      h += '<div class="ann-field"><div class="ann-field-label">Result facets</div>';
      const overallR = (typeof a.result_label === 'string') ? a.result_label
                     : (Array.isArray(a.result_label) ? a.result_label[0] : '');
      if (overallR) h += '<div style="margin:4px 0 6px;">' + resultBadge(overallR) + '</div>';
      for (let i = 0; i < a.result_decomposed.length; i++) {{
        h += '<div class="facet"><span class="facet-text">' + escapeHtml(a.result_decomposed[i]) + '</span></div>';
      }}
      h += '</div>';
    }}

    h += '</div>';
  }}
  h += '</div>';
  return h;
}}

// --- Bootstrap (must run AFTER all const declarations above so the JS
// temporal-dead-zone doesn't trip when selectScenario(0) reaches actionBadge
// / resultBadge, which read ACTION_LABEL_TEXT / RESULT_LABEL_TEXT). ---
const sel = document.getElementById('scenario-select');
DATA.forEach((s, i) => {{
  const opt = document.createElement('option');
  opt.value = i;
  const label = s.scenario_id.length > 60 ? s.scenario_id.substring(0, 57) + '...' : s.scenario_id;
  opt.textContent = (i + 1) + '. ' + label;
  sel.appendChild(opt);
}});
renderRunHeader();
if (DATA.length > 0) {{ sel.value = 0; selectScenario(0); }}

</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--profile", default="anthropic")
    ap.add_argument("--out", default=None,
                    help="Output HTML path (default: results/benchmark/<version>/<version>.html)")
    args = ap.parse_args()

    scenarios, run_meta = load_data(args.version, args.profile)
    print(f"Loaded {len(scenarios)} scenarios for {args.profile}")
    out = build_html(scenarios, args.version, args.profile, run_meta)

    if args.out:
        out_path = args.out
    else:
        from annotator.core.storage import get_benchmark_result_path
        base = get_benchmark_result_path(args.version)
        out_path = str(base / f"{args.version}.html")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
