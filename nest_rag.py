# -*- coding: utf-8 -*-
"""
nest_rag.py — P.R.I.S.M. RAG pipeline
=====================================
Ingests the compact JSON action log produced by mediapipe_inference.py into a
local ChromaDB vector store, generates a natural-language video summary, and
runs an interactive Q&A loop — all powered by the lfm-2.5-thinking:latest
model via Ollama.

Usage
-----
  # First time (builds DB + generates summary):
  python nest_rag.py --json output_action_action_log.json

  # Force-regenerate summary even if cached:
  python nest_rag.py --json output_action_action_log.json --regenerate

  # Query directly without entering interactive loop:
  python nest_rag.py --json output_action_action_log.json --query "What was person 0 doing at 10 seconds?"

Requirements
------------
  pip install chromadb sentence-transformers ollama
  # Also: Ollama running locally with the model pulled:
  ollama pull lfm-2.5-thinking:latest
"""

import sys
import io
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# ── Third-party imports (graceful error messages) ─────────────────────────────
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    sys.exit("ERROR: chromadb not installed. Run: pip install chromadb>=0.5.0")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("ERROR: sentence-transformers not installed. Run: pip install sentence-transformers>=2.6.0")

try:
    import ollama as ollama_client
except ImportError:
    sys.exit("ERROR: ollama not installed. Run: pip install ollama>=0.2.0")

try:
    import google.generativeai as genai
except ImportError:
    print("WARNING: google-generativeai not installed. Gemini bypass will fail. Run: pip install google-generativeai")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"   # 22 MB, pure CPU
LLM_MODEL     = "lfm2.5-thinking:latest"
CHROMA_DIR    = "./nest_chroma_db"                          # persistent vector store root
TOP_K         = 6                                           # chunks to retrieve per query (lowered to save RAM)

# Ollama generation options — tune for your CPU core count.
# num_thread: 4 is a safe default; increase only if you have 8+ physical cores free.
# num_ctx: keep small on CPU — 1024 for Q&A, 2048 for summary.
# keep_alive: "5m" releases the model from RAM after 5 minutes of idle.
# Set num_gpu to 1 to offload to GPU for local LLM
LLM_OPTIONS_QUERY   = {"num_ctx": 1024, "temperature": 0.3, "keep_alive": "5m", "num_gpu": 1}
LLM_OPTIONS_SUMMARY = {"num_ctx": 2048, "temperature": 0.4, "keep_alive": "5m", "num_gpu": 1}

USE_GEMINI = False  # Toggle between on-chip local LLM (False) and Gemini API (True)

SYSTEM_PROMPT = """\
You are P.R.I.S.M. Assistant, a detailed AI analyst for patient-caregiver monitoring videos.
You have access to structured action recognition data: per-second action labels, confidence
scores, bounding box interactions, and timeline segments for each tracked person (Track ID).

Critical reasoning rules:
- The action recognition model operates on 90-frame sliding windows (~3 seconds of video),
  so a single action observed across several consecutive timestamps likely represents ONE
  continuous activity, not separate repeated actions. Reason about temporal continuity:
  merge nearby identical or closely related actions into coherent narrative paragraphs.
- If Track 0 is labelled 'walking towards' at t=4s, 6s, 7s, 9s consecutively, describe
  this as "Track 0 was walking towards [someone/the camera] from 4s to 9s" — not as
  four separate events.
- Always cite timestamp ranges (start–end) not just individual frames.
- Refer to people by Track ID (e.g., Track 0, Track 1).
- If retrieved data is insufficient for a question, say so clearly.
- Give DETAILED, structured answers with sections when answering about the full video.
- For medical/clinical events (falling, staggering, chest pain, nausea, headache, etc.)
  highlight them at the top of your answer with a WARNING label.
- When listing actions, provide: action name, track, time range, average confidence, category.
"""


# ══════════════════════════════════════════════════════════════════════════════
# JSON → TEXT CHUNK CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_time(t: float) -> str:
    """Format seconds as 0:00 style string."""
    m, s = divmod(int(t), 60)
    return f"{m}:{s:02d}s"


def parse_json_to_documents(log: dict) -> list[dict]:
    """
    Convert the compact action log JSON into a list of text chunks for embedding.

    Each chunk is a dict:
        {
            "id"      : str,   # unique document ID for ChromaDB
            "text"    : str,   # natural-language description
            "metadata": dict,  # structured metadata for filtering
        }

    Chunk types produced:
        session_meta        — video metadata and processing info
        overall_summary     — action distribution across the whole video
        per_track_{id}      — dominant action + top-3 per track
        segment_{i}         — contiguous action block with timestamps
        interaction_{i}     — interaction event with start/end times
        frame_{t}           — per-second snapshot of all actions + interactions
    """
    docs = []
    session = log.get("session", {})
    summary = log.get("summary", {})
    frames  = log.get("frames", [])

    video_name   = session.get("video", "unknown")
    fps          = session.get("fps", 25)
    duration     = session.get("duration_sec", 0)
    resolution   = session.get("resolution", [0, 0])
    device       = session.get("device", "cpu")
    model        = session.get("model", "EfficientGCN-B0")
    total_tracks = session.get("total_tracks", 0)
    num_classes  = session.get("num_classes", 45)

    # ── 1. Session metadata chunk ────────────────────────────────────────────
    docs.append({
        "id": "session_meta",
        "text": (
            f"VIDEO METADATA: The video file is '{video_name}'. "
            f"Duration: {duration} seconds ({_fmt_time(duration)}). "
            f"Frame rate: {fps} fps. Resolution: {resolution[0]}x{resolution[1]} pixels. "
            f"Processed on device: {device}. "
            f"Action recognition model: {model} ({num_classes} action classes). "
            f"Total unique person tracks detected: {total_tracks}. "
            f"Processed at: {session.get('processed_at', 'unknown')}."
        ),
        "metadata": {
            "type": "session_meta",
            "video": video_name,
            "duration_sec": duration,
        },
    })

    # ── 2. Overall action distribution summary chunk ────────────────────────
    action_dist = summary.get("action_distribution", {})
    total_recog = summary.get("total_recognitions", 0)
    unique_tracks = summary.get("unique_tracks", [])
    track_ids_str = ", ".join(str(t) for t in unique_tracks)

    dist_lines = [
        f"  • {label}: {count} recognition(s)"
        for label, count in action_dist.items()
    ]
    dist_text = "\n".join(dist_lines) if dist_lines else "  (no actions recognised)"

    docs.append({
        "id": "overall_summary",
        "text": (
            f"OVERALL VIDEO SUMMARY: The video '{video_name}' (duration: {duration}s) "
            f"contains {total_recog} action recognition events across person tracks: {track_ids_str}.\n"
            f"Action frequency breakdown:\n{dist_text}"
        ),
        "metadata": {
            "type": "overall_summary",
            "total_recognitions": total_recog,
        },
    })

    # ── 3. Per-track summary chunks ──────────────────────────────────────────
    per_track_summary = summary.get("per_track_summary", {})
    for track_key, info in per_track_summary.items():
        dominant = info.get("dominant_action", "unknown")
        top3     = info.get("top3", [])
        top3_str = "; ".join(
            f"{lbl} (avg conf {conf:.0%})" for lbl, conf in top3
        )
        docs.append({
            "id": f"per_track_{track_key}",
            "text": (
                f"TRACK {track_key} PROFILE: Person with Track ID {track_key} "
                f"most frequently performed '{dominant}' throughout the video. "
                f"Top actions for Track {track_key}: {top3_str}."
            ),
            "metadata": {
                "type": "per_track_summary",
                "track": track_key,
                "dominant_action": dominant,
            },
        })

    # ── 4. Action segment chunks ─────────────────────────────────────────────
    action_segments = summary.get("action_segments", [])
    for i, seg in enumerate(action_segments):
        t_start    = seg.get("t_start", 0)
        t_end      = seg.get("t_end", 0)
        track      = seg.get("track", "?")
        action     = seg.get("action", "unknown")
        category   = seg.get("category", "unknown")
        avg_conf   = seg.get("avg_conf", 0)
        frame_cnt  = seg.get("frame_count", 1)
        duration_s = round(t_end - t_start, 1)

        docs.append({
            "id": f"segment_{i}",
            "text": (
                f"ACTION SEGMENT: Track {track} performed '{action}' "
                f"({category} category) from {_fmt_time(t_start)} to {_fmt_time(t_end)} "
                f"({t_start}s–{t_end}s), lasting {duration_s} seconds. "
                f"Average confidence: {avg_conf:.0%}. "
                f"Observed in {frame_cnt} logged frame(s)."
            ),
            "metadata": {
                "type": "action_segment",
                "track": str(track),
                "action": action,
                "category": category,
                "t_start": t_start,
                "t_end": t_end,
            },
        })

    # ── 5. Interaction event chunks ──────────────────────────────────────────
    interaction_events = summary.get("interaction_events", [])
    for i, ev in enumerate(interaction_events):
        id_a    = ev.get("track_a", "?")
        id_b    = ev.get("track_b", "?")
        t_start = ev.get("t_start", 0)
        t_end   = ev.get("t_end", 0)
        dur_sec = ev.get("duration_sec", 0)
        dom_act = ev.get("dominant_action", "unknown")

        docs.append({
            "id": f"interaction_{i}",
            "text": (
                f"INTERACTION EVENT: Tracks {id_a} and {id_b} were actively interacting "
                f"from {_fmt_time(t_start)} to {_fmt_time(t_end)} "
                f"({t_start}s–{t_end}s), duration {dur_sec}s. "
                f"The dominant action during this interaction was '{dom_act}'."
            ),
            "metadata": {
                "type": "interaction_event",
                "track_a": str(id_a),
                "track_b": str(id_b),
                "t_start": t_start,
                "t_end": t_end,
                "dominant_action": dom_act,
            },
        })

    # ── 6. Per-frame snapshot chunks ─────────────────────────────────────────
    for frame in frames:
        t        = frame.get("t", 0)
        fi       = frame.get("fi", 0)
        actions  = frame.get("actions", {})
        interact = frame.get("interact", [])
        tracks   = frame.get("tracks", {})

        if not actions and not interact:
            continue  # skip empty frames to save vector DB space

        action_parts = []
        for key, val in actions.items():
            # val is [code, label, conf, category]
            if isinstance(val, list) and len(val) >= 3:
                code, label, conf = val[0], val[1], val[2]
                cat = val[3] if len(val) > 3 else "unknown"
                action_parts.append(
                    f"Track {key}: '{label}' ({cat}, conf {conf:.0%})"
                )

        interact_parts = []
        for iv in interact:
            if iv.get("ia", 0) == 1:
                dist = iv.get("dist", "?")
                interact_parts.append(
                    f"Tracks {iv.get('a','?')} and {iv.get('b','?')} "
                    f"interacting (distance {dist}px)"
                )

        track_list = ", ".join(tracks.keys())
        act_str    = "; ".join(action_parts) if action_parts else "no action recognised"
        int_str    = ("; ".join(interact_parts) if interact_parts else "no active interaction")

        docs.append({
            "id": f"frame_{fi}",
            "text": (
                f"FRAME SNAPSHOT at {_fmt_time(t)} ({t}s, frame #{fi}): "
                f"Active tracks: {track_list}. "
                f"Actions — {act_str}. "
                f"Interactions — {int_str}."
            ),
            "metadata": {
                "type": "frame_snapshot",
                "timestamp_sec": t,
                "frame_index": fi,
            },
        })

    return docs


# ══════════════════════════════════════════════════════════════════════════════
# CHROMA DB BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _collection_name(video_name: str) -> str:
    """Derive a safe ChromaDB collection name from the video filename."""
    stem = Path(video_name).stem
    # ChromaDB collection names: 3-63 chars, alphanumeric + hyphens/underscores
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return f"nest_{safe[:55]}"


def build_or_load_db(
    log: dict,
    json_path: str,
    embed_model: SentenceTransformer,
    force_rebuild: bool = False,
) -> chromadb.Collection:
    """
    Build (or load from disk) a ChromaDB collection for the given JSON log.

    If the collection already exists and force_rebuild is False, the existing
    collection is returned immediately (fast path).
    """
    video_name   = log.get("session", {}).get("video", "unknown")
    coll_name    = _collection_name(video_name)

    persist_path = os.path.join(CHROMA_DIR, Path(json_path).stem)
    client = chromadb.PersistentClient(path=persist_path)

    existing = [c.name for c in client.list_collections()]
    if coll_name in existing and not force_rebuild:
        print(f"  ✓ Loaded existing ChromaDB collection '{coll_name}' from {persist_path}")
        return client.get_collection(name=coll_name)

    # (Re)build
    if coll_name in existing:
        client.delete_collection(name=coll_name)
        print(f"  ⟳ Rebuilding collection '{coll_name}'…")
    else:
        print(f"  ⟳ Building new collection '{coll_name}'…")

    collection = client.create_collection(
        name=coll_name,
        metadata={"hnsw:space": "cosine"},
    )

    docs   = parse_json_to_documents(log)
    texts  = [d["text"]     for d in docs]
    ids    = [d["id"]       for d in docs]
    metas  = [d["metadata"] for d in docs]

    # Embed in small batches to avoid RAM spikes; normalize for cosine similarity
    batch_size = 16
    all_embeds = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        embeds = embed_model.encode(
            batch,
            show_progress_bar=False,
            normalize_embeddings=True,  # pre-normalize → faster cosine search
            batch_size=batch_size,
        ).tolist()
        all_embeds.extend(embeds)

    collection.add(embeddings=all_embeds, documents=texts, ids=ids, metadatas=metas)
    print(f"  ✓ Indexed {len(docs)} document chunks into '{coll_name}'.")
    return collection


# ══════════════════════════════════════════════════════════════════════════════
# LLM HELPERS (Ollama streaming with tok/s display)
# ══════════════════════════════════════════════════════════════════════════════

def _stream_llm(
    messages    : list[dict],
    label       : str  = "Response",
    think       : bool = False,
    llm_options : dict = None,
) -> str:
    """
    Stream a response from lfm2.5-thinking via Ollama.

    Parameters
    ----------
    think : bool
        False (default) = disable internal thinking tokens → much faster on CPU.
        True            = enable full chain-of-thought reasoning (use for summary).
    llm_options : dict
        Ollama model options (num_thread, num_ctx, temperature, etc.).

    Token/sec is measured from the Ollama eval_count + eval_duration fields
    on the final streaming chunk (accurate) with a word-count fallback.
    """
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}\n")

    opts = llm_options or LLM_OPTIONS_QUERY
    full_text    = ""
    eval_count   = 0        # actual output token count from Ollama
    eval_dur_ns  = 0        # eval duration in nanoseconds from Ollama
    word_count   = 0        # fallback word-level count
    t_start      = time.perf_counter()

    # State for filtering <think>...</think> blocks from terminal output.
    # The model sometimes emits thinking tokens in msg.content even when
    # think=False; we strip them so the terminal only shows the real answer.
    _in_think   = False   # currently inside a <think> block?
    _think_buf  = ""      # accumulates partial tag text for boundary detection

    def _print_filtered(text: str) -> str:
        """
        Filter <think>...</think> spans out of `text` before printing.
        Returns only the visible (non-thinking) portion, which is also
        what gets accumulated into full_text.
        """
        nonlocal _in_think, _think_buf
        visible = ""
        i = 0
        while i < len(text):
            ch = text[i]
            if not _in_think:
                # Look for opening tag
                _think_buf += ch
                if "<think>" in _think_buf:
                    # Emit everything before the tag, then enter think mode
                    before = _think_buf[: _think_buf.index("<think>")]
                    visible += before
                    _think_buf = ""
                    _in_think = True
                elif not "<think>".startswith(_think_buf):
                    # No partial match — flush buffer as safe visible text
                    visible += _think_buf
                    _think_buf = ""
            else:
                # Inside think block — look for closing tag
                _think_buf += ch
                if "</think>" in _think_buf:
                    # Discard everything up to and including </think>
                    _think_buf = _think_buf[_think_buf.index("</think>") + len("</think>"):]
                    _in_think = False
                    # Anything left in buf after the tag is visible
                    visible += _think_buf
                    _think_buf = ""
            i += 1
        if visible:
            print(visible, end="", flush=True)
        return visible

    try:
        stream = ollama_client.chat(
            model   = LLM_MODEL,
            messages= messages,
            stream  = True,
            think   = think,
            options = opts,
        )
        for chunk in stream:
            # chunk is an ollama ChatResponse object
            msg = chunk.message
            content = msg.content if msg else ""
            if content:
                visible = _print_filtered(content)
                full_text  += visible
                word_count += len(visible.split())

            # The final chunk carries eval metrics
            if getattr(chunk, "done", False):
                eval_count  = getattr(chunk, "eval_count",    0) or 0
                eval_dur_ns = getattr(chunk, "eval_duration", 0) or 0

    except Exception as e:
        print(f"\n[ERROR] Ollama call failed: {e}")
        print("Make sure Ollama is running and the model is pulled:")
        print(f"  ollama pull {LLM_MODEL}")
        return ""

    elapsed = time.perf_counter() - t_start

    # Prefer Ollama's own eval_count; fall back to word count if unavailable
    tok  = eval_count if eval_count > 0 else word_count
    # Prefer Ollama's eval_duration (ns → s); fall back to wall-clock time
    secs = (eval_dur_ns / 1e9) if eval_dur_ns > 0 else elapsed
    tps  = tok / secs if secs > 0 else 0.0

    thinking_note = " [think=ON]" if think else " [think=OFF, faster]"
    print(f"\n\n{'─'*60}")
    print(f"  ⚡ {tok} tokens | {elapsed:.1f}s wall | {tps:.1f} tok/s{thinking_note}")
    print(f"{'─'*60}\n")
    return full_text

def _stream_gemini(
    messages    : list[dict],
    label       : str  = "Response",
) -> str:
    """
    Stream a response from Gemini API instead of Ollama.
    """
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}\n")

    full_text    = ""
    word_count   = 0
    t_start      = time.perf_counter()

    try:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            print("[WARNING] python-dotenv not installed. If using a .env file, please run: pip install python-dotenv")

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("[ERROR] GEMINI_API_KEY environment variable not set. Please set it in the .env file to use Gemini API.")
            return ""
            
        genai.configure(api_key=api_key)
        
        sys_prompt = ""
        user_prompt = ""
        for m in messages:
            if m["role"] == "system":
                sys_prompt += m["content"] + "\n"
            elif m["role"] == "user":
                user_prompt += m["content"] + "\n"
                
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=sys_prompt.strip()
        )
        
        response = model.generate_content(user_prompt.strip(), stream=True)
        for chunk in response:
            text = chunk.text
            if text:
                print(text, end="", flush=True)
                full_text += text
                word_count += len(text.split())

    except Exception as e:
        print(f"\n[ERROR] Gemini API call failed: {e}")
        return ""

    elapsed = time.perf_counter() - t_start
    tok  = int(word_count * 1.3) # Approximate tokens
    secs = elapsed
    tps  = tok / secs if secs > 0 else 0.0

    print(f"\n\n{'─'*60}")
    print(f"  ⚡ {tok} tokens | {elapsed:.1f}s wall | {tps:.1f} tok/s")
    print(f"{'─'*60}\n")
    return full_text


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_summary(log: dict, json_path: str, force: bool = False) -> str:
    """
    Generate (or load cached) a natural-language video summary.
    The summary is cached as <base>_summary.txt alongside the JSON.
    """
    summary_path = os.path.splitext(json_path)[0] + "_summary.txt"

    if os.path.exists(summary_path) and not force:
        with open(summary_path, "r", encoding="utf-8") as f:
            cached = f.read()
        print(f"  ✓ Loaded cached summary from {summary_path}")
        return cached

    session = log.get("session", {})
    summary = log.get("summary", {})

    video      = session.get("video", "unknown")
    duration   = session.get("duration_sec", 0)
    fps        = session.get("fps", 25)
    resolution = session.get("resolution", [0, 0])
    device     = session.get("device", "cpu")

    action_dist   = summary.get("action_distribution", {})
    total_recog   = summary.get("total_recognitions", 0)
    unique_tracks = summary.get("unique_tracks", [])
    per_track     = summary.get("per_track_summary", {})
    segments      = summary.get("action_segments", [])
    interactions  = summary.get("interaction_events", [])

    dist_lines = "\n".join(
        f"  - {lbl}: {cnt}" for lbl, cnt in action_dist.items()
    ) or "  (none)"

    track_lines = ""
    for tid, info in per_track.items():
        dom = info.get("dominant_action", "unknown")
        top3 = "; ".join(
            f"{l} ({c:.0%})" for l, c in info.get("top3", [])
        )
        track_lines += f"  Track {tid}: dominant='{dom}', top-3=[{top3}]\n"

    seg_lines = ""
    for s in segments[:20]:      # cap at 20 to stay within context
        seg_lines += (
            f"  Track {s['track']}: '{s['action']}' "
            f"({s['t_start']}s–{s['t_end']}s, avg conf {s['avg_conf']:.0%})\n"
        )

    int_lines = ""
    for e in interactions:
        int_lines += (
            f"  Tracks {e['track_a']} & {e['track_b']}: interacting "
            f"({e['t_start']}s–{e['t_end']}s, dominant action: '{e['dominant_action']}')\n"
        )

    context_block = f"""
VIDEO: {video}
Duration: {duration}s | FPS: {fps} | Resolution: {resolution[0]}x{resolution[1]} | Device: {device}
Unique tracks: {unique_tracks}
Total action recognitions: {total_recog}

ACTION DISTRIBUTION:
{dist_lines}

PER-TRACK SUMMARIES:
{track_lines or '  (none)'}

ACTION SEGMENTS (chronological):
{seg_lines or '  (none)'}

INTERACTION EVENTS:
{int_lines or '  (none)'}
""".strip()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Based on the structured action recognition data below from a "
                "patient-caregiver monitoring video, write a DETAILED, well-structured "
                "natural-language summary of what happened.\n\n"
                "Your summary MUST include:\n"
                "  1. Who was present (list all Track IDs and when they appeared)\n"
                "  2. A chronological narrative of each person's activity with timestamp "
                "     ranges — merge consecutive identical actions into one smooth sentence\n"
                "  3. All interactions between people (who, when, how long, what action)\n"
                "  4. Any notable medical/clinical events — mark these WARNING\n"
                "  5. Action frequency statistics (which action dominated and for how long)\n"
                "  6. Overall clinical assessment of the monitoring session\n\n"
                "Use temporal smoothing: if a person performs the same action for several "
                "consecutive seconds, describe it as ONE continuous activity with a duration, "
                "not as multiple separate events.\n\n"
                f"DATA:\n{context_block}"
            ),
        },
    ]

    if USE_GEMINI:
        print("\n📝 Generating video summary via Gemini API…")
        result = _stream_gemini(
            messages,
            label       = "VIDEO SUMMARY"
        )
    else:
        print("\n📝 Generating video summary via local LLM…")
        result = _stream_llm(
            messages,
            label       = "VIDEO SUMMARY",
            think       = False,
            llm_options = LLM_OPTIONS_SUMMARY,
        )

    if result:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"  ✓ Summary saved → {summary_path}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# RAG QUERY ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def query_rag(
    question    : str,
    collection  : chromadb.Collection,
    embed_model : SentenceTransformer,
    summary_text: str,
    log         : dict,
) -> str:
    """
    Retrieve relevant chunks from ChromaDB and answer the question via LFM-2.5-Thinking.
    """
    # Embed the query and retrieve top-k chunks
    q_embed = embed_model.encode([question], show_progress_bar=False).tolist()
    results = collection.query(
        query_embeddings = q_embed,
        n_results        = min(TOP_K, collection.count()),
        include          = ["documents", "metadatas", "distances"],
    )

    retrieved_docs = results.get("documents", [[]])[0]
    retrieved_meta = results.get("metadatas", [[]])[0]
    distances      = results.get("distances",  [[]])[0]

    if not retrieved_docs:
        print("[WARN] No relevant chunks found in the vector store.")
        return ""

    # Build context block from retrieved chunks (ordered by relevance)
    context_parts = []
    for doc, meta, dist in zip(retrieved_docs, retrieved_meta, distances):
        relevance = 1 - dist   # cosine distance → similarity
        context_parts.append(f"[relevance {relevance:.2f}] {doc}")

    context_text = "\n\n".join(context_parts)

    session     = log.get("session", {})
    summary_sec = log.get("summary", {})
    video       = session.get("video", "unknown")
    duration    = session.get("duration_sec", 0)
    tracks      = summary_sec.get("unique_tracks", [])

    # Include action distribution + capped segments as hard facts
    # Capping segments keeps the prompt inside num_ctx and reduces LLM RAM usage.
    action_dist  = summary_sec.get("action_distribution", {})
    all_segments = summary_sec.get("action_segments", [])
    per_track    = summary_sec.get("per_track_summary", {})

    MAX_SEG_IN_PROMPT = 30   # cap to avoid blowing the context window on CPU

    dist_str = "  " + "\n  ".join(
        f"{lbl}: {cnt} recognition(s)" for lbl, cnt in action_dist.items()
    ) if action_dist else "  (none)"

    seg_str = "  " + "\n  ".join(
        f"Track {s['track']}: '{s['action']}' {s['t_start']}s–{s['t_end']}s "
        f"(avg conf {s['avg_conf']:.0%})"
        for s in all_segments[:MAX_SEG_IN_PROMPT]
    ) if all_segments else "  (none)"
    if len(all_segments) > MAX_SEG_IN_PROMPT:
        seg_str += f"\n  … ({len(all_segments) - MAX_SEG_IN_PROMPT} more segments)"

    pt_str = "  " + "\n  ".join(
        f"Track {tid}: dominant='{info['dominant_action']}', "
        f"top-3={[f'{l} ({c:.0%})' for l,c in info.get('top3',[])]}"
        for tid, info in per_track.items()
    ) if per_track else "  (none)"

    hard_facts = (
        f"ACTION DISTRIBUTION:\n{dist_str}\n\n"
        f"ACTION SEGMENTS (capped at {MAX_SEG_IN_PROMPT}):\n{seg_str}\n\n"
        f"PER-TRACK PROFILES:\n{pt_str}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"VIDEO: {video} | Duration: {duration}s | Tracks present: {tracks}\n\n"
                f"VIDEO SUMMARY (pre-generated):\n{summary_text}\n\n"
                f"{hard_facts}\n\n"
                f"TOP RETRIEVED CONTEXT CHUNKS (ranked by relevance):\n{context_text}\n\n"
                f"USER QUESTION: {question}\n\n"
                "Instructions: Answer the question in detail using ALL the data above. "
                "Apply temporal smoothing — consecutive identical actions on the same track "
                "should be described as a single continuous activity with a time range. "
                "Cite timestamp ranges (start–end) not just individual frames. "
                "List all relevant actions with their track, time range, confidence, and category."
            ),
        },
    ]

    short_q = question[:60] + "…" if len(question) > 60 else question
    if USE_GEMINI:
        return _stream_gemini(
            messages,
            label       = f"ANSWER: {short_q}"
        )
    else:
        return _stream_llm(
            messages,
            label       = f"ANSWER: {short_q}",
            think       = False,
            llm_options = LLM_OPTIONS_QUERY,
        )


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE CLI
# ══════════════════════════════════════════════════════════════════════════════

def interactive_loop(
    collection  : chromadb.Collection,
    embed_model : SentenceTransformer,
    summary_text: str,
    log         : dict,
):
    """Run an interactive question-answer loop."""
    video   = log.get("session", {}).get("video", "unknown")
    dur     = log.get("session", {}).get("duration_sec", 0)
    tracks  = log.get("summary", {}).get("unique_tracks", [])

    print("\n" + "═" * 60)
    print("  P.R.I.S.M. RAG ASSISTANT — Interactive Mode")
    print("═" * 60)
    print(f"  Video   : {video}")
    print(f"  Duration: {dur}s")
    print(f"  Tracks  : {tracks}")
    print(f"  Model   : {LLM_MODEL}")
    print(f"  Chunks  : {collection.count()} indexed")
    print("─" * 60)
    print("  Type your question, or 'summary' to re-show the summary.")
    print("  Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("❓ Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n✓ Session ended.")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("✓ Session ended.")
            break

        if question.lower() in ("summary", "show summary"):
            print(f"\n{'─'*60}\n  CACHED SUMMARY\n{'─'*60}\n{summary_text}\n")
            continue

        query_rag(question, collection, embed_model, summary_text, log)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global LLM_MODEL, TOP_K   # declare first, before any reference to these names

    parser = argparse.ArgumentParser(
        description="P.R.I.S.M. RAG pipeline — video action log Q&A",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--json", required=True,
        help="Path to the action log JSON produced by mediapipe_inference.py",
    )
    parser.add_argument(
        "--regenerate", action="store_true",
        help="Force regenerate the summary and rebuild the vector DB",
    )
    parser.add_argument(
        "--query", default=None,
        help="Single query to run (skips interactive loop)",
    )
    parser.add_argument(
        "--embed-model", default=EMBED_MODEL,
        help=f"Sentence-transformers embedding model (default: {EMBED_MODEL})",
    )
    parser.add_argument(
        "--llm", default=LLM_MODEL,
        help=f"Ollama model name (default: {LLM_MODEL})",
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help=f"Number of chunks to retrieve per query (default: {TOP_K})",
    )
    args = parser.parse_args()

    # Apply user-supplied overrides to module-level globals
    LLM_MODEL = args.llm
    TOP_K     = args.top_k

    # ── Load JSON ────────────────────────────────────────────────────────────
    json_path = args.json
    if not os.path.exists(json_path):
        sys.exit(f"ERROR: JSON file not found: {json_path}")

    print(f"\n🔍 Loading action log: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        log = json.load(f)

    video = log.get("session", {}).get("video", "unknown")
    print(f"   Video   : {video}")
    print(f"   Duration: {log.get('session', {}).get('duration_sec', '?')}s")
    print(f"   Tracks  : {log.get('summary', {}).get('unique_tracks', [])}")

    # ── Load embedding model ─────────────────────────────────────────────────
    print(f"\n📦 Loading embedding model: {args.embed_model}")
    t0 = time.perf_counter()
    embed_model = SentenceTransformer(
        args.embed_model,
        device="cpu",          # explicit CPU keeps it from probing CUDA unnecessarily
    )
    # Reduce inter-op threads to avoid over-subscription with the LLM
    import torch
    torch.set_num_threads(2)          # embedding threads (keep low)
    torch.set_num_interop_threads(1)  # prevent thread storms
    print(f"   ✓ Loaded in {time.perf_counter() - t0:.1f}s")

    # ── Build / load ChromaDB ────────────────────────────────────────────────
    print(f"\n🗄  Connecting to ChromaDB ({CHROMA_DIR})…")
    collection = build_or_load_db(
        log          = log,
        json_path    = json_path,
        embed_model  = embed_model,
        force_rebuild= args.regenerate,
    )

    # ── Generate / load summary ──────────────────────────────────────────────
    summary_text = generate_summary(log, json_path, force=args.regenerate)

    # ── Single query mode ────────────────────────────────────────────────────
    if args.query:
        query_rag(args.query, collection, embed_model, summary_text, log)
        return

    # ── Interactive loop ─────────────────────────────────────────────────────
    interactive_loop(collection, embed_model, summary_text, log)


if __name__ == "__main__":
    main()