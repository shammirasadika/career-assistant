# app.py
# Purpose: Glue everything together and launch a Gradio chat UI.
# Output format:
#   Answer:
#   <text with inline [n]>
#
#   --------------------------------
#
#   Sources:
#   [1] filename.pdf, p.X
#   [2] filename.pdf, p.Y
#   Salary Tool (CSV dataset), Samples used: K
#
# Ground Truth mode:
# - If the query matches an entry in ground_truth.json:
#     * Bypass LLM/RAG/tool
#     * Build Answer with inline [n] using answer_segments (or a single [1] if one citation)
#     * Sources: exactly the citations you provided (one line per source, always show page when provided)
# - Otherwise: normal routing (RAG/Tool/Both/LLM).
# - For RAG/BOTH routes, we post-process to RENumber/Filter so the Answer’s [n] match the Sources list 1:1.

import re
import json
import traceback
import yaml
import gradio as gr

from rag import RAGIndex, make_context
from llm import LLMProvider
from tools.salary_tool import SalaryTool
from prompts import (
    ROUTER_SYSTEM,           # kept (not used by heuristic router)
    ROUTER_USER_TEMPLATE,    # kept (not used)
    ASSISTANT_SYSTEM,
    ANSWER_USER_TEMPLATE,
)

# ---------------------------
# Config helpers (safe defaults)
# ---------------------------

def load_cfg():
    with open("config.yml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

CFG = load_cfg()

# safe accessors with defaults
retr_cfg = CFG.get("retrieval", {})
ui_cfg   = CFG.get("ui", {})
llm_cfg  = CFG.get("llm", {})
tool_cfg = CFG.get("tooling", {})
gt_cfg   = CFG.get("ground_truth", {})

INDEX_PATH   = retr_cfg.get("index_path", "data/index.pkl")
TOP_K        = int(retr_cfg.get("top_k", 5))
PROJECT_NAME = CFG.get("project_name", "Career Assistant")

LLM_PROVIDER = llm_cfg.get("provider", "openai")
LLM_MODEL    = llm_cfg.get("model", "gpt-4o-mini")
LLM_TEMP     = float(llm_cfg.get("temperature", 0.2))
LLM_MAXTOK   = int(llm_cfg.get("max_tokens", 800))

SAL_BACKEND  = tool_cfg.get("salary_backend", "dummy")   # fallback to dummy
SAL_COUNTRY  = tool_cfg.get("salary_country", "au")      # default Australia

GT_ENABLED   = bool(gt_cfg.get("enabled", True))
GT_PATH      = gt_cfg.get("path", "ground_truth.json")

# Thresholds
SCORE_THRESHOLD   = float(retr_cfg.get("score_threshold", 0.30))   # routing threshold
INCLUDE_THRESHOLD = float(retr_cfg.get("include_threshold", 0.15)) # include-in-prompt threshold

# ---------------------------
# Initialize services (lazy RAG so UI still loads if index missing)
# ---------------------------

_RAG = None

def get_rag():
    """
    Returns either a RAGIndex instance or an Exception object (if init failed).
    """
    global _RAG
    if _RAG is None:
        try:
            _RAG = RAGIndex(INDEX_PATH, top_k=TOP_K)
        except Exception as e:
            _RAG = e
    return _RAG

LLM = LLMProvider(
    provider=LLM_PROVIDER,
    model=LLM_MODEL,
    temperature=LLM_TEMP,
    max_tokens=LLM_MAXTOK,
)

SAL = SalaryTool(backend=SAL_BACKEND, country=SAL_COUNTRY)

# ---------------------------
# Ground Truth helpers
# ---------------------------

def _safe_load_ground_truth(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or []
    except Exception:
        return []

GROUND_TRUTH = _safe_load_ground_truth(GT_PATH) if GT_ENABLED else []

def _norm(s: str) -> str:
    return " ".join((s or "").lower().strip().split())

def _find_ground_truth_entry(query: str):
    if not (GT_ENABLED and GROUND_TRUTH):
        return None
    qn = _norm(query)
    for item in GROUND_TRUTH:
        for m in item.get("match", []):
            if _norm(m) == qn:
                return item
    return None

def _render_ground_truth_answer(item: dict) -> str:
    """
    Ground-truth mode:
    - If type == "tool": print fixed salary result & tool source.
    - Else (RAG): If 'answer_segments' present, build Answer with inline [n] that
      match 'citations' by index (1-based). Else use 'answer'; if exactly one
      citation exists, append [1].
    - Sources: list all citations in order as [n] lines (pages shown when provided).
    """
    # Tool ground-truth
    if (item.get("type") or "").lower() == "tool":
        tr = item.get("tool_result", {}) or {}
        title = (tr.get("title") or "(unknown)").title()
        loc   = tr.get("location") or "(unknown)"
        avg   = tr.get("average")
        cur   = tr.get("currency") or ""
        samples = tr.get("samples", "?")
        avg_str = f"{avg:,.0f}" if isinstance(avg, (int, float)) else str(avg or "N/A")
        answer = f"Average salary for {title} in {loc}: {avg_str} {cur}".strip()
        body = [
            f"Answer:\n{answer}",
            "",
            "--------------------------------",
            "",
            "Sources:",
            f"Salary Tool (CSV dataset), Samples used: {samples}"
        ]
        return "\n".join(body).strip()

    # RAG ground-truth
    raw_cits = item.get("citations", []) or []
    sources_lines = []
    for i, c in enumerate(raw_cits, start=1):
        src = (c.get("source") or "").strip()
        page = c.get("page")
        if not src:
            continue
        if page not in (None, "", 0):
            sources_lines.append(f"[{i}] {src}, p.{page}")
        else:
            sources_lines.append(f"[{i}] {src}")

    segments = item.get("answer_segments") or []
    if segments:
        parts = []
        for seg in segments:
            txt = (seg.get("text") or "").strip()
            idx = seg.get("cite_index")
            if isinstance(idx, int) and 1 <= idx <= len(raw_cits):
                if txt and not txt.endswith(('.', '!', '?')):
                    txt += "."
                parts.append(f"{txt} [{idx}]")
            else:
                parts.append(txt)
        answer = " ".join([p for p in parts if p]).strip()
    else:
        answer = (item.get("answer") or "").strip()
        if len(sources_lines) == 1 and answer:
            if not answer.endswith(('.', '!', '?')):
                answer += "."
            answer += " [1]"
        if not answer:
            answer = "No ground-truth answer provided."

    body = [
        f"Answer:\n{answer}",
        "",
        "--------------------------------",
        "",
        "Sources:",
        "\n".join(sources_lines) if sources_lines else "LLM (general knowledge)"
    ]
    return "\n".join(body).strip()

# ---------------------------
# Router logic (decide RAG | Tool | BOTH | LLM)
# ---------------------------

def route_query(q: str, rag_snippets, score_threshold: float) -> str:
    """
    Decide between: 'rag' | 'tool' | 'both' | 'llm'
    """
    include_threshold = float(retr_cfg.get("include_threshold", INCLUDE_THRESHOLD))
    qn = " ".join((q or "").lower().strip().split())

    salary_words = {
        "salary","pay","wage","compensation","income","remuneration","earn",
        "average pay","pay scale","salary range","comp","band","range","earnings"
    }
    rag_intent_words = {
        "task","tasks","responsibilit","dutie","duty","skill","skills","according",
        "document","docs","osca","knowledge","requirements","role","roles","job description","jd",
        "what does","according to the documents","according to osca","responsibilities","duties"
    }

    has_salary_intent = any(w in qn for w in salary_words)
    has_rag_intent    = any(w in qn for w in rag_intent_words)

    best = max((s.get("score", 0.0) for s in (rag_snippets or [])), default=0.0)
    has_any_rag = best >= include_threshold
    has_confident_rag = best >= score_threshold

    explicit_doc_ask = ("according to the documents" in qn) or ("according to osca" in qn) \
                       or ("what does" in qn) or ("responsibilities" in qn) or ("duties" in qn)

    if has_salary_intent and has_rag_intent:
        return "both" if (explicit_doc_ask or has_any_rag) else "tool"

    if has_salary_intent:
        return "tool"

    if has_rag_intent and has_confident_rag:
        return "rag"

    return "llm"

# ---------------------------
# Sanitizers / helpers (keep Answer clean)
# ---------------------------

_SOURCES_BLOCK_RE = re.compile(r"(?im)^\s*Sources\s*:.*?(?=(\n^\S|$\Z))", flags=re.DOTALL | re.MULTILINE)
_TOOLRESULT_RE    = re.compile(r"(?im)^\s*Tool\s*result\s*:.*?(?=(\n^\S|$\Z))", flags=re.DOTALL | re.MULTILINE)
_NOTE_BLOCK_RE    = re.compile(r"(?im)^\s*Note\s*:\s*(.*?)\s*(?=(\n^\S|$\Z))", flags=re.DOTALL | re.MULTILINE)

_BRACKET_REF_LINE_RE   = re.compile(r"(?m)^\s*\[\d+\][^\r\n]*[\r]?\n?")
_BRACKET_REF_INLINE_RE = re.compile(r"\[\d+\]")
_STANDALONE_BRACKET_RE = re.compile(r"(?m)^\s*\[\d+\]\s*$")

_TOOL_BACKEND_CHATTER_RE = re.compile(r"(?i)\baccording to (the )?tool result(s)?(?: from the backend)?[^\n]*")
_BACKEND_LABEL_RE        = re.compile(r"(?im)\bbackend\b\s*:\s*[^\n]*")
_NONE_LINE_RE            = re.compile(r"(?im)^\s*\(none\)\s*$")
_NONE_SOLO_LINE_RE       = re.compile(r"(?im)^\s*None\s*$")
_DANGLING_NONE_PAREN_RE  = re.compile(r"(?m)\bNone\s*\(")

_NO_PAGE_PAREN_RE = re.compile(
    r"\(\s*(?:no\s+page\s+number\s+(?:not\s+)?(?:provided|specified|given)|no\s+specific\s+page\s+(?:mentioned|provided|given)|no\s+page\s+info|no\s+page[^)]*)\s*\)",
    re.IGNORECASE
)

def _normalize_text(s: str) -> str:
    return re.sub(r"[, \t]+", "", s or "").lower()

def _strip_salary_duplicates_from_answer(text: str, tr: dict, tool_result_md: str) -> str:
    """
    Remove salary lines echoed by the LLM so we only show our clean tool_result_md once.
    """
    if not text:
        return text

    title = (tr.get("title") or "").strip()
    location = (tr.get("location") or "").strip()

    if location.upper() == "VIC":
        loc_alt = r"(?:VIC|Victoria)"
    elif location.upper() == "NSW":
        loc_alt = r"(?:NSW|New South Wales)"
    else:
        loc_alt = re.escape(location) if location else r".*"

    num_re = r"\d{2,3}(?:[,\s]?\d{3})*(?:\.\d+)?"
    cur_re = r"(?:AUD|Australian\s+Dollars)?"
    hint_re = r"(?i)\b(?:average|median|typical)\s+(?:salary|pay|compensation)\b"
    title_re = re.escape(title) if title else r".+?"

    p1 = re.compile(
        rf"{hint_re}.*?\b(?:for\s+a\s+)?{title_re}\b(?:.*?\bin\s+{loc_alt}\b)?[^.\d]*?(?:is|:)\s*{num_re}\s*{cur_re}\b[,]?",
        re.IGNORECASE,
    )
    p2 = re.compile(
        rf"{hint_re}.*?\b(?:for\s+a\s+)?{title_re}\b.*?{num_re}\s*{cur_re}\b[,]?",
        re.IGNORECASE,
    )
    p3 = re.compile(
        rf"{hint_re}.*?{num_re}\s*{cur_re}\b[,]?",
        re.IGNORECASE,
    )

    tool_norm = _normalize_text(tool_result_md)

    cleaned_lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if tool_result_md and _normalize_text(s) == tool_norm:
            continue
        if title and (p1.search(s) or p2.search(s)):
            continue
        if tool_result_md and p3.search(s):
            continue

        cleaned_lines.append(ln)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned

def _strip_and_collect_notes_from_answer(text: str, keep_inline_brackets: bool):
    """
    Sanitize LLM answer body and collect any 'Note:' lines to merge later.
    If keep_inline_brackets=True (RAG/BOTH), we KEEP inline [n] markers.
    Returns (cleaned_text, extracted_notes_list).
    """
    if not text:
        return text, []

    cleaned = text
    cleaned = _SOURCES_BLOCK_RE.sub("", cleaned)
    cleaned = _TOOLRESULT_RE.sub("", cleaned)

    notes_found = []
    def _collect_note(m):
        body = (m.group(1) or "").strip()
        if body:
            notes_found.append(body)
        return ""
    cleaned = _NOTE_BLOCK_RE.sub(_collect_note, cleaned)

    cleaned = _BRACKET_REF_LINE_RE.sub("", cleaned)
    cleaned = _NO_PAGE_PAREN_RE.sub("", cleaned)
    cleaned = re.sub(
        r"\[\d+\]\s*\(\s*(?:no\s+page\s+number\s+(?:not\s+)?(?:provided|specified|given)|no\s+specific\s+page\s+(?:mentioned|provided|given)|no\s+page\s+info|no\s+page[^)]*)\s*\)",
        "",
        cleaned, flags=re.IGNORECASE
    )

    if not keep_inline_brackets:
        cleaned = _BRACKET_REF_INLINE_RE.sub("", cleaned)

    cleaned = _TOOL_BACKEND_CHATTER_RE.sub("", cleaned)
    cleaned = _BACKEND_LABEL_RE.sub("", cleaned)
    cleaned = _NONE_LINE_RE.sub("", cleaned)
    cleaned = _NONE_SOLO_LINE_RE.sub("", cleaned)
    cleaned = _DANGLING_NONE_PAREN_RE.sub("", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\s+\)$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\(no specific page\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    cleaned = _STANDALONE_BRACKET_RE.sub("", cleaned)

    return cleaned, notes_found

def _drop_stray_none_lines(lines):
    out = []
    for line in lines:
        if line is None:
            continue
        s = str(line).strip()
        if not s or s.lower() == "none" or s == "(none)":
            continue
        out.append(s)
    return out

# ---------------------------
# Sources block builder (docs/tools cited only here)
# ---------------------------

def _format_sources_block(route: str, filt_for_prompt: list, tool_ok: bool, tr: dict) -> str:
    """
    One line per (source, page) in the SAME order used in context.
    With page:  ", p.X"
    Without:    just filename
    """
    sources_lines = []

    # Documents first
    if route in ("rag", "both") and filt_for_prompt:
        seen = set()
        numbered = []
        for s in filt_for_prompt:
            src = (s.get("source") or "").strip()
            page = s.get("page")
            key = (src, page)
            if not src or key in seen:
                continue
            seen.add(key)
            if page not in (None, "", 0):
                numbered.append(f"{src}, p.{page}")
            else:
                numbered.append(f"{src}")
        for i, entry in enumerate(numbered, start=1):
            sources_lines.append(f"[{i}] {entry}")

    # Tool source line (never numbered)
    if route in ("tool", "both") and tool_ok:
        samples = tr.get("samples", "?") if tr else "?"
        sources_lines.append(f"Salary Tool (CSV dataset), Samples used: {samples}")

    # Fallback
    if not sources_lines:
        sources_lines.append("LLM (general knowledge)")

    return "Sources:\n" + "\n".join(sources_lines)

# ---------------------------
# Renumbering helper (Answer + Sources match exactly)
# ---------------------------

def renumber_used_markers(answer_text: str, sources_lines: list[str]) -> tuple[str, list[str]]:
    """
    Map whatever [n] markers appear in the answer to a compact 1..K sequence,
    in order of first appearance. Then keep only those sources (in the same
    first-use order) and renumber their labels.

    Input:
      - answer_text: the LLM answer containing [n] markers (e.g., [1], [3], [3], [2])
      - sources_lines: ONLY the numbered document lines, e.g. ["[1] file, p.1", "[2] file, p.2", ...]

    Output:
      - new_answer: with markers remapped to [1]..[K]
      - new_sources_core: the filtered/renumbered list (K lines), WITHOUT leading [n]
                          (caller will add the new [n] labels back)
    """
    if not answer_text or not sources_lines:
        return answer_text, []

    used_old_nums = []
    mapping = {}

    def _swap(m):
        old = int(m.group(1))
        if old not in mapping:
            mapping[old] = len(mapping) + 1
            used_old_nums.append(old)
        return f"[{mapping[old]}]"

    # Renumber the answer markers to [1..K]
    new_answer = re.sub(r"\[(\d+)\]", _swap, answer_text)

    # Build the filtered sources list in the same first-use order
    stripped = [re.sub(r"^\[(\d+)\]\s*", "", line).strip() for line in sources_lines]
    new_sources_core = []
    for old in used_old_nums:
        if 1 <= old <= len(stripped):
            new_sources_core.append(stripped[old - 1])

    return new_answer, new_sources_core

# ---------------------------
# Diversity helper — ensure multiple distinct pages so model can cite [1]..[n]
# ---------------------------

def _select_unique_pages(snippets, max_pages=4, include_threshold=INCLUDE_THRESHOLD):
    """
    Pick up to max_pages unique (source,page) entries from ranked snippets.
    Prefer items >= include_threshold, but allow a top-up to reach diversity.
    """
    if not snippets:
        return []

    ranked = sorted(snippets, key=lambda s: s.get("score", 0.0), reverse=True)

    seen = set()
    chosen = []

    # Pass 1: strong scores
    for s in ranked:
        key = ((s.get("source") or "").strip(), s.get("page"))
        if not key[0] or key in seen:
            continue
        if s.get("score", 0.0) >= include_threshold:
            chosen.append(s)
            seen.add(key)
            if len(chosen) >= max_pages:
                return chosen

    # Pass 2: top-up if fewer than max_pages
    for s in ranked:
        if len(chosen) >= max_pages:
            break
        key = ((s.get("source") or "").strip(), s.get("page"))
        if not key[0] or key in seen:
            continue
        chosen.append(s)
        seen.add(key)

    return chosen

# ---------------------------
# Simple renderer (single Sources header)
# ---------------------------

def _render_simple_output(answer_text: str,
                          route: str,
                          filt_for_prompt: list,
                          tool_ok: bool,
                          tr: dict,
                          notes_list: list) -> str:
    parts = []

    # 1) Answer
    answer = (answer_text or "").strip()
    if not answer:
        answer = "Sorry, I don’t have information for that right now."
    parts.append(f"Answer:\n{answer}\n\n--------------------------------")

    # 2) Sources (single header)
    parts.append(_format_sources_block(route, filt_for_prompt, tool_ok, tr))

    # 3) Note (single header, only if present)
    notes = [n.strip() for n in (notes_list or []) if n and n.strip()]
    if notes:
        parts.append("Note:\n" + " ".join(notes))

    return "\n\n".join(parts).strip()

# ---------------------------
# Main handler
# ---------------------------

def handle_query(q: str) -> str:
    try:
        # 0) Ground Truth override
        gt_item = _find_ground_truth_entry(q)
        if gt_item:
            return _render_ground_truth_answer(gt_item)

        # 1) Get RAG
        rag_obj = get_rag()
        if isinstance(rag_obj, Exception):
            snippets = []
            rag_unavailable = True
        else:
            snippets = rag_obj.retrieve(q)
            rag_unavailable = False

        # 2) Route
        route = route_query(q, snippets, SCORE_THRESHOLD)

        # 3) RAG context (best-effort for BOTH)
        filt_for_prompt = []
        context_md = ""
        if route in ("rag", "both"):
            filt_for_prompt = _select_unique_pages(
                snippets or [],
                max_pages=4,
                include_threshold=INCLUDE_THRESHOLD
            )
            if not filt_for_prompt and snippets:
                best = max(snippets, key=lambda x: x.get("score", 0.0))
                filt_for_prompt = [best]
            if filt_for_prompt:
                context_md = make_context(filt_for_prompt)

        # 4) Tool call
        tool_result_md, tr, tool_ok = "", None, False
        if route in ("tool", "both"):
            tr = SAL.lookup(q)
            tool_ok = bool(tr and tr.get("average") not in (None, "N/A", ""))
            if tool_ok:
                avg = tr.get("average", "N/A")
                avg_str = f"{avg:,.0f}" if isinstance(avg, (int, float)) else str(avg)
                tool_result_md = (
                    f"Average salary for {tr.get('title','(unknown)').title()} "
                    f"in {tr.get('location','(unknown)')}: {avg_str} {tr.get('currency','')}"
                )
            else:
                if route == "tool":
                    route = "rag" if context_md else "llm"

        # 5) Compose LLM answer
        answer_text = ""
        llm_notes = []

        if route in ("rag", "both", "llm"):
            user_msg = ANSWER_USER_TEMPLATE.format(
                query=q,
                context=context_md or "",
                tool_result=tool_result_md or ""
            )
            ans = LLM.chat([
                {"role": "system", "content": ASSISTANT_SYSTEM},
                {"role": "user", "content": user_msg},
            ])
            keep_inline = route in ("rag", "both")
            cleaned_ans, llm_notes = _strip_and_collect_notes_from_answer(
                ans, keep_inline_brackets=keep_inline
            )

            if (route in ("tool", "both")) and tool_ok and tool_result_md:
                cleaned_ans = _strip_salary_duplicates_from_answer(cleaned_ans, tr, tool_result_md)

            answer_text = cleaned_ans.strip()

        # 6) SAFEGUARD: Remove phantom [n] if no docs were actually retrieved
        if route in ("rag", "both") and not filt_for_prompt:
            answer_text = re.sub(r"\[\d+\]", "", answer_text)

        # 7) Append the canonical salary line exactly once (if tool used)
        if route in ("tool", "both") and tool_result_md:
            if _normalize_text(tool_result_md) not in _normalize_text(answer_text):
                answer_text = (answer_text + ("\n" if answer_text else "") + tool_result_md).strip()

        # 8) Notes
        notes = []
        if rag_unavailable:
            notes.append("RAG index unavailable — answered without document retrieval.")
        if route == "llm" and not context_md and not tool_result_md:
            notes.append("No high-confidence documents or salary data matched your query.")
        if (route in ("tool", "both")) and not tool_ok:
            notes.append("Salary tool returned no usable data for the given title/location.")
        if route in ("rag", "both") and filt_for_prompt and all(
            s.get("score", 0.0) < SCORE_THRESHOLD for s in filt_for_prompt
        ):
            notes.append("Included low-confidence document excerpts.")
        if route in ("rag", "both") and any(s.get("page") in (None, "", 0) for s in (filt_for_prompt or [])):
            notes.append("One or more document citations did not specify a page number.")

        # 9) Final render (pre-renumber)
        final = _render_simple_output(
            answer_text=answer_text,
            route=route,
            filt_for_prompt=filt_for_prompt or [],
            tool_ok=tool_ok,
            tr=tr,
            notes_list=notes + llm_notes,
        )

        # 10) Post-process for RAG/BOTH: make citations match Sources exactly
        if route in ("rag", "both"):
            m = re.search(r"Sources:\n([\s\S]*?)(?:\nNote:|\Z)", final)
            if m:
                src_lines = [ln.strip() for ln in m.group(1).strip().splitlines() if ln.strip()]
                numbered = [ln for ln in src_lines if re.match(r"^\[\d+\]\s", ln)]
                others   = [ln for ln in src_lines if not re.match(r"^\[\d+\]\s", ln)]

                if numbered and re.search(r"\[(\d+)\]", answer_text or ""):
                    # Renumber: feed the Answer & ONLY the numbered doc lines
                    new_ans, core_sources = renumber_used_markers(answer_text, numbered)
                    renum_block = [f"[{i}] {s}" for i, s in enumerate(core_sources, 1)]

                    # If the answer ended up with no markers, clear doc lines
                    if not re.search(r"\[(\d+)\]", new_ans):
                        renum_block = []

                    new_sources_block = "\n".join(renum_block + others) if (renum_block or others) else "LLM (general knowledge)"

                    # Replace Answer and Sources in final
                    final = re.sub(r"(?s)Answer:\n.*?\n\n--------------------------------",
                                   f"Answer:\n{new_ans}\n\n--------------------------------", final)
                    final = re.sub(r"(?s)Sources:\n[\s\S]*?(?=\nNote:|\Z)",
                                   "Sources:\n" + new_sources_block, final)

        return final

    except Exception:
        return "Sorry, something went wrong.\n\n```\n" + traceback.format_exc() + "\n```"

# ---------------------------
# UI (Gradio)
# ---------------------------

with gr.Blocks(title=PROJECT_NAME) as demo:
    header_md = f"## {PROJECT_NAME}\nAsk about tasks/skills (RAG) or salary (Tool)."
    rag_status = get_rag()
    if isinstance(rag_status, Exception):
        header_md += "\n\n> ⚠️ RAG index failed to load; answers will rely on salary tool and general reasoning."

    gr.Markdown(header_md)

    chat = gr.Chatbot(type="messages", height=420)
    inp  = gr.Textbox(placeholder="e.g., According to OSCA, what are the duties for a Software Engineer?")
    ask_btn = gr.Button("Ask", variant="primary")
    clr_btn = gr.Button("Clear")

    gr.Examples(
        examples=[
            "According to OSCA, what are the main duties, required qualification, and alternative titles for an ICT Quality Assurance Engineer?",
            "In OSCA, what duties and responsibilities are listed for a Software Engineer?",
            "What does OSCA say about the ICT Business Analyst role, including required skill level and tasks?",
            "What is the average salary for a Software Engineer in Australia?",
            "For a Data Analyst in Australia, what skills and salary?"
        ],
        inputs=inp,
    )

    def respond(history, message):
        out = handle_query(message)
        new_hist = list(history or [])
        new_hist.append({"role": "user", "content": message})
        new_hist.append({"role": "assistant", "content": out})
        return new_hist, ""

    ask_btn.click(respond, [chat, inp], [chat, inp])
    inp.submit(respond, [chat, inp], [chat, inp])
    clr_btn.click(lambda: [], None, chat)

if __name__ == "__main__":
    # Defaults can be overridden in config.yml (ui.share, ui.server_port, ui.server_name).
    demo.launch(
        share=ui_cfg.get("share", False),
        server_port=int(ui_cfg.get("server_port", 7860)),
        # server_name=ui_cfg.get("server_name", "0.0.0.0"),
    )
