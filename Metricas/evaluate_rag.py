# -*- coding: utf-8 -*-
"""
evaluate_rag.py — Evaluación de QA con RAG (ES) + Telemetría
------------------------------------------------------------
Calcula, para cada modelo A/B y temperatura t ∈ {0.0, 0.2}:

- Métricas de QA por ítem (QID):
  Precision, Recall, F1, ROUGE1_F1, Groundedness, Hallucination,
  Recall@k, PrecCtx, Hit, lat_ms.

- Resúmenes agregados por modelo/temperatura.

- Estadística pareada A vs B (diseño dentro de ítem):
  t-test pareado, etc. para:
    F1, ROUGE1_F1, Groundedness, Hallucination, lat_ms,
    Recall@k, PrecCtx, Hit.

- Telemetría desde:
    metrics_usage-*.txt   (promedios por pregunta/run)
    all_usage-*.txt       (timeline segundo a segundo)

  Incluye P50/P95/mean/min/max de CPU, RAM, VRAM, GPU, temperatura +
  duty cycles (GPU>=90%, RAM>=80%, Temp>=80°C, VRAM>=0.9*capacidad).

Hipótesis resumen (para cada temperatura t, todas de dos colas):

  H1(t): ROUGE1_F1_A != ROUGE1_F1_B
  H2(t): Groundedness_A != Groundedness_B
  H3(t): lat_ms_A != lat_ms_B
  H4(t): GPU_MB_A != GPU_MB_B      (A usa distinta VRAM que B, por pregunta)

Los resultados de H1–H4 se escriben en:
  - hoja "hypothesis_0.0"
  - hoja "hypothesis_0.2"

Notas extra:
- Se sanitizan las etiquetas de modelo para que:
    * El nombre del Excel no use caracteres inválidos en Windows (como ':').
    * Los nombres de las hojas de Excel no contengan caracteres prohibidos
      ('[]:*?/\\') ni excedan 31 caracteres.
- P95_A_ms y P95_B_ms ahora se rellenan para TODAS las filas de hipótesis,
  incluida H4 (GPU_MB).
"""

import argparse
import json
import re
import unicodedata
import math
import inspect
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

# --------- SciPy (opcional para p-values) ---------
_HAS_SCIPY = False
try:
    import scipy.stats as stats  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# ------------------------ Helpers nombres ficheros / hojas Excel ------------------------

INVALID_FS_CHARS = r'<>:"/\\|?*'
EXCEL_INVALID_SHEET_CHARS = r'[]:*?/\\'


def safe_fs_component(name: str, max_len: int = 64) -> str:
    """
    Sanitiza una etiqueta para usarla en nombres de archivo (Windows-safe).
    Reemplaza caracteres inválidos por '_', colapsa '_' repetidos y recorta longitud.
    """
    if name is None:
        return "model"
    t = name.strip()
    if not t:
        return "model"
    t = t.replace(" ", "_")
    t = re.sub(f"[{re.escape(INVALID_FS_CHARS)}]", "_", t)
    t = re.sub(r"_+", "_", t)
    if len(t) > max_len:
        t = t[:max_len]
    return t or "model"


def safe_sheet_name(name: str) -> str:
    """
    Sanitiza nombres de hoja Excel:
    - Elimina caracteres no permitidos: []:*?/\\
    - Recorta a 31 caracteres.
    - Evita nombres vacíos.
    """
    if name is None:
        return "Sheet1"
    t = str(name)
    t = re.sub(f"[{re.escape(EXCEL_INVALID_SHEET_CHARS)}]", "_", t)
    t = t.strip()
    if not t:
        t = "Sheet1"
    if len(t) > 31:
        t = t[:31]
    return t

# ------------------------ Normalización & tokenización (ES) ------------------------

SPANISH_ARTICLES = {
    "el",
    "la",
    "los",
    "las",
    "un",
    "una",
    "unos",
    "unas",
    "lo",
    "al",
    "del",
}


def normalize_text(
    s: str,
    *,
    lowercase: bool = True,
    collapse_whitespace: bool = True,
    remove_punct: bool = True,
    remove_articles: bool = True,
    fold_accents: bool = False,
) -> str:
    if s is None:
        return ""
    # limpia colas tipo "[...]" y "(Ubicación: ...)"
    t = re.sub(r"\s*\[.*\]\s*$", "", s).strip()
    t = re.sub(r"\s*\(Ubicación:\s*[^)]+\)\s*$", "", t).strip()
    if lowercase:
        t = t.lower()
    if fold_accents:
        t = "".join(
            ch
            for ch in unicodedata.normalize("NFD", t)
            if unicodedata.category(ch) != "Mn"
        )
    if remove_punct:
        t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    tokens = t.split()
    if remove_articles:
        tokens = [tok for tok in tokens if tok not in SPANISH_ARTICLES]
    return " ".join(tokens) if collapse_whitespace else " ".join(tokens)


def word_tokens(s: str) -> List[str]:
    if not s:
        return []
    return re.findall(r"\w+", s, flags=re.UNICODE)

# ------------------------ Canonicalización de DocIDs ------------------------


def _alias_norm_key(s: str) -> str:
    """Normaliza un DocID para lookup de alias: minúsculas, sin acentos, solo [a-z0-9]."""
    if not s:
        return ""
    t = s.strip().lower()
    t = "".join(
        ch
        for ch in unicodedata.normalize("NFD", t)
        if unicodedata.category(ch) != "Mn"
    )
    t = re.sub(r"[^a-z0-9]", "", t)
    return t


CANON_MAP = {
    _alias_norm_key("Reglamento_General"): "REGLAMENTO_GENERAL_DE_ESTUDIOS_DE_PREGRADO_UV",
    _alias_norm_key("REGLAMENTO_GENERAL_DE_ESTUDIOS_DE_PREGRADO_UV"): "REGLAMENTO_GENERAL_DE_ESTUDIOS_DE_PREGRADO_UV",
    _alias_norm_key("Reglamento_ICI"): "Reglamento_ICI",
    _alias_norm_key("REGLAMENTO_ICI"): "Reglamento_ICI",
}


def canon_docid(docid: str) -> str:
    key = _alias_norm_key(docid)
    return CANON_MAP.get(key, docid.strip())

# ------------------------ Precisión / Recall / F1 ------------------------


def precision_recall_f1(
    pred: str, gold: str, fold_accents: bool = False
) -> Tuple[float, float, float]:
    pred_toks = word_tokens(normalize_text(pred, fold_accents=fold_accents))
    gold_toks = word_tokens(normalize_text(gold, fold_accents=fold_accents))
    if len(pred_toks) == 0 and len(gold_toks) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return 0.0, 0.0, 0.0
    pred_counts = Counter(pred_toks)
    gold_counts = Counter(gold_toks)
    common = sum((pred_counts & gold_counts).values())
    if common == 0:
        return 0.0, 0.0, 0.0
    prec = common / max(1, sum(pred_counts.values()))
    rec = common / max(1, sum(gold_counts.values()))
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def rouge1_f1(pred: str, gold: str, fold_accents: bool = False) -> float:
    pred_unigrams = word_tokens(normalize_text(pred, fold_accents=fold_accents))
    gold_unigrams = word_tokens(normalize_text(gold, fold_accents=fold_accents))
    if len(pred_unigrams) == 0 or len(gold_unigrams) == 0:
        return 0.0
    pred_count = Counter(pred_unigrams)
    gold_count = Counter(gold_unigrams)
    common = sum((pred_count & gold_count).values())
    prec = common / sum(pred_count.values()) if sum(pred_count.values()) > 0 else 0.0
    rec = common / sum(gold_count.values()) if sum(gold_count.values()) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

# ------------------------ Citas y oraciones ------------------------


def split_sentences_es(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[\.\?\!;])\s+", text.strip(), flags=re.UNICODE)
    return [p.strip() for p in parts if p.strip()]


def extract_refs(text: str, article_level: bool = False) -> List[str]:
    """
    Devuelve referencias únicas.
    - article_level=False -> solo DocIDs
    - article_level=True  -> "Doc:Art.N"
    Soporta: "[Doc:Art.4][Doc:Art.5]" y "Doc: [Art. 4, Art. 5]".
    """
    out: List[str] = []

    # Caso "[Doc:ArtX][Doc:ArtY]"
    full_matches = re.findall(r"\[([^:\]]+):([^\]]+)\]", text)
    if full_matches:
        for doc, arts_str in full_matches:
            doc = canon_docid(doc.strip())
            arts = re.findall(r"Art\.?\s*(\d+)", arts_str, flags=re.IGNORECASE)
            if article_level:
                if arts:
                    for art in arts:
                        out.append(f"{doc}:Art.{art}")
                else:
                    out.append(doc)
            else:
                out.append(doc)
        return list(dict.fromkeys(out))

    # Caso "Doc: [Art. 4, Art. 5] , Doc2: [Art. 19]"
    sections = re.findall(r"([^:\]]+):\s*\[([^\]]+)\]", text)
    for doc, arts_str in sections:
        doc = canon_docid(doc.strip())
        arts_list = re.split(r"\s*,\s*", arts_str.strip())
        arts = []
        for a in arts_list:
            m = re.search(r"Art\.?\s*(\d+)", a, flags=re.IGNORECASE)
            if m:
                arts.append(m.group(1))
        if article_level:
            if arts:
                for art in arts:
                    out.append(f"{doc}:Art.{art}")
            else:
                out.append(doc)
        else:
            out.append(doc)

    # Si nada coincidió, intenta detectar "[Doc]" suelto
    if not out:
        only_docs = re.findall(r"\[([^\]:]+)\]", text)
        for d in only_docs:
            out.append(canon_docid(d.strip()))

    return list(dict.fromkeys(out))

# ------------------------ Groundedness / Hallucination ------------------------


def groundedness(
    response_text: str,
    retrieved_refs: List[str],
    *,
    contexts_by_doc: Optional[Dict[str, str]] = None,
    min_overlap_tokens: int = 0,
    require_citation: bool = False,
    fold_accents: bool = False,
    article_level: bool = False,
) -> float:
    sents = split_sentences_es(response_text)
    if not sents:
        return 0.0

    # chequeo a nivel DocID
    retrieved_docs = set(canon_docid(r.split(":", 1)[0]) for r in retrieved_refs)
    concat_ctx = (
        " ".join(contexts_by_doc.get(d, "") for d in retrieved_docs)
        if contexts_by_doc
        else ""
    )
    ctoks = set(word_tokens(normalize_text(concat_ctx, fold_accents=fold_accents)))

    ok = 0
    for s in sents:
        cited_raw = extract_refs(s, article_level=article_level)
        cited_docs = set(canon_docid(c.split(":", 1)[0]) for c in cited_raw)
        if require_citation and not (cited_docs & retrieved_docs):
            continue
        if min_overlap_tokens > 0 and contexts_by_doc:
            stoks = set(word_tokens(normalize_text(s, fold_accents=fold_accents)))
            if len(stoks & ctoks) < min_overlap_tokens:
                continue
        ok += 1
    return ok / len(sents) if len(sents) > 0 else 0.0


def hallucination_rate(
    response_text: str, retrieved_refs: List[str], **kwargs
) -> float:
    return 1.0 - groundedness(response_text, retrieved_refs, **kwargs)

# ------------------------ Métricas de recuperación ------------------------


def retrieval_metrics(
    retrieved_refs: List[str], relevant_refs: List[str]
) -> Dict[str, Optional[float]]:
    # dedup preservando orden a nivel *ref completa* (Doc:Art.N si article_level=1)
    R = list(dict.fromkeys(retrieved_refs or []))
    T = list(dict.fromkeys(relevant_refs or []))
    Rset, Tset = set(R), set(T)
    if len(Tset) == 0:
        return {
            "Recall@k": None,
            "PrecCtx": None,
            "Hit": None,
            "GoldRelCount": 0,
            "RetrievedCount": len(R),
        }
    inter = len(Rset & Tset)
    recall_k = inter / len(Tset) if len(Tset) > 0 else None
    prec_ctx = inter / len(R) if len(R) > 0 else None
    hit = 1.0 if inter > 0 else 0.0
    return {
        "Recall@k": recall_k,
        "PrecCtx": prec_ctx,
        "Hit": hit,
        "GoldRelCount": len(Tset),
        "RetrievedCount": len(R),
    }

# ------------------------ Eficiencia (latencia simple) ------------------------


def latency_percentiles(latencies_ms: List[float]) -> Dict[str, float]:
    if not latencies_ms:
        return {"P50_ms": None, "P95_ms": None, "Mean_ms": None}
    arr = np.array(latencies_ms, dtype=float)
    return {
        "P50_ms": float(np.percentile(arr, 50)),
        "P95_ms": float(np.percentile(arr, 95)),
        "Mean_ms": float(np.mean(arr)),
    }

# ------------------------ Telemetría: metrics_usage-*.txt ------------------------


def parse_metrics_usage(path: str) -> pd.DataFrame:
    """
    Parsea archivos metrics_usage-*.txt generados por el harness.

    Formato esperado por bloque:
      -Prompt: <texto de la pregunta>
      real    1m15.348s
      user    0m0.007s
      sys     0m0.000s
      CPU avg (%): 44.27 | Mem (%): 3.5, Mem (KB): 1174748 | GPU (%): 100, GPU (MB): 14726, GPU (°C): 68
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()

    blocks = [b.strip() for b in txt.split("-Prompt:") if b.strip()]
    rows = []
    for b in blocks:
        lines = b.splitlines()
        prompt = lines[0].strip()

        real = user = sys_t = None
        cpu_avg = mem_pct = mem_kb = gpu_pct = gpu_mb = gpu_c = None

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            # real / user / sys  =>  XmY.YYs
            if line.startswith("real"):
                m = re.search(r"(\d+)m([\d\.]+)s", line)
                if m:
                    real = int(m.group(1)) * 60 + float(m.group(2))
            elif line.startswith("user"):
                m = re.search(r"(\d+)m([\d\.]+)s", line)
                if m:
                    user = int(m.group(1)) * 60 + float(m.group(2))
            elif line.startswith("sys"):
                m = re.search(r"(\d+)m([\d\.]+)s", line)
                if m:
                    sys_t = int(m.group(1)) * 60 + float(m.group(2))

            # línea de promedios
            elif line.startswith("CPU avg"):
                m = re.search(
                    r"CPU avg \(%\):\s*([\d\.]+)\s*\|\s*Mem \(%\):\s*([\d\.]+),\s*Mem \(KB\):\s*(\d+)\s*\|\s*GPU \(%\):\s*([\d\.]+),\s*GPU \(MB\):\s*(\d+),\s*GPU \(°C\):\s*(\d+)",
                    line,
                )
                if m:
                    cpu_avg = float(m.group(1))
                    mem_pct = float(m.group(2))
                    mem_kb = int(m.group(3))
                    gpu_pct = float(m.group(4))
                    gpu_mb = int(m.group(5))
                    gpu_c = int(m.group(6))

        rows.append(
            {
                "prompt": prompt,
                "real_s": real,
                "user_s": user,
                "sys_s": sys_t,
                "CPU_avg_pct": cpu_avg,
                "Mem_pct": mem_pct,
                "Mem_KB": mem_kb,
                "GPU_pct": gpu_pct,
                "GPU_MB": gpu_mb,
                "GPU_C": gpu_c,
            }
        )

    return pd.DataFrame(rows)


def summarize_metrics_usage_df(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Resumen por *run* de metrics_usage:
    mean / P50 / P95 / min / max de real/user/sys/CPU/Mem/GPU.
    """
    out: Dict[str, Any] = {}
    if df.empty:
        return out

    cols = [
        "real_s",
        "user_s",
        "sys_s",
        "CPU_avg_pct",
        "Mem_pct",
        "Mem_KB",
        "GPU_pct",
        "GPU_MB",
        "GPU_C",
    ]

    for col in cols:
        if col not in df.columns:
            continue
        vals = df[col].dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        out[f"{col}_mean"] = float(np.mean(vals))
        out[f"{col}_P50"] = float(np.percentile(vals, 50))
        out[f"{col}_P95"] = float(np.percentile(vals, 95))
        out[f"{col}_min"] = float(np.min(vals))
        out[f"{col}_max"] = float(np.max(vals))
    return out

# ------------------------ Telemetría: all_usage-*.txt (timeline) ------------------------


def parse_all_usage(path: str) -> pd.DataFrame:
    """
    Parsea archivos all_usage-*.txt (sample ~1s) con formato:

      -Prompt: <texto de la pregunta>
      60.9 | 2.7 895908 | 0\t14973\t 57

    Interpretación aproximada:
      CPU_pct | Mem_pct Mem_KB | GPU_pct  GPU_MB  GPU_C
    """
    rows = []
    current_prompt = None
    block_idx = -1
    sample_idx = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue

            if line.startswith("-Prompt:"):
                current_prompt = line.split(":", 1)[1].strip()
                block_idx += 1
                sample_idx = 0
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 3:
                continue

            # CPU %
            try:
                cpu_pct = float(parts[0])
            except ValueError:
                continue

            # Mem %
            mem_tokens = parts[1].split()
            if len(mem_tokens) < 2:
                continue
            try:
                mem_pct = float(mem_tokens[0])
                mem_kb = int(mem_tokens[1])
            except ValueError:
                continue

            # GPU %, GPU_MB, GPU_C
            gpu_tokens = [t for t in re.split(r"[\s\t]+", parts[2]) if t]
            if len(gpu_tokens) < 3:
                continue
            try:
                gpu_pct = float(gpu_tokens[0])
                gpu_mb = int(gpu_tokens[1])
                gpu_c = int(gpu_tokens[2])
            except ValueError:
                continue

            rows.append(
                {
                    "prompt_idx": block_idx,
                    "prompt": current_prompt,
                    "sample_idx": sample_idx,
                    "CPU_pct": cpu_pct,
                    "Mem_pct": mem_pct,
                    "Mem_KB": mem_kb,
                    "GPU_pct": gpu_pct,
                    "GPU_MB": gpu_mb,
                    "GPU_C": gpu_c,
                }
            )
            sample_idx += 1

    return pd.DataFrame(rows)


def summarize_timeline(
    df: pd.DataFrame,
    vram_capacity_mb: Optional[float] = None,
    gpu_busy_thresh: float = 90.0,
    vram_high_ratio: float = 0.9,
    ram_high_pct: float = 80.0,
    gpu_temp_high: float = 80.0,
) -> Dict[str, Any]:
    """
    Resumen global de la línea de tiempo:
    P50/P95/mean/max por CPU/Mem/GPU/VRAM/Temp + duty cycles.
    """
    out: Dict[str, Any] = {}
    if df.empty:
        return out

    for col in ["CPU_pct", "Mem_pct", "Mem_KB", "GPU_pct", "GPU_MB", "GPU_C"]:
        arr = df[col].to_numpy(dtype=float)
        out[f"{col}_P50"] = float(np.percentile(arr, 50))
        out[f"{col}_P95"] = float(np.percentile(arr, 95))
        out[f"{col}_mean"] = float(np.mean(arr))
        out[f"{col}_max"] = float(np.max(arr))

    # duty cycles (fracción de muestras sobre umbral)
    out["duty_GPU_ge_90"] = float((df["GPU_pct"] >= gpu_busy_thresh).mean())
    out["duty_RAM_ge_80"] = float((df["Mem_pct"] >= ram_high_pct).mean())
    out["duty_TEMP_ge_80"] = float((df["GPU_C"] >= gpu_temp_high).mean())

    if vram_capacity_mb is not None and vram_capacity_mb > 0:
        thr = vram_capacity_mb * vram_high_ratio
        out["duty_VRAM_ge_90cap"] = float((df["GPU_MB"] >= thr).mean())
    else:
        out["duty_VRAM_ge_90cap"] = float("nan")

    return out


def summarize_timeline_per_prompt(
    df: pd.DataFrame,
    vram_capacity_mb: Optional[float] = None,
    gpu_busy_thresh: float = 90.0,
    vram_high_ratio: float = 0.9,
    ram_high_pct: float = 80.0,
    gpu_temp_high: float = 80.0,
) -> pd.DataFrame:
    """
    Igual que summarize_timeline pero por prompt (para usar en tablas por ítem).
    """
    rows = []
    if df.empty:
        return pd.DataFrame()

    for pid, g in df.groupby("prompt_idx"):
        base = {
            "prompt_idx": pid,
            "prompt": g["prompt"].iloc[0],
        }
        s = summarize_timeline(
            g,
            vram_capacity_mb=vram_capacity_mb,
            gpu_busy_thresh=gpu_busy_thresh,
            vram_high_ratio=vram_high_ratio,
            ram_high_pct=ram_high_pct,
            gpu_temp_high=gpu_temp_high,
        )
        base.update(s)
        rows.append(base)
    return pd.DataFrame(rows)

# ------------------------ Parseo de contexts_from_golden ------------------------


def parse_gold_contexts(
    gold_data: Dict[str, str], article_level: bool = False
) -> Dict[str, Dict[str, Any]]:
    gold_by_qid: Dict[str, Dict[str, Any]] = {}
    for key, val in gold_data.items():
        m = re.match(r"^\[Q(\d+)\]\s*(.*)", val.strip(), re.DOTALL)
        if not m:
            continue
        qnum = m.group(1)
        qid = f"Q{qnum.zfill(2)}"
        gold_text_full = m.group(2).strip()
        gold_text = re.sub(
            r"\s*\(Ubicación:\s*[^)]+\)\s*$", "", gold_text_full
        ).strip()
        relevant_refs = extract_refs(key, article_level=article_level)
        gold_by_qid[qid] = {
            "gold_answer": gold_text,
            "relevant_refs": relevant_refs,
            "gold_context": val.strip(),
        }
    return gold_by_qid

# ------------------------ Evaluación por ítem / modelo ------------------------


def evaluate_record(
    record: Dict[str, Any],
    gold_info: Dict[str, Any],
    *,
    contexts_by_doc: Optional[Dict[str, str]] = None,
    min_overlap_tokens: int = 0,
    require_citation: bool = False,
    fold_accents: bool = False,
    article_level: bool = False,
) -> Dict[str, Any]:
    qid = record.get("qid")
    pred = (
        record.get("answ", "")
        or record.get("pred", "")
        or record.get("prediction", "")
        or record.get("response", "")
    )
    gold = gold_info.get("gold_answer", "")
    relevant_refs = gold_info.get("relevant_refs", [])
    # si no hay recuperación por ítem, usamos las refs relevantes como retrieved "ideal"
    retrieved_refs = relevant_refs
    lat_ms = (
        record.get("lat_ms", None)
        or record.get("latency_ms", None)
        or record.get("latency", None)
    )

    prec, rec, f1 = precision_recall_f1(pred, gold, fold_accents=fold_accents)
    rouge1 = rouge1_f1(pred, gold, fold_accents=fold_accents)
    gnd = groundedness(
        pred,
        retrieved_refs,
        contexts_by_doc=contexts_by_doc,
        min_overlap_tokens=min_overlap_tokens,
        require_citation=require_citation,
        fold_accents=fold_accents,
        article_level=article_level,
    )
    hal = 1.0 - gnd
    rdict = retrieval_metrics(retrieved_refs, relevant_refs)
    out = {
        "qid": qid,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROUGE1_F1": rouge1,
        "Groundedness": gnd,
        "Hallucination": hal,
        "lat_ms": float(lat_ms) if lat_ms is not None else None,
        **rdict,
    }
    return out


def evaluate_model(
    records: List[Dict[str, Any]],
    gold_by_qid: Dict[str, Dict[str, Any]],
    *,
    contexts_by_doc: Optional[Dict[str, str]] = None,
    min_overlap_tokens: int = 0,
    require_citation: bool = False,
    fold_accents: bool = False,
    article_level: bool = False,
) -> pd.DataFrame:
    rows = []
    missing_qids = []
    for r in records:
        qid = r.get("qid")
        if qid not in gold_by_qid:
            missing_qids.append(qid)
            continue
        gold_info = gold_by_qid[qid]
        ev = evaluate_record(
            r,
            gold_info,
            contexts_by_doc=contexts_by_doc,
            min_overlap_tokens=min_overlap_tokens,
            require_citation=require_citation,
            fold_accents=fold_accents,
            article_level=article_level,
        )
        rows.append(ev)
    if missing_qids:
        print(
            "[WARN] QIDs no en golden: "
            + ", ".join(str(x) for x in missing_qids if x is not None)
        )
    return pd.DataFrame(rows)


def summarize_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    keys = [
        "Precision",
        "Recall",
        "F1",
        "ROUGE1_F1",
        "Groundedness",
        "Hallucination",
        "Recall@k",
        "PrecCtx",
        "Hit",
    ]
    summary: Dict[str, Any] = {}
    for k in keys:
        if k in df.columns:
            vals = df[k].dropna()
            summary[f"mean_{k}"] = float(vals.mean()) if len(vals) > 0 else float("nan")
            summary[f"nan_count_{k}"] = int(df[k].isna().sum())
            summary[f"nan_pct_{k}"] = (
                float(df[k].isna().mean() * 100) if len(df) > 0 else float("nan")
            )
    if "lat_ms" in df.columns:
        vals = df["lat_ms"].dropna()
        if len(vals) > 0:
            summary.update(latency_percentiles(vals.tolist()))
        summary["nan_count_lat_ms"] = int(df["lat_ms"].isna().sum())
        summary["nan_pct_lat_ms"] = (
            float(df["lat_ms"].isna().mean() * 100) if len(df) > 0 else float("nan")
        )
    summary["n_items"] = int(len(df))
    return summary

# ------------------------ Estadística pareada & efecto ------------------------


def _paired_stats(
    a: np.ndarray, b: np.ndarray, alternative: str = "two-sided"
) -> Dict[str, Any]:
    """
    Estadística de comparación pareada A vs B.

    alternative:
      - 'two-sided' -> H0: medias iguales; H1: medias distintas.
      - 'greater'   -> H1: media_A > media_B.
      - 'less'      -> H1: media_A < media_B.
    """
    mask = ~np.isnan(a) & ~np.isnan(b)
    a = a[mask]
    b = b[mask]
    n = int(len(a))
    if n < 2:
        return {
            "n": n,
            "t": float("nan"),
            "p": float("nan"),
            "mean_A": float(np.nanmean(a) if n else float("nan")),
            "mean_B": float(np.nanmean(b) if n else float("nan")),
            "cohen_d": float("nan"),
            "mean_diff": float("nan"),
            "sd_diff": float("nan"),
        }
    diffs = a - b
    mean_A = float(np.mean(a))
    mean_B = float(np.mean(b))
    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1))
    t = mean_diff / (sd_diff / math.sqrt(n)) if sd_diff > 0 else float("nan")

    # Cohen's d para muestras pareadas
    d = mean_diff / sd_diff if sd_diff > 0 else float("nan")

    p = float("nan")
    if _HAS_SCIPY:
        try:
            if "alternative" in inspect.signature(stats.ttest_rel).parameters:
                res = stats.ttest_rel(a, b, nan_policy="omit", alternative=alternative)
                p = float(res.pvalue)
            else:
                res = stats.ttest_rel(a, b, nan_policy="omit")
                p2 = float(res.pvalue)
                if alternative == "two-sided":
                    p = p2
                elif alternative == "greater":
                    p = p2 / 2 if mean_diff > 0 else 1 - p2 / 2
                else:
                    p = p2 / 2 if mean_diff < 0 else 1 - p2 / 2
        except Exception:
            p = float("nan")

    return {
        "n": n,
        "t": t,
        "p": p,
        "mean_A": mean_A,
        "mean_B": mean_B,
        "cohen_d": d,
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
    }


def _direction_from_diff(mean_diff: float) -> str:
    """
    Direccionalidad práctica a partir del signo de (media_A - media_B).
    """
    if mean_diff is None or math.isnan(mean_diff):
        return "undetermined"
    if mean_diff > 0:
        return "A > B"
    if mean_diff < 0:
        return "A < B"
    return "A ≈ B"


def paired_diff_summary(
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
    on: str = "qid",
    metrics: List[str] = (
        "F1",
        "ROUGE1_F1",
        "Groundedness",
        "Hallucination",
        "lat_ms",
        "Recall@k",
        "PrecCtx",
        "Hit",
    ),
    control_shared: List[str] = ("Recall@k", "PrecCtx", "Hit"),
) -> pd.DataFrame:
    merged = pd.merge(
        dfA[[on] + list(metrics)],
        dfB[[on] + list(metrics)],
        on=on,
        suffixes=("_A", "_B"),
    )
    rows = []
    for m in metrics:
        a = merged[f"{m}_A"].astype(float).values
        b = merged[f"{m}_B"].astype(float).values
        stats_row = _paired_stats(a, b, alternative="two-sided")
        row = {
            "metric": m,
            "mean_A": stats_row["mean_A"],
            "mean_B": stats_row["mean_B"],
            "mean_diff_AminusB": stats_row["mean_diff"],
            "sd_diff": stats_row["sd_diff"],
            "t_statistic": (
                stats_row["t"] if m not in control_shared else float("nan")
            ),
            "p_value_two_sided": (
                stats_row["p"] if m not in control_shared else float("nan")
            ),
            "cohen_d_paired": stats_row["cohen_d"],
            "n": stats_row["n"],
            "note": ("control_shared" if m in control_shared else ""),
        }
        rows.append(row)
    return pd.DataFrame(rows)

# ------------------------ Contextos desde golden ------------------------


def build_contexts_by_doc(context_from_golden: Dict[str, str]) -> Dict[str, str]:
    acc = defaultdict(list)
    for key, text in context_from_golden.items():
        # Doc-level para contextualizar GND (concatenamos textos por DocID)
        docids = extract_refs(key, article_level=False)
        for d in docids:
            acc[canon_docid(d.split(":", 1)[0])].append(text)
    return {d: " ".join(vs) for d, vs in acc.items()}

# ------------------------ I/O ------------------------


def load_json_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "records" in data and isinstance(data["records"], list):
        return data["records"]
    raise ValueError(f"Formato JSON no reconocido en {path} (espera list of dicts)")


def load_contexts(
    path: str, article_level: bool = False
) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            "--contexts debe ser dict { '[Doc:Sec]...[Doc:Sec]': '[Qxx] texto', ... }"
        )
    gold_by_qid = parse_gold_contexts(raw, article_level=article_level)
    contexts_by_doc = build_contexts_by_doc(raw)
    return contexts_by_doc, gold_by_qid

# ------------------------ Núcleo por temperatura ------------------------


def evaluate_two_datasets(
    path_A: str,
    path_B: str,
    gold_by_qid: Dict[str, Dict[str, Any]],
    *,
    contexts_by_doc: Dict[str, str],
    min_overlap_tokens: int,
    require_citation: bool,
    fold_accents: bool,
    article_level: bool,
    label_A: str,
    label_B: str,
):
    recs_A = load_json_records(path_A)
    recs_B = load_json_records(path_B)
    dfA = evaluate_model(
        recs_A,
        gold_by_qid,
        contexts_by_doc=contexts_by_doc,
        min_overlap_tokens=min_overlap_tokens,
        require_citation=require_citation,
        fold_accents=fold_accents,
        article_level=article_level,
    )
    dfB = evaluate_model(
        recs_B,
        gold_by_qid,
        contexts_by_doc=contexts_by_doc,
        min_overlap_tokens=min_overlap_tokens,
        require_citation=require_citation,
        fold_accents=fold_accents,
        article_level=article_level,
    )
    sumA = pd.DataFrame([summarize_metrics(dfA)], index=[label_A])
    sumB = pd.DataFrame([summarize_metrics(dfB)], index=[label_B])
    paired = paired_diff_summary(dfA, dfB)
    dfA.insert(1, "model", label_A)
    dfB.insert(1, "model", label_B)
    return dfA, dfB, sumA, sumB, paired

# ------------------------ Hipótesis H1–H4 ------------------------


def hypothesis_tests(
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
    labelA: str,
    labelB: str,
    temperature: str,
    telemetry_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    H1: ROUGE1_F1_A != ROUGE1_F1_B
    H2: GND_A != GND_B
    H3: lat_ms_A != lat_ms_B
    H4: GPU_MB_A != GPU_MB_B   (usa metrics_usage_per_prompt si está disponible)

    Todas las pruebas se realizan como t pareado de dos colas; la columna
    'direction' se deriva del signo de la diferencia media (A-B).
    """
    merged = pd.merge(
        dfA[["qid", "ROUGE1_F1", "Groundedness", "lat_ms"]],
        dfB[["qid", "ROUGE1_F1", "Groundedness", "lat_ms"]],
        on="qid",
        suffixes=("_A", "_B"),
    )

    rows: List[Dict[str, Any]] = []

    # P95 de latencias (común para todas las hipótesis, incluida H4)
    if merged["lat_ms_A"].notna().any():
        p95A = latency_percentiles(merged["lat_ms_A"].dropna().tolist())["P95_ms"]
    else:
        p95A = float("nan")
    if merged["lat_ms_B"].notna().any():
        p95B = latency_percentiles(merged["lat_ms_B"].dropna().tolist())["P95_ms"]
    else:
        p95B = float("nan")

    # H1 (ROUGE-1 F1) — calidad léxica, más alto es mejor
    a1 = merged["ROUGE1_F1_A"].astype(float).values
    b1 = merged["ROUGE1_F1_B"].astype(float).values
    st1 = _paired_stats(a1, b1, alternative="two-sided")
    rows.append(
        {
            "temperature": temperature,
            "hypothesis": "H1: ROUGE1_F1_A != ROUGE1_F1_B",
            "metric_primary": "ROUGE1_F1",
            "better_if": "higher",
            "direction": _direction_from_diff(st1["mean_diff"]),
            "model_A": labelA,
            "model_B": labelB,
            "t_stat": st1["t"],
            "p_value_two_sided": st1["p"],
            "cohen_d": st1["cohen_d"],
            "mean_A": st1["mean_A"],
            "mean_B": st1["mean_B"],
            "mean_diff_AminusB": st1["mean_diff"],
            "n": st1["n"],
            "alpha": 0.05,
            "reject_H0_at_0.05": (
                st1["p"] is not None
                and not math.isnan(st1["p"])
                and st1["p"] < 0.05
            ),
            "P95_A_ms": p95A,
            "P95_B_ms": p95B,
        }
    )

    # H2 (Groundedness) — fidelidad, más alto es mejor
    a2 = merged["Groundedness_A"].astype(float).values
    b2 = merged["Groundedness_B"].astype(float).values
    st2 = _paired_stats(a2, b2, alternative="two-sided")
    rows.append(
        {
            "temperature": temperature,
            "hypothesis": "H2: GND_A != GND_B",
            "metric_primary": "Groundedness",
            "better_if": "higher",
            "direction": _direction_from_diff(st2["mean_diff"]),
            "model_A": labelA,
            "model_B": labelB,
            "t_stat": st2["t"],
            "p_value_two_sided": st2["p"],
            "cohen_d": st2["cohen_d"],
            "mean_A": st2["mean_A"],
            "mean_B": st2["mean_B"],
            "mean_diff_AminusB": st2["mean_diff"],
            "n": st2["n"],
            "alpha": 0.05,
            "reject_H0_at_0.05": (
                st2["p"] is not None
                and not math.isnan(st2["p"])
                and st2["p"] < 0.05
            ),
            "P95_A_ms": p95A,
            "P95_B_ms": p95B,
        }
    )

    # H3 (latencia) — eficiencia, más bajo es mejor
    a3 = merged["lat_ms_A"].astype(float).values
    b3 = merged["lat_ms_B"].astype(float).values
    st3 = _paired_stats(a3, b3, alternative="two-sided")
    rows.append(
        {
            "temperature": temperature,
            "hypothesis": "H3: lat_ms_A != lat_ms_B",
            "metric_primary": "lat_ms",
            "better_if": "lower",
            "direction": _direction_from_diff(st3["mean_diff"]),
            "model_A": labelA,
            "model_B": labelB,
            "t_stat": st3["t"],
            "p_value_two_sided": st3["p"],
            "cohen_d": st3["cohen_d"],
            "mean_A": st3["mean_A"],
            "mean_B": st3["mean_B"],
            "mean_diff_AminusB": st3["mean_diff"],
            "n": st3["n"],
            "alpha": 0.05,
            "reject_H0_at_0.05": (
                st3["p"] is not None
                and not math.isnan(st3["p"])
                and st3["p"] < 0.05
            ),
            "P95_A_ms": p95A,
            "P95_B_ms": p95B,
        }
    )

    # H4 (GPU_MB_A != GPU_MB_B) usando metrics_usage_per_prompt
    if (
        telemetry_df is not None
        and not telemetry_df.empty
        and "GPU_MB" in telemetry_df.columns
    ):
        tA = telemetry_df[telemetry_df["model_temp"] == labelA]
        tB = telemetry_df[telemetry_df["model_temp"] == labelB]
        if not tA.empty and not tB.empty:
            merged_t = pd.merge(
                tA[["prompt", "GPU_MB"]],
                tB[["prompt", "GPU_MB"]],
                on="prompt",
                suffixes=("_A", "_B"),
            )
            if not merged_t.empty:
                a4 = merged_t["GPU_MB_A"].to_numpy(dtype=float)
                b4 = merged_t["GPU_MB_B"].to_numpy(dtype=float)
                st4 = _paired_stats(a4, b4, alternative="two-sided")
                rows.append(
                    {
                        "temperature": temperature,
                        "hypothesis": "H4: GPU_MB_A != GPU_MB_B",
                        "metric_primary": "GPU_MB",
                        "better_if": "lower",
                        "direction": _direction_from_diff(st4["mean_diff"]),
                        "model_A": labelA,
                        "model_B": labelB,
                        "t_stat": st4["t"],
                        "p_value_two_sided": st4["p"],
                        "cohen_d": st4["cohen_d"],
                        "mean_A": st4["mean_A"],
                        "mean_B": st4["mean_B"],
                        "mean_diff_AminusB": st4["mean_diff"],
                        "n": st4["n"],
                        "alpha": 0.05,
                        "reject_H0_at_0.05": (
                            st4["p"] is not None
                            and not math.isnan(st4["p"])
                            and st4["p"] < 0.05
                        ),
                        "note": "GPU_MB por pregunta (metrics_usage)",
                        "P95_A_ms": p95A,
                        "P95_B_ms": p95B,
                    }
                )

    return pd.DataFrame(rows)

# ------------------------ CLI ------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Evaluación RAG A vs B por temperatura (0.0 y 0.2)"
    )
    ap.add_argument("--a0", required=True, help="JSON de modelo A @ 0.0")
    ap.add_argument("--b0", required=True, help="JSON de modelo B @ 0.0")
    ap.add_argument("--a2", required=False, help="JSON de modelo A @ 0.2")
    ap.add_argument("--b2", required=False, help="JSON de modelo B @ 0.2")
    ap.add_argument(
        "--contexts",
        required=True,
        help="JSON dict con claves '[Doc:Sec]...[Doc:Sec]' y valores '[Qxx] texto'",
    )
    ap.add_argument("--labelA", default="ModelA", help="Etiqueta para A (sin @temp)")
    ap.add_argument("--labelB", default="ModelB", help="Etiqueta para B (sin @temp)")
    ap.add_argument(
        "--min-overlap-tokens",
        type=int,
        default=3,
        help="Min tokens comunes para marcar oración grounded",
    )
    ap.add_argument(
        "--require-citation",
        type=int,
        default=0,
        help="1 exige [cita] en oración para GND; 0 solo overlap",
    )
    ap.add_argument(
        "--fold-accents",
        type=int,
        default=1,
        help="1 pliega acentos/ñ (lenient), 0 no",
    )
    ap.add_argument(
        "--article-level",
        type=int,
        default=1,
        help="1 usa 'Doc:Art.N' para conteos únicos; 0 Doc-level",
    )
    ap.add_argument("--out", required=False, help="Excel salida (auto si no)")

    # ---- TELEMETRÍA: metrics_usage ----
    ap.add_argument(
        "--metrics-a0",
        required=False,
        help="Ruta metrics_usage-*.txt para modelo A @ 0.0",
    )
    ap.add_argument(
        "--metrics-b0",
        required=False,
        help="Ruta metrics_usage-*.txt para modelo B @ 0.0",
    )
    ap.add_argument(
        "--metrics-a2",
        required=False,
        help="Ruta metrics_usage-*.txt para modelo A @ 0.2",
    )
    ap.add_argument(
        "--metrics-b2",
        required=False,
        help="Ruta metrics_usage-*.txt para modelo B @ 0.2",
    )

    # ---- TELEMETRÍA: all_usage (timeline) ----
    ap.add_argument(
        "--all-a0",
        required=False,
        help="Ruta all_usage-*.txt (timeline) para modelo A @ 0.0",
    )
    ap.add_argument(
        "--all-b0",
        required=False,
        help="Ruta all_usage-*.txt (timeline) para modelo B @ 0.0",
    )
    ap.add_argument(
        "--all-a2",
        required=False,
        help="Ruta all_usage-*.txt (timeline) para modelo A @ 0.2",
    )
    ap.add_argument(
        "--all-b2",
        required=False,
        help="Ruta all_usage-*.txt (timeline) para modelo B @ 0.2",
    )

    # Parámetros de umbrales de telemetría
    ap.add_argument(
        "--vram-cap-mb",
        type=float,
        default=None,
        help="Capacidad de VRAM de la GPU en MB (p.ej. 16375 para RTX A4000)",
    )
    ap.add_argument(
        "--gpu-busy-thresh",
        type=float,
        default=90.0,
        help="Umbral de GPU%% para duty cycle (default 90)",
    )
    ap.add_argument(
        "--vram-high-ratio",
        type=float,
        default=0.9,
        help="Fracción de la VRAM (0-1) para considerar '>=90%% capacidad'",
    )
    ap.add_argument(
        "--ram-high-pct",
        type=float,
        default=80.0,
        help="Umbral de RAM%% para duty cycle (default 80)",
    )
    ap.add_argument(
        "--gpu-temp-high",
        type=float,
        default=80.0,
        help="Umbral de temperatura GPU (°C) para duty cycle (default 80)",
    )

    args = ap.parse_args()

    # -------- Contextos y golden --------
    contexts_by_doc, gold_by_qid = load_contexts(
        args.contexts, article_level=bool(args.article_level)
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.out:
        out_path = args.out
    else:
        # nombres de archivo seguros para Windows
        labelA_fs = safe_fs_component(args.labelA)
        labelB_fs = safe_fs_component(args.labelB)
        out_path = f"RAG_eval_{labelA_fs}_vs_{labelB_fs}_{ts}.xlsx"

    # ---------------- Evaluación principal (RAG) ----------------
    # 0.0
    labelA_0 = f"{args.labelA}@0.0"
    labelB_0 = f"{args.labelB}@0.0"

    dfA0, dfB0, sumA0, sumB0, paired0 = evaluate_two_datasets(
        args.a0,
        args.b0,
        gold_by_qid=gold_by_qid,
        contexts_by_doc=contexts_by_doc,
        min_overlap_tokens=args.min_overlap_tokens,
        require_citation=bool(args.require_citation),
        fold_accents=bool(args.fold_accents),
        article_level=bool(args.article_level),
        label_A=labelA_0,
        label_B=labelB_0,
    )

    # 0.2 (si existe)
    have_02 = bool(args.a2 and args.b2)
    if have_02:
        labelA_2 = f"{args.labelA}@0.2"
        labelB_2 = f"{args.labelB}@0.2"
        dfA2, dfB2, sumA2, sumB2, paired2 = evaluate_two_datasets(
            args.a2,
            args.b2,
            gold_by_qid=gold_by_qid,
            contexts_by_doc=contexts_by_doc,
            min_overlap_tokens=args.min_overlap_tokens,
            require_citation=bool(args.require_citation),
            fold_accents=bool(args.fold_accents),
            article_level=bool(args.article_level),
            label_A=labelA_2,
            label_B=labelB_2,
        )

    # ---------------- Telemetría: contenedores ----------------
    telemetry_metrics_runs: List[Dict[str, Any]] = []        # resumen por run de metrics_usage
    telemetry_metrics_per_prompt: List[pd.DataFrame] = []    # metrics_usage por pregunta
    timeline_runs: List[Dict[str, Any]] = []                 # resumen por run de all_usage
    timeline_per_prompt_all: List[pd.DataFrame] = []         # resumen por pregunta de all_usage

    def _handle_telemetry_for_combo(
        metrics_path: Optional[str],
        all_path: Optional[str],
        model_temp_label: str,
    ):
        # metrics_usage (promedios por pregunta)
        if metrics_path:
            try:
                df_mu = parse_metrics_usage(metrics_path)
            except FileNotFoundError:
                print(f"[WARN] metrics_usage no encontrado: {metrics_path}")
                df_mu = pd.DataFrame()
            if not df_mu.empty:
                df_mu = df_mu.copy()
                df_mu["model_temp"] = model_temp_label
                telemetry_metrics_per_prompt.append(df_mu)

                summ_mu = summarize_metrics_usage_df(df_mu)
                summ_mu["model_temp"] = model_temp_label
                telemetry_metrics_runs.append(summ_mu)

        # all_usage (timeline)
        if all_path:
            try:
                df_tl = parse_all_usage(all_path)
            except FileNotFoundError:
                print(f"[WARN] all_usage no encontrado: {all_path}")
                df_tl = pd.DataFrame()
            if not df_tl.empty:
                df_tl = df_tl.copy()
                df_tl["model_temp"] = model_temp_label

                # resumen global
                summ_tl = summarize_timeline(
                    df_tl,
                    vram_capacity_mb=args.vram_cap_mb,
                    gpu_busy_thresh=args.gpu_busy_thresh,
                    vram_high_ratio=args.vram_high_ratio,
                    ram_high_pct=args.ram_high_pct,
                    gpu_temp_high=args.gpu_temp_high,
                )
                summ_tl["model_temp"] = model_temp_label
                timeline_runs.append(summ_tl)

                # resumen por pregunta
                df_tlp = summarize_timeline_per_prompt(
                    df_tl,
                    vram_capacity_mb=args.vram_cap_mb,
                    gpu_busy_thresh=args.gpu_busy_thresh,
                    vram_high_ratio=args.vram_high_ratio,
                    ram_high_pct=args.ram_high_pct,
                    gpu_temp_high=args.gpu_temp_high,
                )
                if not df_tlp.empty:
                    df_tlp["model_temp"] = model_temp_label
                    timeline_per_prompt_all.append(df_tlp)

    # Telemetría A@0.0, B@0.0, A@0.2, B@0.2 (si existen rutas)
    _handle_telemetry_for_combo(args.metrics_a0, args.all_a0, labelA_0)
    _handle_telemetry_for_combo(args.metrics_b0, args.all_b0, labelB_0)
    if have_02:
        _handle_telemetry_for_combo(args.metrics_a2, args.all_a2, labelA_2)
        _handle_telemetry_for_combo(args.metrics_b2, args.all_b2, labelB_2)

    # DataFrame global de metrics_usage por pregunta (para H4)
    metrics_per_prompt_df: Optional[pd.DataFrame] = None
    if telemetry_metrics_per_prompt:
        metrics_per_prompt_df = pd.concat(
            telemetry_metrics_per_prompt, axis=0, ignore_index=True
        )

    # ---------------- Hipótesis H1–H4 ----------------
    hypo0 = hypothesis_tests(
        dfA0,
        dfB0,
        labelA=labelA_0,
        labelB=labelB_0,
        temperature="0.0",
        telemetry_df=metrics_per_prompt_df,
    )

    if have_02:
        hypo2 = hypothesis_tests(
            dfA2,
            dfB2,
            labelA=labelA_2,
            labelB=labelB_2,
            temperature="0.2",
            telemetry_df=metrics_per_prompt_df,
        )

    # ---------------- Escritura de Excel ----------------
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        # Nombres de hoja sanitizados
        items_A0_sheet = safe_sheet_name(f"items_{args.labelA}_0.0")
        items_B0_sheet = safe_sheet_name(f"items_{args.labelB}_0.0")
        summary_A0_sheet = safe_sheet_name(f"summary_{args.labelA}_0.0")
        summary_B0_sheet = safe_sheet_name(f"summary_{args.labelB}_0.0")

        dfA0.to_excel(writer, sheet_name=items_A0_sheet, index=False)
        dfB0.to_excel(writer, sheet_name=items_B0_sheet, index=False)
        sumA0.to_excel(writer, sheet_name=summary_A0_sheet)
        sumB0.to_excel(writer, sheet_name=summary_B0_sheet)
        paired0.to_excel(writer, sheet_name="paired_diff_0.0", index=False)
        hypo0.to_excel(writer, sheet_name="hypothesis_0.0", index=False)

        if have_02:
            items_A2_sheet = safe_sheet_name(f"items_{args.labelA}_0.2")
            items_B2_sheet = safe_sheet_name(f"items_{args.labelB}_0.2")
            summary_A2_sheet = safe_sheet_name(f"summary_{args.labelA}_0.2")
            summary_B2_sheet = safe_sheet_name(f"summary_{args.labelB}_0.2")

            dfA2.to_excel(writer, sheet_name=items_A2_sheet, index=False)
            dfB2.to_excel(writer, sheet_name=items_B2_sheet, index=False)
            sumA2.to_excel(writer, sheet_name=summary_A2_sheet)
            sumB2.to_excel(writer, sheet_name=summary_B2_sheet)
            paired2.to_excel(writer, sheet_name="paired_diff_0.2", index=False)
            hypo2.to_excel(writer, sheet_name="hypothesis_0.2", index=False)

        summaries = [sumA0.assign(temperature="0.0"), sumB0.assign(temperature="0.0")]
        if have_02:
            summaries += [
                sumA2.assign(temperature="0.2"),
                sumB2.assign(temperature="0.2"),
            ]
        summary_all = pd.concat(summaries, axis=0)
        summary_all.to_excel(writer, sheet_name="summary_all")

        # --------- HOJAS DE TELEMETRÍA ---------
        if telemetry_metrics_runs:
            df_tm_runs = pd.DataFrame(telemetry_metrics_runs)
            df_tm_runs.to_excel(
                writer, sheet_name="telemetry_metrics_runs", index=False
            )

        if metrics_per_prompt_df is not None and not metrics_per_prompt_df.empty:
            metrics_per_prompt_df.to_excel(
                writer, sheet_name="metrics_usage_per_prompt", index=False
            )

        if timeline_runs:
            df_tl_runs = pd.DataFrame(timeline_runs)
            df_tl_runs.to_excel(
                writer, sheet_name="telemetry_timeline_runs", index=False
            )

        if timeline_per_prompt_all:
            df_tl_prompts = pd.concat(
                timeline_per_prompt_all, axis=0, ignore_index=True
            )
            df_tl_prompts.to_excel(
                writer, sheet_name="timeline_per_prompt", index=False
            )

    print(f"[OK] Exportado a: {out_path}")


if __name__ == "__main__":
    main()
