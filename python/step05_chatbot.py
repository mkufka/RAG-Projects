# step05_chatbot.py
from __future__ import annotations
import sys
import numpy as np
import tiktoken
from typing import List, Tuple
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from qdrant_client.models import ScoredPoint, Filter, FieldCondition, MatchValue


from config import Settings
from step03_embeddings import l2_normalize  # gleiche Normierung wie beim Index

ENC = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(ENC.encode(text))

def trim_to_tokens(texts: List[str], max_tokens: int) -> Tuple[str, int]:
    """Fügt Texte nacheinander zusammen, bis max_tokens erreicht sind."""
    out, used = [], 0
    for t in texts:
        t_tokens = count_tokens(t)
        if used + t_tokens > max_tokens:
            break
        out.append(t)
        used += t_tokens
    return ("\n\n".join(out), used)

def build_system_prompt() -> str:
    return (
        "Du bist ein hilfreicher Assistent für Retrieval-Augmented Generation (RAG). "
        "Du DARFST NUR Informationen verwenden, die im Abschnitt 'Kontext' bereitgestellt werden. "
        "Wenn der Kontext nicht ausreicht, sage klar: 'Dazu habe ich im bereitgestellten Material nichts gefunden.' "
        "Antworte auf Deutsch, kurz und präzise, benutze ausschließlich das generische Maskulinum."
    )

def format_hit(h: ScoredPoint) -> str:
    p = h.payload or {}
    doc = p.get("document_id", "unbekannt")
    cs = p.get("chunk_index", -1)
    ps = p.get("page_start", "?")
    pe = p.get("page_end", "?")
    txt = p.get("text", "")
    header = f"[{doc} | Chunk {cs} | Seiten {ps}-{pe} | Score {h.score:.3f}]"
    return header + "\n" + txt

def summarize_sources(hits: List[ScoredPoint]) -> str:
    lines = []
    for h in hits:
        p = h.payload or {}
        doc = p.get("document_id", "unbekannt")
        ps = p.get("page_start", "?")
        pe = p.get("page_end", "?")
        lines.append(f"- {doc} (S. {ps}-{pe}) — Score {h.score:.3f}")
    # Doppelte Zeilen vermeiden (selten nötig, aber sicher ist sicher)
    seen = set()
    uniq = []
    for ln in lines:
        if ln not in seen:
            uniq.append(ln); seen.add(ln)
    return "\n".join(uniq)


def embed_query(client: OpenAI, model: str, text: str, dim: int) -> List[float]:
    resp = client.embeddings.create(model=model, input=[text])
    vec = resp.data[0].embedding
    if len(vec) != dim:
        print(f"Warnung: Query-Embedding-Dim {len(vec)} != erwarteten {dim}", file=sys.stderr)
    return l2_normalize(vec)

# def search_qdrant(qc: QdrantClient, s: Settings, query_vec: List[float]) -> List[ScoredPoint]:
#     """
#     Nutzt bevorzugt die neue Qdrant-API query_points(...).
#     Fällt bei älteren qdrant-client-Versionen auf search(...) zurück.
#     """
#     try:
#         # Korrekt: query ist direkt der dichte Vektor (List[float])
#         resp = qc.query_points(
#             collection_name=s.collection,
#             query=query_vec,                 # <-- statt {"vector": query_vec}
#             limit=s.top_k,
#             with_payload=True,
#             score_threshold=s.score_threshold,
#         )
#         hits = resp.points  # List[ScoredPoint]
#     except AttributeError:
#         # Fallback auf alte API
#         hits = qc.search(
#             collection_name=s.collection,
#             query_vector=query_vec,
#             limit=s.top_k,
#             with_payload=True,
#             score_threshold=s.score_threshold,
#         )
#     return hits

def search_qdrant(qc: QdrantClient, s: Settings, query_vec: List[float]) -> List[ScoredPoint]:
    """
    Query mit Filter & Vektoren (für MMR). Handhabt unterschiedliche Client-Signaturen.
    """
    doc_whitelist = parse_doc_filter(s.doc_filter)
    flt = build_filter(doc_whitelist)
    limit_candidates = max(s.candidate_k, s.top_k)

    # 1) Bevorzugt: neue API query_points(...)
    try:
        try:
            # Variante A: query_filter (häufiger in aktuellen Clients)
            resp = qc.query_points(
                collection_name=s.collection,
                query=query_vec,
                limit=limit_candidates,
                with_payload=True,
                with_vectors=True,            # nötig für MMR
                query_filter=flt,             # <— WICHTIG: query_filter statt filter
                score_threshold=s.score_threshold,
            )
        except (TypeError, AssertionError):
            # Variante B: manche Builds akzeptieren 'filter' statt 'query_filter'
            resp = qc.query_points(
                collection_name=s.collection,
                query=query_vec,
                limit=limit_candidates,
                with_payload=True,
                with_vectors=True,
                filter=flt,                   # Fallback
                score_threshold=s.score_threshold,
            )
        candidates = resp.points

    except AttributeError:
        # 2) Fallback: alte API search(...)
        try:
            candidates = qc.search(
                collection_name=s.collection,
                query_vector=query_vec,
                limit=limit_candidates,
                with_payload=True,
                with_vectors=True,
                query_filter=flt,             # neuere Signatur der alten Methode
                score_threshold=s.score_threshold,
            )
        except TypeError:
            # ganz alt: ohne query_filter
            candidates = qc.search(
                collection_name=s.collection,
                query_vector=query_vec,
                limit=limit_candidates,
                with_payload=True,
                with_vectors=True,
                score_threshold=s.score_threshold,
            )

    # 3) MMR auf Kandidaten
    reranked = mmr_rerank(query_vec, candidates, s.top_k, s.mmr_lambda)
    return reranked



def build_context(hits: List[ScoredPoint], max_tokens: int) -> Tuple[str, List[ScoredPoint]]:
    # Formatiere jeden Treffer mit Kopf + Text, trimme auf das Tokenlimit
    formatted = [format_hit(h) for h in hits]
    context, _ = trim_to_tokens(formatted, max_tokens)
    # Reduziere Trefferliste passend zur tatsächlich verwendeten Context-Menge
    used_hits = []
    used_tokens = 0
    for h, block in zip(hits, formatted):
        t = count_tokens(block)
        if used_tokens + t > max_tokens:
            break
        used_hits.append(h)
        used_tokens += t
    return context, used_hits

# def chat_once(client: OpenAI, s: Settings, context: str, user_query: str) -> str:
#     system_prompt = build_system_prompt()
#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content":
#             "Kontext:\n" + context + "\n\n"
#             "Aufgabe: Beantworte die folgende Frage ausschließlich anhand des obigen Kontexts. "
#             "Wenn der Kontext nicht reicht, sag ehrlich, dass du keine Information im Material findest.\n\n"
#             f"Frage: {user_query}"
#         },
#     ]
#     resp = client.chat.completions.create(
#         model=s.chat_model,
#         messages=messages,
#         temperature=0.2,
#         max_tokens=s.max_answer_tokens,
#     )
#     return resp.choices[0].message.content.strip()

def chat_once(client: OpenAI, s: Settings, context: str, user_query: str) -> str:
    system_prompt = build_system_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content":
            "Kontext:\n" + context + "\n\n"
            "Aufgabe: Beantworte die folgende Frage ausschließlich anhand des obigen Kontexts. "
            "Wenn der Kontext nicht reicht, sag ehrlich, dass du keine Information im Material findest.\n\n"
            f"Frage: {user_query}"
        },
    ]

    if s.stream:
        # Streamed Ausgabe live in die Konsole
        print()
        acc = []
        with client.chat.completions.stream(
            model=s.chat_model,
            messages=messages,
            temperature=0.2,
            max_tokens=s.max_answer_tokens,
        ) as stream:
            for event in stream:
                if event.type == "content.delta":
                    chunk = event.delta
                    if chunk:
                        text = chunk
                        sys.stdout.write(text)
                        sys.stdout.flush()
                        acc.append(text)
                elif event.type == "content.completed":
                    print()
                elif event.type == "error":
                    print(f"\n[Stream-Fehler] {event.error}", file=sys.stderr)
        return "".join(acc).strip()

    # Non-Streaming (wie bisher)
    resp = client.chat.completions.create(
        model=s.chat_model,
        messages=messages,
        temperature=0.2,
        max_tokens=s.max_answer_tokens,
    )
    return resp.choices[0].message.content.strip()


def main():
    s = Settings()

    # OpenAI + Qdrant
    oa = OpenAI(api_key=s.openai_api_key)
    qc = QdrantClient(host=s.qdrant_host, grpc_port=s.qdrant_grpc_port, prefer_grpc=True)

    print("RAG-Chat gestartet. Tippe deine Frage. Mit 'exit' beenden.\n")
    while True:
        try:
            user_query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTschüss.")
            break

        if user_query.lower() in {"exit", "quit", ":q", "bye"}:
            print("Tschüss.")
            break
        if not user_query:
            continue

        # 1) Query einbetten (L2-normalisiert)
        qvec = embed_query(oa, s.embedding_model, user_query, s.vector_size)

        # 2) Suche in Qdrant
        hits = search_qdrant(qc, s, qvec)

        if not hits:
            print("Keine passenden Stellen im Material gefunden.")
            continue

        # 3) Kontext bauen (Token-begrenzt)
        context, used_hits = build_context(hits, s.max_context_tokens)

        # 4) Chat-Antwort generieren
        answer = chat_once(oa, s, context, user_query)

        # 5) Quellenhinweis drucken
        sources = summarize_sources(used_hits)
        if sources:
            answer += "\n\nQuellen:\n" + sources

        print("\n" + answer + "\n")

def parse_doc_filter(raw: str) -> list[str] | None:
    if not raw:
        return None
    vals = [x.strip() for x in raw.split(",")]
    vals = [x for x in vals if x]
    return vals or None

def build_filter(doc_whitelist: list[str] | None) -> Filter | None:
    if not doc_whitelist:
        return None
    # OR über mehrere document_id-Werte
    should = [FieldCondition(key="document_id", match=MatchValue(value=v)) for v in doc_whitelist]
    return Filter(should=should)

def mmr_rerank(query_vec: list[float], hits: list[ScoredPoint], k: int, lambda_mult: float) -> list[ScoredPoint]:
    """
    Maximal Marginal Relevance:
      score = λ * sim(query, doc) - (1-λ) * max_sim(doc, already_selected)
    Erwartet, dass die ScoredPoints die gespeicherten Vektoren enthalten (with_vectors=True).
    """
    if not hits:
        return []
    # Hole Dokumentvektoren
    def vec_of(h: ScoredPoint) -> np.ndarray:
        v = getattr(h, "vector", None)
        if v is None:
            # Falls der Client statt 'vector' ein Mapping liefert (Multi-Vector-Setup), versuche es abzufangen
            v = getattr(h, "vectors", None)
            if isinstance(v, dict):
                # nimm den ersten Eintrag
                v = next(iter(v.values()))
        if v is None:
            # Notfall: verwende Qdrant-Score als Proxy (kein echtes MMR möglich)
            return None
        return np.array(v, dtype=np.float32)

    q = np.array(query_vec, dtype=np.float32)
    candidate = []
    doc_vecs: list[np.ndarray] = []

    # Precompute Ähnlichkeiten zum Query (cosine ~ dot, da normalisiert)
    sim_to_q: list[float] = []
    for h in hits:
        v = vec_of(h)
        if v is None:
            # Wenn Vektoren fehlen, brich ab und gib Original-Top-k zurück
            return hits[:k]
        sim_to_q.append(float(np.dot(q, v)))
        doc_vecs.append(v)

    selected = []
    selected_idx: list[int] = []
    k = min(k, len(hits))

    while len(selected) < k:
        best_idx, best_score = -1, -1e9
        for i, h in enumerate(hits):
            if i in selected_idx:
                continue
            # max Ähnlichkeit zu bereits ausgewählten
            if selected_idx:
                max_sim = max(float(np.dot(doc_vecs[i], doc_vecs[j])) for j in selected_idx)
            else:
                max_sim = 0.0
            score = lambda_mult * sim_to_q[i] - (1.0 - lambda_mult) * max_sim
            if score > best_score:
                best_score, best_idx = score, i
        selected.append(hits[best_idx])
        selected_idx.append(best_idx)

    return selected


if __name__ == "__main__":
    main()
