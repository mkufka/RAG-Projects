# step03_embeddings.py
from __future__ import annotations
import time
from typing import List, Dict, Any, Iterable
import numpy as np
from openai import OpenAI
from config import Settings
# Wir nutzen die Chunks aus Schritt 2 erneut:
from step02_pdf_chunking import build_chunks_for_directory, Chunk


def batched(iterable: Iterable[Any], n: int) -> Iterable[list[Any]]:
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def l2_normalize(vec: List[float]) -> List[float]:
    v = np.array(vec, dtype=np.float32)
    n = float(np.linalg.norm(v)) or 1.0
    return (v / n).tolist()


def embed_chunks(
    chunks: List[Chunk],
    model: str,
    batch_size: int = 96,
    max_retries: int = 5,
) -> List[Dict[str, Any]]:
    """
    Erzeugt Embeddings für alle Chunks und liefert
    eine Liste aus { "id", "vector", "payload" } zurück.
    - id: z. B. "<document_id>#<chunk_index>"
    - vector: L2-normalisierter Vektor (für DOT-Ähnlichkeit)
    - payload: Metadaten (document_id, chunk_index, text, source_path, page_start, page_end)
    """
    s = Settings()
    client = OpenAI(api_key=s.openai_api_key)

    records: List[Dict[str, Any]] = []
    total = len(chunks)
    done = 0

    for batch in batched(chunks, batch_size):
        inputs = [c.text for c in batch]

        # Retry-Loop (Rate Limits, temporäre Fehler)
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.embeddings.create(model=model, input=inputs)
                break
            except Exception as e:
                if attempt >= max_retries:
                    raise
                sleep_s = min(2 ** attempt, 30)
                print(f"Embedding-Fehler (Versuch {attempt}/{max_retries}): {e}. "
                      f"War­te {sleep_s}s und versuche erneut...")
                time.sleep(sleep_s)

        assert len(resp.data) == len(batch), "Embedding-Antwort-Länge unerwartet"

        for c, emb in zip(batch, resp.data):
            vec = emb.embedding
            # Sanity-Checks
            if len(vec) != s.vector_size:
                print(f"Warnung: Embedding-Dimension {len(vec)} != erwarteten {s.vector_size}")

            rec = {
                "id": f"{c.document_id}#{c.chunk_index}",
                "vector": l2_normalize(vec),   # wichtig für Distance.DOT
                "payload": {
                    "document_id": c.document_id,
                    "chunk_index": c.chunk_index,
                    "text": c.text,
                    "source_path": c.source_path,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                },
            }
            records.append(rec)

        done += len(batch)
        print(f"Embeddings: {done}/{total} fertig")

    return records


def main():
    s = Settings()
    # Chunks erneut erzeugen (einfachste Variante).
    chunks = build_chunks_for_directory(s)

    print(f"Starte Embeddings mit Modell: {s.embedding_model}")
    records = embed_chunks(chunks, model=s.embedding_model, batch_size=96)

    # Kleine Vorschau
    print("\nVorschau (2 Einträge):")
    for r in records[:2]:
        v = r["vector"]
        norm = np.linalg.norm(np.array(v, dtype=np.float32))
        print(f"- {r['id']} | dim={len(v)} | ‖v‖≈{norm:.3f} | "
              f"Seiten {r['payload']['page_start']}-{r['payload']['page_end']}")
        print(f"  Text: {r['payload']['text'][:160].replace('\n',' ')}"
              f"{'...' if len(r['payload']['text'])>160 else ''}")

    # WICHTIG: Hier noch kein Upsert. Das folgt in Schritt 4.
    # Wir geben nur die Anzahl aus:
    print(f"\nGesamt erzeugte Embeddings: {len(records)}")


if __name__ == "__main__":
    main()
