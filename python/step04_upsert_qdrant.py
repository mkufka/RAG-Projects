# step04_upsert_qdrant.py
from __future__ import annotations

# Nutze Windows-Zertifikatspeicher für SSL (Firmen-Proxy-Unterstützung)
import truststore
truststore.inject_into_ssl()
import uuid
from typing import List, Dict, Any, Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from config import Settings

# Aus Schritt 3 holen wir die Embedding-Erzeugung wieder rein
from step03_embeddings import embed_chunks
# Und aus Schritt 2 die Chunks
from step02_pdf_chunking import build_chunks_for_directory
from step01_qdrant_setup import ensure_collection


def batched(iterable: Iterable[Any], n: int) -> Iterable[list[Any]]:
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def records_to_points(records):
    points = []
    for r in records:
        points.append(
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, r["payload"]["source_path"] + "#" + r["id"])),
                vector=r["vector"],
                payload=r["payload"],
            )
        )
    return points


def upsert_records(client: QdrantClient, collection: str, records: List[Dict[str, Any]], batch_size: int = 256) -> int:
    """
    Schreibt die Records in Batches nach Qdrant. Liefert die Anzahl geschriebener Punkte zurück.
    """
    total = 0
    for batch in batched(records, batch_size):
        points = records_to_points(batch)
        client.upsert(
            collection_name=collection,
            points=points,
            wait=True,           # bis Indexierung abgeschlossen ist
        )
        total += len(points)
        print(f"Upsert: {total}/{len(records)} Punkte geschrieben …")
    return total


def main():
    s = Settings()
    s.require_api_key()
    client = QdrantClient(host=s.qdrant_host, grpc_port=s.qdrant_grpc_port, prefer_grpc=True)
    ensure_collection(client, s.collection, s.vector_size)
    # 1) Chunks (Schritt 2) laden/erzeugen
    chunks = build_chunks_for_directory(s)
    if not chunks:
        print("Keine Chunks gefunden – bitte PDFs prüfen.")
        return

    # 2) Embeddings (Schritt 3) erzeugen
    print(f"Erzeuge Embeddings mit Modell: {s.embedding_model}")
    records = embed_chunks(chunks, model=s.embedding_model, batch_size=96)

    # 4) Upsert in Batches
    written = upsert_records(client, s.collection, records, batch_size=256)
    print(f"\nFertig. Insgesamt geschrieben: {written} Punkte in '{s.collection}'.")

    # 5) Optional: Count anzeigen (falls Server das Feature unterstützt)
    try:
        cnt = client.count(collection_name=s.collection, exact=True).count
        print(f"Collection-Zähler (exakt): {cnt}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
