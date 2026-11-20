"""PDF-Erstimport (falls leer) und anschließend interaktiver Chat."""
from config import Settings
from qdrant_client import QdrantClient
from step01_qdrant_setup import ensure_collection
from step04_upsert_qdrant import main as ingest
from step05_chatbot import main as chat

def main():
    s = Settings()
    s.require_api_key()
    client = QdrantClient(host=s.qdrant_host, grpc_port=s.qdrant_grpc_port, prefer_grpc=True)
    try:
        ensure_collection(client, s.collection, s.vector_size)
        if client.count(s.collection, exact=True).count == 0:
            ingest()
    finally:
        client.close()
    chat()

if __name__ == "__main__":
    main()
