# step01_qdrant_setup.py
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from config import Settings

def ensure_collection(client: QdrantClient, name: str, size: int) -> None:
    if client.collection_exists(name):
        print(f"Collection '{name}' existiert bereits – nichts zu tun.")
        return
    # In .NET: Distance.Dot -> hier Distance.DOT
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=size, distance=Distance.DOT),
    )
    print(f"Collection '{name}' angelegt (size={size}, distance=DOT).")

def main() -> None:
    s = Settings()
    # gRPC verwenden, weil Ziel 'docker:6334' ist
    client = QdrantClient(host=s.qdrant_host, grpc_port=s.qdrant_grpc_port, prefer_grpc=True)
    ensure_collection(client, s.collection, s.vector_size)

if __name__ == "__main__":
    main()
