# RAG mit Python, OpenAI und Qdrant – README

Diese README beschreibt Setup, Konfiguration und Nutzung der kleinen RAG-Referenzimplementierung:

* **PDFs einlesen → Chunks bilden → Embeddings erzeugen → in Qdrant schreiben → Chatbot**
* Fokus: einfache, nachvollziehbare Skripte (`step01` … `step05`)

---

## Architektur / Pipeline

1. **PDF-Import & Chunking** (tokenbasiert über `tiktoken`)
2. **Embeddings** mit `text-embedding-3-large` (L2-normalisiert)
3. **Qdrant** als Vektordatenbank (Distance **DOT**; normalisierte Vektoren ≙ Cosine)
4. **Chat**: Query → Einbettung → Qdrant-Suche → Kontext → Antwort **nur** aus Kontext

---

## Voraussetzungen

* **Python** 3.10–3.13
* **OpenAI API Key**
* **Docker** (für Qdrant) oder erreichbare Qdrant-Instanz
* **VS Code** optional (Interpreter korrekt auswählen)

---

## Installation

### 0) pip install -r requirements.txt

### 1) Virtuelle Umgebung (empfohlen)

**Windows (PowerShell):**

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Pakete installieren

```bash
pip install qdrant-client openai python-dotenv pypdf tiktoken numpy
```

> Optional für Entwicklung: `pip install ruff black` (Lint/Format)

---

## Konfiguration (.env)

Lege eine Datei `.env` im Projektroot an. Minimal:

```env
OPENAI_API_KEY=dein_api_key
RAG_PDF_DIR=/pfad/zu/deinen/pdfs
QDRANT_HOST=docker
QDRANT_GRPC_PORT=6334
RAG_COLLECTION_NAME=CollectionWithData
EMBEDDING_MODEL=text-embedding-3-large
```

Optionale Feineinstellungen:

```env
# Chunking
RAG_CHUNK_TOKENS=500
RAG_CHUNK_OVERLAP=50

# Chat
CHAT_MODEL=gpt-4o-mini
RAG_TOP_K=5
RAG_CANDIDATE_K=20
RAG_MMR_LAMBDA=0.5
RAG_SCORE_THRESHOLD=0.25
RAG_MAX_CONTEXT_TOKENS=2000
RAG_MAX_ANSWER_TOKENS=400
RAG_DOC_FILTER=             # z. B. "Businessplan SmartPlanAI,Azure Kostenkalkulation SmartPlanAI"
RAG_STREAM=false            # true aktiviert Streaming-Ausgabe
```

---

## Qdrant starten (Docker)

```bash
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

* HTTP: 6333, gRPC: **6334**
* `.env` entsprechend: `QDRANT_HOST=docker`, `QDRANT_GRPC_PORT=6334`

---

## Erstbefüllung & Start (Kurzfassung)

1. **Collection anlegen**

   ```bash
   python step01_qdrant_setup.py
   ```
2. **Indizieren** (PDFs → Chunks → Embeddings → Upsert)

   ```bash
   python step04_upsert_qdrant.py
   ```
3. **Chatbot starten**

   ```bash
   python step05_chatbot.py
   ```

> Einzeltests: `step02_pdf_chunking.py` (Chunks-Vorschau), `step03_embeddings.py` (Embeddings-Vorschau)

---

## Skripte im Überblick

### `step01_qdrant_setup.py`

* Verbindet sich zu Qdrant (gRPC) und legt die Collection an
* **VectorParams**: Größe **3072** (für `text-embedding-3-large`), Distanz **DOT**

### `step02_pdf_chunking.py`

* Liest PDFs aus `RAG_PDF_DIR`
* Normalisiert Text, chunked tokenbasiert (`RAG_CHUNK_TOKENS`, `RAG_CHUNK_OVERLAP`)
* Metadaten: `document_id`, `chunk_index`, `source_path`, `page_start`, `page_end`
* Gibt eine Vorschau im Terminal aus

### `step03_embeddings.py`

* Erzeugt Embeddings für alle Chunks (`EMBEDDING_MODEL`)
* **L2-Normalisierung** der Vektoren (damit DOT ≙ Cosine)
* Batchweise mit Retry-Logik
* Prüft Dimension (sollte **3072** sein)

### `step04_upsert_qdrant.py`

* Baut Chunks (Step 2) → Embeddings (Step 3) → schreibt als Punkte in Qdrant
* Point-ID: **UUID (String)**; alternativ deterministisch `"{document_id}#{chunk_index}"`
* Batch-Upsert mit `wait=True`

### `step05_chatbot.py`

* Endlosschleife: Eingabe lesen, mit `exit` beenden
* Query-Embedding (L2-normalisiert) → Qdrant-Suche
* Kontext bauen (Tokenlimit) → Antwort generieren (**nur** aus Kontext)
* **MMR-Reranking**, optional **Dokumentfilter**, **Streaming**
* Quellenliste mit Score

---

## Wichtige Designpunkte

* **DOT + Normalisierung**: Durch L2-Normierung der Vektoren wird die Punktprodukt-Suche faktisch zur Cosine-Suche. Das ist stabil in Praxis.
* **Tokenbasiertes Chunking**: kompatibel zum OpenAI-Tokenizer (`cl100k_base`).
* **Striktes Sourcing**: System-Prompt zwingt Antworten ausschließlich aus dem bereitgestellten Kontext.

---

## Troubleshooting

### Pylance: `Import ... could not be resolved`

* VS Code → **Python: Select Interpreter** auf die **richtige** Umgebung stellen
* Prüfen im Terminal:

  ```powershell
  python -c "import sys; print(sys.executable)"
  ```
* Optional `.vscode/settings.json`:

  ```json
  {
    "python.defaultInterpreterPath": ".venv\\Scripts\\python.exe",
    "python.analysis.extraPaths": [".venv/Lib/site-packages"]
  }
  ```

### Qdrant `query_points` Signatur

* Manche `qdrant-client`-Versionen erwarten `query_filter=...` statt `filter=...`.
* Der Code enthält einen Fallback (erst `query_filter`, dann `filter`).
* Bei sehr alten Versionen wird auf `search(...)` zurückgefallen.

### Deprecation: `search`

* `search(...)` ist veraltet; der Patch nutzt bevorzugt `query_points(...)`.

### Fehler: `Unsupported query type: <class 'dict'>`

* Bei `query_points`: `query` **muss der Vektor** selbst sein (Liste `float`), **kein** Dict.

### Fehler: `PointStruct.id` ValidationError

* `id` muss **int oder str** sein – **`str(uuid.uuid4())`** verwenden.

### Verbindung gRPC schlägt fehl

* Ports 6333/6334 im Docker-Run veröffentlicht?
* Hostname `docker` erreichbar? Ggf. `QDRANT_HOST=localhost` setzen, wenn lokal.

### OpenAI 401/429

* API Key prüfen; bei 429 greift die **Retry-Logik** im Embedding-Code automatisch.

### Dimension passt nicht

* Embedding-Modell muss zu **3072** passen: `text-embedding-3-large`.

---

## Tipps & Anpassungen

* **Chunk-Größe**: 300–600 Tokens sind oft ein guter Start. `RAG_CHUNK_TOKENS` anpassen.
* **Overlap**: 30–80 Tokens – zu groß bläht Index auf, zu klein reißt Kontext.
* **Score-Threshold**: Erhöhen (z. B. 0.35), wenn Treffer zu breit sind.
* **MMR λ**: `0.3–0.7`. Niedriger = mehr Diversität, höher = mehr Relevanz.
* **Deterministische IDs**: Upserts werden idempotent, nützlich beim Reindex.

---

## Lizenz / Urheber

Interne Demo. Verwendung nach eigenem Ermessen.
