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

* **Python** 3.12 oder neuer; die bestehende Umgebung wurde mit **3.14.2** geprüft.
  Die verwendete f-String-Syntax setzt mindestens Python 3.12 voraus.
* **OpenAI API Key** mit API-Guthaben
* **Docker** (für Qdrant) oder erreichbare Qdrant-Instanz
* **VS Code** optional (Interpreter korrekt auswählen)
* PDFs mit extrahierbarem Text; eine OCR-Erkennung für reine Scans ist nicht enthalten.

---

## Installation

Alle Installations- und Startbefehle in dieser Anleitung werden aus dem Python-Ordner ausgeführt.
Wechsle vom Repository-Hauptverzeichnis zuerst mit `Set-Location .\python` dorthin (PowerShell).

### 1) Virtuelle Umgebung (empfohlen)

**Windows (PowerShell):**

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Falls Python 3.12 oder 3.13 verwendet wird, passe den Versionsschalter entsprechend an.
Eine bereits vorhandene Umgebung muss nicht neu erstellt werden.

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Pakete installieren

```powershell
python -m pip install -r requirements.txt truststore
python -m pip check
```

Der Qdrant-Client ist auf die Version 1.20.x passend zum eingesetzten Qdrant-Server 1.20.2 begrenzt.
`truststore` wird von den Embedding-, Import- und Chatskripten benötigt, um den
System-Zertifikatspeicher zu verwenden, ist aber derzeit nicht in
[requirements.txt](requirements.txt) enthalten und wird deshalb explizit mitinstalliert.
Maßgeblich für die übrigen Paketversionen ist diese Datei; bestehende Umgebungen können davon abweichen.

Falls die PowerShell-Aktivierung blockiert ist, verwende den Interpreter direkt, zum Beispiel
`.\.venv\Scripts\python.exe -m pip install -r requirements.txt truststore` und
`.\.venv\Scripts\python.exe step05_chatbot.py`. Eine Änderung der Execution Policy ist nicht erforderlich.

> Optional für Entwicklung: `pip install ruff black` (Lint/Format)

---

## Konfiguration (.env)

Kopiere [.env.example](../.env.example) nach `.env` ins Repository-Hauptverzeichnis
(eine Ebene oberhalb dieses Python-Ordners). Beide Anwendungen laden diese zentrale Datei;
bereits gesetzte Umgebungsvariablen haben Vorrang. Die Suche ist unabhängig vom
Arbeitsverzeichnis. Eine bisherige `python/.env` wird im Python-Ordner nicht mehr
geladen: Verschiebe sie ins Repository-Hauptverzeichnis oder führe ihre Werte mit
einer dort bereits vorhandenen `.env` zusammen. Überschreibe keine bestehende Konfiguration. Minimal:

```env
OPENAI_API_KEY=dein_api_key
RAG_PDF_DIR='C:\Pfad\zu\PDFs'
QDRANT_HOST=localhost
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
RAG_DOC_FILTER=             # optional: kommaseparierte document_id-Werte
RAG_STREAM=false            # true aktiviert Streaming-Ausgabe
```

---

## Qdrant starten (Docker)

```bash
docker run --name rag-qdrant -d -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant:v1.20.2
```

Nur ausführen, wenn noch keine passende Instanz läuft. Einen mit diesem Befehl
angelegten und später gestoppten Container kannst Du mit `docker start rag-qdrant` erneut starten.

* HTTP: 6333, gRPC: **6334**
* Zentrale `.env` entsprechend: `QDRANT_HOST=localhost`, `QDRANT_GRPC_PORT=6334`, wenn die Anwendungen lokal laufen.

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

Bei einem bereits befüllten Index genügt Schritt 3, auch wenn der Index mit .NET erzeugt wurde.
Die .NET-Anwendung importiert beim Start allerdings nicht in bereits vorhandene Collections:
Nach einem alleinigen Python-Collection-Setup muss deshalb Schritt 2 ausgeführt werden.
Wiederholte Importe erzeugen durch zufällige UUIDs zusätzliche Einträge statt vorhandene Chunks zu ersetzen.
PDF-Texte und Fragen werden an OpenAI übertragen; Embedding-Vorschauen, Import und Chat können API-Kosten verursachen.

---

## Skripte im Überblick

### `step01_qdrant_setup.py`

* Verbindet sich zu Qdrant (gRPC) und legt die Collection an
* **VectorParams**: Größe **3072** (für `text-embedding-3-large`), Distanz **DOT**

### `step02_pdf_chunking.py`

* Liest PDFs rekursiv aus `RAG_PDF_DIR`, einschließlich Unterverzeichnissen
* Normalisiert Text, chunked tokenbasiert (`RAG_CHUNK_TOKENS`, `RAG_CHUNK_OVERLAP`)
* Metadaten: `document_id`, `chunk_index`, `source_path`, `page_start`, `page_end`
* Gibt eine Vorschau im Terminal aus

### `step03_embeddings.py`

* Erzeugt Embeddings für alle Chunks (`EMBEDDING_MODEL`)
* **L2-Normalisierung** der Vektoren (damit DOT ≙ Cosine)
* Batchweise mit Retry-Logik
* Warnt bei abweichender Dimension (erwartet **3072**), bricht deswegen aber nicht selbst ab

### `step04_upsert_qdrant.py`

* Baut Chunks (Step 2) → Embeddings (Step 3) → schreibt als Punkte in Qdrant
* Point-ID: zufällige **UUID (String)** bei jedem Import; der Import ist deshalb nicht idempotent
* Batch-Upsert mit `wait=True`

### `step05_chatbot.py`

* Endlosschleife: Eingabe lesen, mit `exit` beenden
* Query-Embedding (L2-normalisiert) → Qdrant-Suche; jede Frage wird ohne Gesprächshistorie beantwortet
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
* Für das Repository als geöffneten Workspace gilt in [.vscode/settings.json](../.vscode/settings.json):

  ```json
  {
    "python.defaultInterpreterPath": "python\\.venv\\Scripts\\python.exe",
    "python.analysis.extraPaths": ["python\\.venv\\Lib\\site-packages"]
  }
  ```

### Qdrant-Client und Server

* Vorgesehen sind `qdrant-client` **1.20.x** und Qdrant-Server **1.20.2**.
* Bei einer Kompatibilitätswarnung Client- und Serverversion abgleichen, statt die Prüfung abzuschalten.
* Die Suche nutzt bevorzugt `query_points(...)` mit `query_filter` und einem Vektor als `query`.

Der Code enthält zusätzlich Fallbacks auf `filter` und auf die ältere `search(...)`-API.
Sie ersetzen keine passende Client-/Serverkombination.

### Deprecation: `search`

* `search(...)` ist veraltet; die Implementierung nutzt bevorzugt `query_points(...)`.

### Fehler: `Unsupported query type: <class 'dict'>`

* Bei `query_points`: `query` **muss der Vektor** selbst sein (Liste `float`), **kein** Dict.

### Fehler: `PointStruct.id` ValidationError

* `id` muss **int oder str** sein – **`str(uuid.uuid4())`** verwenden.
* Ein zusammengesetzter Wert wie `"Dokument#0"` ist keine gültige UUID.

### Verbindung gRPC schlägt fehl

* Ports 6333/6334 im Docker-Run veröffentlicht?
* Für lokal gestartete Anwendungen `QDRANT_HOST=localhost` setzen, wenn Qdrant lokal läuft.
* Bei einem entfernten Server dessen erreichbaren Hostnamen verwenden.

### OpenAI 401/429

* API-Key in der zentralen `.env`, API-Guthaben und Modellzugriff prüfen.
* Bei Embedding-Anfragen greift die Retry-Logik; dauerhafte Authentifizierungsfehler
  oder fehlendes Guthaben werden dadurch nicht behoben.

### Dimension passt nicht

* Embedding-Modell muss zu **3072** passen: `text-embedding-3-large`.

---

## Tipps & Anpassungen

* **Chunk-Größe**: 300–600 Tokens sind oft ein guter Start. `RAG_CHUNK_TOKENS` anpassen.
* **Overlap**: 30–80 Tokens – zu groß bläht Index auf, zu klein reißt Kontext.
* **Score-Threshold**: Erhöhen (z. B. 0.35), wenn Treffer zu breit sind.
* **MMR λ**: `0.3–0.7`. Niedriger = mehr Diversität, höher = mehr Relevanz.
* **Reindexierung**: Aktuell werden zufällige UUIDs verwendet. Für idempotente
  Upserts müsste der Code auf stabile, gültige UUIDs oder numerische IDs umgestellt werden.

---

## Lizenz / Urheber

Interne Demo. Verwendung nach eigenem Ermessen.
