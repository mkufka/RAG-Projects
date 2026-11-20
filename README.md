# RAG mit .NET, Python, OpenRouter und Qdrant

Beide Konsolenprogramme verarbeiten PDFs aus `C:\PDFs` und verwenden dieselbe zentrale `.env` und denselben Qdrant-Index. Die vorhandene PDF ist bereits vorkonfiguriert.

## Start

In `C:\Projekte\RAG-Projects\.env` nur `OPENROUTER_API_KEY=` ausfüllen. Der Key bleibt lokal und wird von Git ignoriert.

Python wie bisher über die einzelnen Schritte starten (PowerShell):

```powershell
Set-Location C:\Projekte\RAG-Projects\python
.\.venv\Scripts\python.exe step01_qdrant_setup.py
.\.venv\Scripts\python.exe step04_upsert_qdrant.py
.\.venv\Scripts\python.exe step05_chatbot.py
```

`step02_pdf_chunking.py` und `step03_embeddings.py` lassen sich ebenfalls direkt ausführen. Sie dienen als Vorschau; Schritt 4 führt Chunking und Embeddings selbst aus. Schritt 3 erzeugt dabei zusätzliche API-Aufrufe.

.NET wie bisher in Visual Studio starten oder direkt aus dem Repository-Hauptverzeichnis:

```powershell
Set-Location C:\Projekte\RAG-Projects
dotnet run --project .\.NET\RetrievalAugmentedGeneration\RetrievalAugmentedGeneration.csproj
```

Bei .NET wird beim ersten Start eine fehlende Collection angelegt und eine leere Collection mit der PDF befüllt. Ein bereits befüllter Index wird wiederverwendet. Den Chat mit `exit` beenden. Neue/geänderte PDFs werden nicht automatisch synchronisiert.

Optional: `Start-DotNet.ps1` und `Start-Python.ps1` im Hauptverzeichnis sind Abkürzungen. `python/run.py` kombiniert Erstimport bei leerem Index und Chat. Diese zusätzlichen Dateien sind für die bisherige Nutzung nicht erforderlich.

## Konfiguration

- `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1`
- `EMBEDDING_MODEL=openai/text-embedding-3-large` (3072 Dimensionen)
- `CHAT_MODEL=openai/gpt-6-luna`

Der Chat verwendet `reasoning_effort=none` für direkte, kostengünstige Antworten und sendet keinen `temperature`-Parameter. Embeddings und der vorhandene Index bleiben unverändert.
- `RAG_PDF_DIR='C:\PDFs'`
- `RAG_COLLECTION_NAME=rag_pdfs_openrouter`
- `QDRANT_HOST=localhost`, `QDRANT_GRPC_PORT=6334`

Bereits gesetzte Umgebungsvariablen haben Vorrang vor der `.env`. Beide Anwendungen verwenden die OpenAI-kompatiblen SDKs mit OpenRouter als Endpoint, für Embeddings und Chat. Dokumenttexte und Fragen werden beim Import/Chat an OpenRouter und dessen Modellanbieter übertragen; API-Guthaben wird benötigt. Ohne eingetragenen Key können keine echten Embeddings oder Antworten erzeugt werden. PDFs benötigen extrahierbaren Text; OCR ist nicht enthalten.

Die zusätzlichen Chunking-, MMR- und Chatparameter in `.env.example` gelten für Python. .NET chunkt nach Zeichen und führt einen Chat mit Gesprächshistorie. Der gemeinsame Index verwendet DOT-Distanz und Embeddings desselben Modells. Bei Modellwechsel einen neuen Collection-Namen wählen; Dimensionen allein garantieren keine kompatiblen Embeddings.

## Umgebung und Builds

Installiert/verwendet: .NET SDK 11 mit Ziel `net10.0` und vorhandener .NET-10-Runtime; Python 3.13 in `python\.venv`.

```powershell
dotnet build .\.NET\RetrievalAugmentedGeneration.sln -c Release --source https://api.nuget.org/v3/index.json
.\python\.venv\Scripts\python.exe -m pip install -r .\python\requirements.txt
.\python\.venv\Scripts\python.exe -m pip check
```

Für eine neue Python-Umgebung: `py -3.13 -m venv python\.venv`. Python mindestens 3.12 wegen der f-String-Syntax. Die beim Setup aufgelösten Paketversionen stehen in `python\requirements.lock.txt`.

Prüfung von .NET-Konfiguration, Qdrant-gRPC und PDF-Extraktion ohne API-Key:

```powershell
dotnet run -c Release --no-build --project .\.NET\RetrievalAugmentedGeneration -- --check
```

## Qdrant

Stand 20.11.2026: neueste stabile Veröffentlichung **1.20.2**. Lokaler Container `qdrant`, HTTP-Port 6333, gRPC-Port 6334, Neustartregel `unless-stopped`. Daten liegen dauerhaft im Docker-Volume `rag_qdrant_storage`.

Dashboard: http://localhost:6333/dashboard

```powershell
docker compose up -d
docker compose ps
```

Die bisherigen zehn Collections wurden gesichert und erhalten, werden aber von den Projekten nicht verwendet. Sicherung: `backups\qdrant-1.15.4-20261120`; zusätzlich bleibt der ursprüngliche, gestoppte Container `qdrant-backup-1.15.4-20261120` erhalten. Niemals den alten und neuen Container gleichzeitig auf denselben Ports starten. Das Upgrade erfolgte über 1.15.5, 1.16.3, 1.17.1, 1.18.3 und 1.19.2 auf 1.20.2. Nach jedem Schritt wurden Status und Punktzahlen der bisherigen Collections geprüft.

Quellen: https://qdrant.tech/documentation/upgrades/ und https://openrouter.ai/docs/api/api-reference/embeddings/submit-an-embedding-request
