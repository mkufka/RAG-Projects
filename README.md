
<img width="384" height="256" alt="RAG" src="https://github.com/user-attachments/assets/ec5032ea-7a14-4ae4-bd80-73b44403b102" />

# RAG Implementierungen mit Qdrant

In diesem Repository befinden sich zwei unterschiedliche Implementierungen des Retrieval-Augmented Generation (RAG) Konzepts mit PDF-Dateien:

- **.NET Ordner**: Enthält eine C# Implementierung von RAG.  
  Installation und Start: [.NET/README.md](.NET/README.md).

- **python Ordner**: Enthält eine Python Implementierung von RAG.  
  Installation und Start: [python/README.md](python/README.md).

Jede Implementierung ist eigenständig und besitzt eine eigene README Datei mit spezifischen Informationen.

Für beide Implementierungen ist eine Qdrant-Vektor-Datenbank vorgesehen.

## Voraussetzungen

- **.NET:** SDK mit Unterstützung für das Ziel `net9.0` und eine .NET-9-Runtime.
- **Python:** Python 3.12 oder neuer; die bestehende Umgebung wurde mit Python 3.14.2 geprüft.
- **Qdrant:** erreichbarer Server; aktuell wird Version **1.20.2** verwendet.
- **OpenAI:** gültiger API-Key mit API-Guthaben und Zugriff auf die konfigurierten Modelle.
- **Dokumente:** PDFs mit extrahierbarem Text. Eine OCR-Erkennung für reine Scans ist nicht enthalten.

## Zentrale Konfiguration

Beide Anwendungen laden dieselbe `.env` im Repository-Hauptverzeichnis. Kopiere
[.env.example](.env.example) nach `.env` und trage dort Deinen OpenAI-API-Key,
den absoluten PDF-Verzeichnispfad und die Qdrant-Verbindungsdaten ein.
Bereits gesetzte Umgebungsvariablen haben Vorrang. Die `.env` wird nicht von Git erfasst.

Einmalig aus dem Repository-Hauptverzeichnis (PowerShell; eine vorhandene Datei bleibt erhalten):

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Anschließend mindestens `OPENAI_API_KEY` und `RAG_PDF_DIR` setzen.
Für lokal gestartete Anwendungen mit lokalem Qdrant verwende `QDRANT_HOST=localhost`
und `QDRANT_GRPC_PORT=6334`. Beide Implementierungen erwarten 3072-dimensionale
Embeddings, standardmäßig mit `text-embedding-3-large`.
Die in der Vorlage als Python-Optionen markierten Chunking- und Chatparameter
werden von der .NET-Anwendung nicht ausgewertet.

Die bisherige `python/.env` wird nicht mehr geladen. Bei einer bestehenden Installation
verschiebe sie ins Repository-Hauptverzeichnis, sofern dort noch keine `.env` liegt;
andernfalls führe die Einstellungen zusammen, ohne die bestehende Datei zu überschreiben.
Da beide Anwendungen denselben Collection-Namen laden, greifen sie auf denselben Index zu.
Ein bereits befüllter Index muss nicht mit beiden Anwendungen erneut importiert werden.
Die Chunking-Verfahren unterscheiden sich. Wiederholte Python-Importe erzeugen neue
UUIDs und damit zusätzliche, gegebenenfalls doppelte Einträge. .NET importiert beim
Start nur, wenn die Collection noch nicht existiert; auch eine vorhandene leere
Collection wird nicht automatisch befüllt. Weitere Hinweise stehen in den jeweiligen Anleitungen.

Beide Anwendungen sind Konsolenprogramme; den Chat beendest Du mit `exit`.
PDF-Texte und Fragen werden zur Embedding-Erzeugung und Beantwortung an OpenAI übertragen;
dabei können API-Kosten entstehen.

*Bitte beachten: Die Implementierungen funktionieren ausschließlich mit PDF-Dateien!*
---
