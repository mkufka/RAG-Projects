# Python-RAG

Die aktuelle Einrichtung und gemeinsame Konfiguration stehen in [README](../README.md).

Start aus dem Repository-Hauptverzeichnis:

```powershell
.\python\.venv\Scripts\python.exe .\python\run.py
```

`run.py` legt den Index bei Bedarf an, importiert die PDFs bei leerem Index und startet den Chat. Nur `OPENROUTER_API_KEY` in der zentralen `.env` ausfüllen.

Einzelschritte aus `python` mit `.\.venv\Scripts\python.exe`:

1. `step01_qdrant_setup.py`: Collection anlegen/Dimension und Distanz prüfen.
2. `step02_pdf_chunking.py`: lokale PDF-Extraktion und Token-Chunking als Vorschau (kein API-Key nötig).
3. `step03_embeddings.py`: Embedding-Vorschau über OpenRouter (API-Aufruf).
4. `step04_upsert_qdrant.py`: PDFs einlesen, Embeddings erzeugen und in Qdrant schreiben; Collection wird bei Bedarf angelegt.
5. `step05_chatbot.py`: vorhandenen Index abfragen.

Die Schritte 2 und 3 sind optionale Diagnosen. Schritt 4 führt diese Verarbeitung selbst aus. Wiederholte Python-Importe derselben Pfade verwenden stabile UUIDs, ersetzen aber keine veralteten Chunks bei verkürzten/gelöschten Dokumenten. Für eine vollständige Neuindizierung einen neuen Collection-Namen verwenden. .NET und Python erzeugen unterschiedliche Chunks; einen befüllten Index mit beiden Startprogrammen wiederverwenden, statt die Importe zu mischen.

Python nutzt 3072-dimensionale, normalisierte Embeddings, DOT-Distanz, MMR, optionale Dokumentfilter und optionales Streaming. Die Konfiguration ist in `../.env.example` dokumentiert. Beide Anwendungen können den gemeinsamen Index lesen.
