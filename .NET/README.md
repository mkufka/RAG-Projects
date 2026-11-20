# .NET-RAG

Die aktuelle Einrichtung und gemeinsame Konfiguration stehen in [README](../README.md).

Aus dem Repository-Hauptverzeichnis:

```powershell
dotnet build .\.NET\RetrievalAugmentedGeneration.sln -c Release --source https://api.nuget.org/v3/index.json
.\Start-DotNet.ps1
```

Nur `OPENROUTER_API_KEY` in der zentralen `.env` ausfüllen. Import und Chat verwenden OpenRouter für das Embedding- und Chatmodell. Die `.env` wird vom Anwendungsverzeichnis aus aufwärts gesucht; Umgebungsvariablen haben Vorrang.

Beim Start wird eine fehlende Collection erstellt und eine leere Collection befüllt. Bereits befüllte Collections werden wiederverwendet. Fehler beim Import brechen den Start ab. PDFs werden rekursiv gesucht. Der Import nutzt Zeichen-Chunking; .NET schreibt keine Seitenmetadaten. Bei .NET-importierten Daten zeigen Quellen gegebenenfalls `?` als Seitenzahl. Nach einem unterbrochenen Teilimport einen neuen Collection-Namen für einen vollständigen Neuimport verwenden.

Diagnose ohne Key (PDF-Extraktion, Konfiguration, Qdrant-gRPC):

```powershell
dotnet run -c Release --no-build --project .\.NET\RetrievalAugmentedGeneration -- --check
```

Zum Beenden des Chats `exit` eingeben. Ziel ist `net10.0`; benötigt wird eine passende Runtime und ein kompatibles SDK.
