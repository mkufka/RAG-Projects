## Verwendung der C#-Implementierung (Retrieval Augmented Generation)

Dieses Verzeichnis enthält eine Beispielimplementierung für Retrieval Augmented Generation (RAG) in C# mit einer Qdrant-Vektor-Datenbank.

### Voraussetzungen & Installation

Benötigt werden:

- Ein .NET-SDK mit Unterstützung für `net9.0` sowie eine .NET-9-Runtime.
- Eine erreichbare Qdrant-Instanz, aktuell Version **1.20.2**, mit gRPC-Port **6334**.
- Ein OpenAI-API-Key mit API-Guthaben.
- Ein vorhandenes Verzeichnis mit PDFs mit extrahierbarem Text; OCR ist nicht enthalten.

Die NuGet-Abhängigkeiten sind bereits in
[RetrievalAugmentedGeneration.csproj](RetrievalAugmentedGeneration/RetrievalAugmentedGeneration.csproj)
deklariert: Qdrant.Client, PdfPig, OpenAI und DotNetEnv. Ein `dotnet restore` genügt;
die Pakete müssen nicht mit `dotnet add package` erneut hinzugefügt werden.

Alle folgenden PowerShell-Befehle werden aus dem Repository-Hauptverzeichnis ausgeführt.

### Nutzung

1. **Projekt öffnen**

   Öffne die [Solution-Datei](RetrievalAugmentedGeneration.sln) mit Visual Studio
   oder das Repository mit VS Code.

2. **Projekt bauen**

   ```powershell
   dotnet restore ".\.NET\RetrievalAugmentedGeneration\RetrievalAugmentedGeneration.csproj"
   dotnet build ".\.NET\RetrievalAugmentedGeneration\RetrievalAugmentedGeneration.csproj" --no-restore
   ```

3. **Konfigurationsdaten anpassen**

   Kopiere einmalig die zentrale Vorlage [.env.example](../.env.example) nach `.env`
   im Repository-Hauptverzeichnis. Beide Implementierungen verwenden diese Datei.

   ```env
   OPENAI_API_KEY=dein_api_key
   RAG_PDF_DIR='C:\Pfad\zu\PDFs'
   QDRANT_HOST=localhost
   QDRANT_GRPC_PORT=6334
   RAG_COLLECTION_NAME=CollectionWithData
   EMBEDDING_MODEL=text-embedding-3-large
   CHAT_MODEL=gpt-4o-mini
   ```

   Bereits gesetzte Umgebungsvariablen haben Vorrang. `OPENAI_API_KEY` und
   `RAG_PDF_DIR` sind erforderlich; das PDF-Verzeichnis muss existieren.
   Die anderen Werte besitzen die oben gezeigten Standardwerte.
   Der Index erwartet weiterhin 3072 Dimensionen; verwende ein dazu passendes Embedding-Modell.
   `QDRANT_HOST` ist ein Hostname ohne URL-Schema; `QDRANT_GRPC_PORT` muss zwischen 1 und 65535 liegen.
   Die zusätzlichen Python-Chunking- und Chatoptionen in der Vorlage werden von .NET nicht ausgewertet.
   Die Datei wird vom Ausgabeverzeichnis aus in übergeordneten Verzeichnissen gesucht,
   unabhängig vom Arbeitsverzeichnis. Bei einer Veröffentlichung außerhalb des Repositorys
   muss die `.env` neben der Anwendung oder in einem übergeordneten Verzeichnis liegen;
   alternativ können die Werte als Umgebungsvariablen gesetzt werden.
   Zugangsdaten gehören nicht in den Quelltext oder ins Git-Repository.

4. **Projektstruktur**

   [Program.cs](RetrievalAugmentedGeneration/Program.cs) lädt die Konfiguration und startet
   Import und Chat. [HandleQDrant.cs](RetrievalAugmentedGeneration/HandleQDrant.cs)
   liest PDFs und schreibt die Embeddings nach Qdrant.

5. **Eigene Anpassungen**

   Passe die Implementierung nach Bedarf an, um eigene Dokumente oder Datenquellen einzubinden.

6. **Ausführen**

   ```powershell
   dotnet run --no-restore --project ".\.NET\RetrievalAugmentedGeneration\RetrievalAugmentedGeneration.csproj"
   ```

   Alternativ kann in VS Code die vorhandene `.NET Launch`-Konfiguration verwendet werden.
   Es startet ein interaktiver Konsolenchat; `exit` beendet ihn.

### Verhalten beim Import

- Ist die konfigurierte Collection noch nicht vorhanden, wird sie angelegt und befüllt.
  Es werden nur PDFs direkt im konfigurierten Verzeichnis gelesen, nicht in Unterverzeichnissen.
- Existiert die Collection bereits, werden aktuell keine PDFs nachimportiert, auch wenn sie leer ist.
  Für den Erstimport entweder .NET vor dem Python-Setup starten oder die vorhandene Collection
  mit dem Python-Importschritt befüllen.
- Beide Anwendungen können denselben befüllten Index verwenden. Die .NET-Importdaten
  enthalten keine Seitenmetadaten; Quellenangaben können deshalb `?` als Seitenzahl zeigen.
- Import und Chat übertragen Dokumenttexte beziehungsweise Fragen an OpenAI und können API-Kosten verursachen.

### Troubleshooting

**NuGet-Restore schlägt mit HTTP 401 fehl**

Eine konfigurierte Paketquelle verlangt eine Anmeldung. Stelle den Zugriff auf diese Quelle wieder her.
Falls für dieses Projekt die öffentliche NuGet-Quelle verwendet werden darf, kann der Restore
ohne Änderung der globalen NuGet-Konfiguration so ausgeführt werden:

```powershell
dotnet restore ".\.NET\RetrievalAugmentedGeneration\RetrievalAugmentedGeneration.csproj" --source https://api.nuget.org/v3/index.json
```

**Konfiguration fehlt oder ist ungültig**

Prüfe die zentrale `.env`, insbesondere `OPENAI_API_KEY`, `RAG_PDF_DIR` und `QDRANT_GRPC_PORT`.
Eine fehlende oder leere API-Key-Einstellung, ein nicht vorhandenes PDF-Verzeichnis und ungültige
Ports verhindern den Start. Ein tatsächlich gültiger API-Key wird erst bei einem OpenAI-Aufruf geprüft.
