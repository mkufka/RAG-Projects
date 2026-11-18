using RetrievalAugmentedGeneration;
using OpenAI.Chat;
using OpenAI.Embeddings;
using Qdrant.Client;
using Qdrant.Client.Grpc;
using System.Text;
using QValue = Qdrant.Client.Grpc.Value;

DotNetEnv.Env.NoClobber().TraversePath().Load(Path.Combine(AppContext.BaseDirectory, ".env"));

string apiKey = GetRequiredSetting("OPENAI_API_KEY");
string memoryCollectionName = Environment.GetEnvironmentVariable("RAG_COLLECTION_NAME") ?? "CollectionWithData";
string pathToQdrant = Environment.GetEnvironmentVariable("QDRANT_HOST") ?? "localhost";
int portToQdrant = int.Parse(Environment.GetEnvironmentVariable("QDRANT_GRPC_PORT") ?? "6334");
if (portToQdrant is < 1 or > 65535)
    throw new InvalidOperationException("QDRANT_GRPC_PORT must be between 1 and 65535.");
string textEmbeddingModel = Environment.GetEnvironmentVariable("EMBEDDING_MODEL") ?? "text-embedding-3-large";
string chatModel = Environment.GetEnvironmentVariable("CHAT_MODEL") ?? "gpt-4o-mini";

/* ATTENTION:
* text-embedding-3-large provides a vector with 3072 dimensions
* text-embedding-3-small provides a vector with 1536 dimensions
*/
const int vectorSize = 3072;

string pathToPdfFiles = GetRequiredSetting("RAG_PDF_DIR");
if (!Directory.Exists(pathToPdfFiles))
    throw new DirectoryNotFoundException("RAG_PDF_DIR must point to an existing PDF directory.");

// Entry: first try to load data (ignore if already exists), then start interactive RAG chat.
await LoadIfNeededSafe();
await StartRagChatAsync();

// Try to (re)load PDFs into Qdrant, but don't crash if collection already exists.
static string GetRequiredSetting(string name)
{
    string? value = Environment.GetEnvironmentVariable(name);
    if (string.IsNullOrWhiteSpace(value) || value == "your_openai_api_key")
        throw new InvalidOperationException($"Set {name} in the repository root .env file or environment.");
    return value;
}

async Task LoadIfNeededSafe()
{
    try
    {
        HandleQDrant handleQDrant = new();
        await handleQDrant.LoadDataToQdrant(memoryCollectionName, pathToPdfFiles, apiKey, textEmbeddingModel, pathToQdrant, portToQdrant, vectorSize);
    }
    catch (Exception ex)
    {
        // If collection already exists or Qdrant reports a minor recoverable error, continue.
        if (!ex.Message.Contains("exist", StringComparison.OrdinalIgnoreCase))
        {
            Console.WriteLine($"Error: Could not load data: {ex.Message}");
        }
    }
}

async Task StartRagChatAsync()
{
    // Clients
    var qdrant = new QdrantClient(pathToQdrant, portToQdrant);
    EmbeddingClient embeddingClient = new(model: textEmbeddingModel, apiKey: apiKey);
    ChatClient chatClient = new(model: chatModel, apiKey: apiKey);

    // Conversation memory (keeps Q&A, not the raw retrieved context to avoid prompt bloat)
    var history = new List<ChatMessage>
    {
        new SystemChatMessage(
            "Du bist ein hilfreicher Assistent. Beantworte Fragen ausschließlich auf Basis der bereitgestellten Kontextausschnitte. " +
            "Wenn die Information nicht im Kontext steht, sage ehrlich, dass sie nicht vorhanden ist. Antworte kurz und präzise auf Deutsch. Damit Du auch Fragen zu Datums-Werten beantworten kannst, heute ist der " + DateTime.Now.ToString())
    };

    Console.WriteLine("Frage den Assistenten zu den PDF-Inhalten in Qdrant. Tippe 'exit' zum Beenden.\n");
    while (true)
    {
        Console.Write("Frage > ");
        string? question = Console.ReadLine();
        if (string.IsNullOrWhiteSpace(question))
            continue;
        if (string.Equals(question.Trim(), "exit", StringComparison.OrdinalIgnoreCase))
            break;

        try
        {
            // 1) Embed user query
            var queryEmbeddingResult = embeddingClient.GenerateEmbedding(input: question);
            var queryVectorMem = queryEmbeddingResult.Value.ToFloats();
            var queryVector = RagHelpers.MemoryToArray(queryVectorMem);

            // 2) Retrieve top-k relevant chunks from Qdrant with payload
            const int topK = 6;
            var results = await qdrant.SearchAsync(
                collectionName: memoryCollectionName,
                vector: queryVector,
                limit: topK);

            // 3) Build concise context block
            var context = RagHelpers.BuildContextFromResults(results);

            // 4) Ask chat model with context + conversation memory
            var messages = new List<ChatMessage>(history)
            {
                new UserChatMessage(
                    $"Frage: {question}\n\n" +
                    "Hier sind relevante Auszüge aus den Dokumenten. Nutze sie bei der Beantwortung:\n" +
                    context)
            };

            ChatCompletionOptions options = new()
            {
                Temperature = 0.7f,
                MaxOutputTokenCount = 500
            };

            // get the answer in a simple way:
            // ChatCompletion response = chatClient.CompleteChat(messages, options);
            // string answer = response.Content[0].Text;

            // or get the answer in a more structured way:
            var response = chatClient.CompleteChat(messages, options);
            string answer = response.Value?.Content is { } parts && parts.Count > 0
                ? string.Concat(parts.Select(part =>
                    (string?)part?.GetType().GetProperty("Text")?.GetValue(part) ?? part?.ToString() ?? string.Empty))
                : string.Empty;

            Console.WriteLine($"\nAntwort: {answer}");

            // 5) Quellenhinweis ausgeben
            string sources = RagHelpers.SummarizeSources(results);
            if (!string.IsNullOrWhiteSpace(sources))
            {
                Console.WriteLine($"\nQuellen:\n{sources}");
            }
            Console.WriteLine();

            // 6) Update memory with the summarized turn (question + answer)
            history.Add(new UserChatMessage(question));
            history.Add(new AssistantChatMessage(answer));
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error generating response: {ex.Message}\n");
        }
    }

    Console.WriteLine("Chat beendet.");}

internal static class RagHelpers
{
    public static string BuildContextFromResults(IReadOnlyList<ScoredPoint> results)
    {
        if (results == null || results.Count == 0)
            return "(Keine relevanten Treffer gefunden)";

        var sb = new StringBuilder();
        int idx = 1;
        foreach (var p in results)
        {
            string docId = TryGetString(p.Payload, "document_id") ?? "unbekannt";
            string chunkIndex = TryGetNumberString(p.Payload, "chunk_index") ?? "?";
            string text = TryGetString(p.Payload, "text") ?? string.Empty;

            if (!string.IsNullOrWhiteSpace(text))
            {
                sb.AppendLine($"[Kontext {idx}] Dokument: {docId}, Abschnitt: {chunkIndex}");
                sb.AppendLine(text.Trim());
                sb.AppendLine();
            }
            idx++;
        }
        return sb.ToString();
    }

    public static string? TryGetString(IDictionary<string, QValue>? payload, string key)
    {
        if (payload != null && payload.TryGetValue(key, out var val))
        {
            // Qdrant QValue hat verschiedene Werttypen - prüfe den KindCase
            try
            {
                // StringValue für String-Werte
                var strProp = typeof(QValue).GetProperty("StringValue");
                if (strProp != null)
                {
                    var s = strProp.GetValue(val) as string;
                    if (!string.IsNullOrEmpty(s)) return s;
                }

                // IntegerValue für Integer-Werte
                var intProp = typeof(QValue).GetProperty("IntegerValue");
                if (intProp != null)
                {
                    var intVal = intProp.GetValue(val);
                    if (intVal != null)
                    {
                        // Prüfe ob es ein gültiger Integer ist (nicht 0 bei nicht-integer Feldern)
                        var kindProp = typeof(QValue).GetProperty("KindCase");
                        if (kindProp != null)
                        {
                            var kind = kindProp.GetValue(val)?.ToString();
                            if (kind == "IntegerValue")
                                return intVal.ToString();
                        }
                    }
                }

                // DoubleValue für Double-Werte
                var doubleProp = typeof(QValue).GetProperty("DoubleValue");
                if (doubleProp != null)
                {
                    var kindProp = typeof(QValue).GetProperty("KindCase");
                    if (kindProp != null)
                    {
                        var kind = kindProp.GetValue(val)?.ToString();
                        if (kind == "DoubleValue")
                        {
                            var dblVal = doubleProp.GetValue(val);
                            return dblVal?.ToString();
                        }
                    }
                }
            }
            catch { }

            // Fallback: ToString() und versuchen JSON zu parsen
            var rawStr = val.ToString();
            if (rawStr != null && rawStr.Contains("integerValue"))
            {
                // Versuche { "integerValue": "X" } zu parsen
                var match = System.Text.RegularExpressions.Regex.Match(rawStr, @"""integerValue"":\s*""?(\d+)""?");
                if (match.Success)
                    return match.Groups[1].Value;
            }
            return rawStr;
        }
        return null;
    }

    public static string? TryGetNumberString(IDictionary<string, QValue>? payload, string key)
    {
        var s = TryGetString(payload, key);
        if (string.IsNullOrWhiteSpace(s)) return null;
        // Bereinige eventuelle Anführungszeichen
        return s.Trim('"', '\'', ' ');
    }

    public static float[] MemoryToArray(ReadOnlyMemory<float> mem)
    {
        var arr = new float[mem.Length];
        mem.Span.CopyTo(arr);
        return arr;
    }

    /// <summary>
    /// Fasst die Quellen der Suchergebnisse zusammen (wie in der Python-Version).
    /// </summary>
    public static string SummarizeSources(IReadOnlyList<ScoredPoint> results)
    {
        if (results == null || results.Count == 0)
            return string.Empty;

        var lines = new List<string>();
        var seen = new HashSet<string>();

        foreach (var p in results)
        {
            string docId = TryGetString(p.Payload, "document_id") ?? "unbekannt";
            string pageStart = TryGetNumberString(p.Payload, "page_start") ?? "?";
            string pageEnd = TryGetNumberString(p.Payload, "page_end") ?? "?";
            double score = p.Score;

            string line = $"- {docId} (S. {pageStart}-{pageEnd}) — Score {score:F3}";
            
            // Doppelte Zeilen vermeiden
            if (!seen.Contains(line))
            {
                lines.Add(line);
                seen.Add(line);
            }
        }

        return string.Join("\n", lines);
    }
}

