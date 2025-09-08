using OpenAI.Embeddings;
using Qdrant.Client;
using Qdrant.Client.Grpc;
using System.Text;
using System.Text.RegularExpressions;
using UglyToad.PdfPig;
using UglyToad.PdfPig.Content;

namespace RetrievalAugmentedGeneration;

public class HandleQDrant
{
    public async Task LoadDataToQdrant(string memoryCollectionName, string pathToPdfFiles, string openAiApiKey, string textEmbeddingModel, string pathToQdrant, int portToQdrant, ulong vectorSize = 3072)
    {
        // Ensure the collection exists, if not create it.
        // The C# client uses Qdrant's gRPC interface
        var client = new QdrantClient(pathToQdrant, portToQdrant);

        await client.GetCollectionInfoAsync(memoryCollectionName)
            .ContinueWith(t => 
            {
                if (t.IsFaulted || t.Result == null)
                {
                    Console.WriteLine($"Collection '{memoryCollectionName}' does not exist. Creating new collection.");
                }
                else
                {
                    Console.WriteLine($"Collection '{memoryCollectionName}' already exists. Skipping creation.");
                    return;
                }
            });

        await client.CreateCollectionAsync(collectionName: memoryCollectionName, vectorsConfig: new VectorParams
        {
            Size = vectorSize,
            Distance = Distance.Dot
        });

        EmbeddingClient embeddingClient = new(
            model: textEmbeddingModel,
            apiKey: openAiApiKey
        );

        // Ensure the directory exists
        if (Directory.Exists(pathToPdfFiles))
        {
            // Iterate through all PDF files in the directory
            foreach (string pdfFile in Directory.GetFiles(pathToPdfFiles, "*.pdf"))
            {
                string pdfContent = string.Empty;

                pdfContent = ExtractTextFromPdf(pdfFile);

                string documentId = Path.GetFileNameWithoutExtension(pdfFile);

                const int maxChunkSize = 1000;

                int chunkIndex = 0;
                foreach (var chunk in ChunkTextByWords(pdfContent, maxChunkSize))
                {
                    OpenAIEmbedding embedding = embeddingClient.GenerateEmbedding(input: chunk);
                    ReadOnlyMemory<float> vector = embedding.ToFloats();

                    var points = new List<PointStruct>
                    {
                        new()
                        {
                            Id = Guid.NewGuid(),
                            Vectors = vector.ToArray(),
                            Payload =
                            {
                                ["document_id"] = documentId,
                                ["chunk_index"] = chunkIndex,
                                ["text"] = chunk
                            }
                        }
                    };

                    var operationInfo = await client.UpsertAsync(
                        collectionName: memoryCollectionName,
                        points: points
                    );

                    chunkIndex++;
                }
                
                Console.WriteLine($"Imported PDF: {documentId}");
            }
        }
    }

    public async Task DeleteQdrantCollection(string pathToQdrant, int portToQdrant, string memoryCollectionName)
    {
        // The C# client uses Qdrant's gRPC interface
        var client = new QdrantClient(pathToQdrant, portToQdrant);
        await client.DeleteCollectionAsync(collectionName: memoryCollectionName);
    }

    private string ExtractTextFromPdf(string pdfFile)
    {
        using (PdfDocument document = PdfDocument.Open(pdfFile))
        {
            StringBuilder text = new();
            foreach (Page page in document.GetPages())
            {
                text.Append(page.Text);
            }
            return text.ToString();
        }
    }

    // Teilt Text in Chunks, die maxChunkSize nicht überschreiten.
    // Bevorzugt Trennung an Wortgrenzen; bricht extrem lange "Wörter" hart.
    private static IEnumerable<string> ChunkTextByWords(string text, int maxChunkSize)
    {
        if (string.IsNullOrWhiteSpace(text))
            yield break;

        // Whitespace (inkl. \r, \n, \t) zu einfachen Spaces vereinheitlichen
        text = Regex.Replace(text, @"\s+", " ").Trim();

        var sb = new StringBuilder();

        foreach (var word in text.Split(' '))
        {
            // Muss ein Leerzeichen vorangestellt werden?
            int extra = sb.Length == 0 ? 0 : 1;

            // Passt das Wort (inkl. Space) nicht mehr in den aktuellen Chunk?
            if (sb.Length > 0 && sb.Length + extra + word.Length > maxChunkSize)
            {
                yield return sb.ToString();
                sb.Clear();
                extra = 0;
            }

            // Falls ein einzelnes "Wort" größer als maxChunkSize ist -> hart splitten
            if (word.Length > maxChunkSize)
            {
                int idx = 0;
                while (idx < word.Length)
                {
                    int space = maxChunkSize - sb.Length - extra;
                    if (space <= 0)
                    {
                        if (sb.Length > 0)
                        {
                            yield return sb.ToString();
                            sb.Clear();
                        }
                        extra = 0;
                        space = maxChunkSize;
                    }

                    int take = Math.Min(space, word.Length - idx);
                    if (extra == 1) sb.Append(' ');
                    sb.Append(word.AsSpan(idx, take));
                    idx += take;
                    extra = 0;

                    if (sb.Length >= maxChunkSize)
                    {
                        yield return sb.ToString();
                        sb.Clear();
                    }
                }
            }
            else
            {
                if (extra == 1) sb.Append(' ');
                sb.Append(word);
            }
        }

        if (sb.Length > 0)
            yield return sb.ToString();
    }

}
