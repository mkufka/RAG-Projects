using System.Text;
using Qdrant.Client;
using Qdrant.Client.Grpc;
using UglyToad.PdfPig.Content;
using UglyToad.PdfPig;
using OpenAI.Embeddings;

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

                var lines = pdfContent.Split('\n');
                var currentChunk = new StringBuilder();
                var chunkIndex = 0;
                foreach (var line in lines)
                {
                    if (currentChunk.Length > 0 && currentChunk.Length + line.Length > maxChunkSize)
                    {
                        OpenAIEmbedding embedding = embeddingClient.GenerateEmbedding(input: currentChunk.ToString());
                        ReadOnlyMemory<float> vector = embedding.ToFloats();

                        // Convert to a List<PointStruct>
                        var points = new List<PointStruct>
                        {
                            new()
                            {
                                Id = Guid.NewGuid(), // Or use a numeric ID if preferred
                                Vectors = vector.ToArray(), // Convert ReadOnlyMemory<float> to float[]
                                Payload =
                                {
                                    ["document_id"] = documentId,
                                    ["chunk_index"] = chunkIndex,
                                    ["text"] = currentChunk.ToString()
                                }
                            }
                        };

                        // Now you can call UpsertAsync
                        var operationInfo = await client.UpsertAsync(
                            collectionName: memoryCollectionName,
                            points: points
                        );


                        currentChunk.Clear();
                        chunkIndex++;
                    }
                    currentChunk.AppendLine(line);
                }

                if (currentChunk.Length > 0)
                {
                    //await memory.SaveInformationAsync(memoryCollectionName, currentChunk.ToString(),
                    //    $"{documentId}-chunk-{chunkIndex}");
                    OpenAIEmbedding embedding = embeddingClient.GenerateEmbedding(input: currentChunk.ToString());
                    ReadOnlyMemory<float> vector = embedding.ToFloats();

                    // Convert to a List<PointStruct>
                    var points = new List<PointStruct>
                    {
                        new()
                        {
                            Id = Guid.NewGuid(), // Or use a numeric ID if preferred
                            Vectors = vector.ToArray(), // Convert ReadOnlyMemory<float> to float[]
                            Payload =
                            {
                                ["document_id"] = documentId,
                                ["chunk_index"] = chunkIndex,
                                ["text"] = currentChunk.ToString()
                            }
                        }
                    };

                    // Now you can call UpsertAsync
                    var operationInfo = await client.UpsertAsync(
                        collectionName: memoryCollectionName,
                        points: points
                    );
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
}
