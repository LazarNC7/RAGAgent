using System.ClientModel;
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel.Connectors.InMemory;
using OpenAI;
using OpenAI.Chat;

//    nomic-embed-text        dims=768   ctx=8192  
//    mxbai-embed-large       dims=1024  ctx=512   
//    snowflake-arctic-embed2 dims=1024  ctx=8192  


var ollamaEndpoint   = Environment.GetEnvironmentVariable("OLLAMA_ENDPOINT")    ?? "http://localhost:11434";
var chatModelId      = Environment.GetEnvironmentVariable("OLLAMA_CHAT_MODEL")  ?? "llama3.2";
var embeddingModelId = Environment.GetEnvironmentVariable("OLLAMA_EMBED_MODEL") ?? "snowflake-arctic-embed2";

var embeddingDims = embeddingModelId switch
{
    "nomic-embed-text"        => 768,
    "mxbai-embed-large"       => 1024,
    "snowflake-arctic-embed2" => 1024,
    _ => throw new InvalidOperationException(
        $"Unknown embedding model '{embeddingModelId}'. " +
        "Add its dimension to the switch or set a known model.")
};

var chunkSize = embeddingModelId == "mxbai-embed-large" ? 400 : 1500;
var overlap   = embeddingModelId == "mxbai-embed-large" ? 80  : 200;

Console.WriteLine($"Embedding model : {embeddingModelId}  (dims={embeddingDims}, chunkSize={chunkSize})");
Console.WriteLine($"Chat model      : {chatModelId}");
Console.WriteLine($"Ollama endpoint : {ollamaEndpoint}");
Console.WriteLine();


var openAIClient = new OpenAIClient(
    new ApiKeyCredential("ollama"),               
    new OpenAIClientOptions
    {
        Endpoint = new Uri($"{ollamaEndpoint.TrimEnd('/')}/v1")
    });


var embeddingClient  = openAIClient.GetEmbeddingClient(embeddingModelId);

OpenAI.Embeddings.EmbeddingGenerationOptions? embeddingOptions = null;

var vectorStore = new InMemoryVectorStore();
var collection  = vectorStore.GetCollection<Guid, Chunk>(
    "company-docs",
    new VectorStoreCollectionDefinition
    {
        Properties =
        [
            new VectorStoreKeyProperty(nameof(Chunk.Id), typeof(Guid)),
            new VectorStoreDataProperty(nameof(Chunk.Text),     typeof(string)),
            new VectorStoreDataProperty(nameof(Chunk.Source),   typeof(string)),
            new VectorStoreDataProperty(nameof(Chunk.Category), typeof(string)) { IsIndexed = true },
            new VectorStoreVectorProperty(nameof(Chunk.Vector), embeddingDims)
            {
                Type             = typeof(ReadOnlyMemory<float>),
                DistanceFunction = DistanceFunction.CosineSimilarity,
            },
        ]
    });

await collection.EnsureCollectionExistsAsync();

var docsPath = Path.Combine(AppContext.BaseDirectory, "Docs");

Console.WriteLine($"Ingesting documents from {docsPath}...");

foreach (var file in Directory.GetFiles(docsPath, "*.md"))
{
    var text     = await File.ReadAllTextAsync(file);
    var category = InferCategory(Path.GetFileName(file));
    var chunks   = Chunker.Split(text, chunkSize, overlap);
    var count    = 0;

    foreach (var chunk in chunks)
    {
        var result = await embeddingClient.GenerateEmbeddingAsync(chunk, embeddingOptions);
        var vector = new ReadOnlyMemory<float>(result.Value.ToFloats().ToArray());
        await collection.UpsertAsync(new Chunk
        {
            Text     = chunk,
            Source   = Path.GetFileName(file),
            Category = category,
            Vector   = vector,
        });
        count++;
    }

    Console.WriteLine($"  {Path.GetFileName(file),30}  →  {count} chunk(s)  [{category}]");
}

Console.WriteLine($"\nReady — chunks indexed.\n");


var kb = new KnowledgeBase(collection, embeddingClient);

var agent = openAIClient
    .GetChatClient(chatModelId)
    .AsAIAgent(
        instructions:
            "You are a helpful company assistant for Acme Software Ltd. " +
            "For any question about internal policies, IT, HR, benefits, expenses, or products, " +
            "always call search_knowledge_base before answering. " +
            "You may pass a category filter (HR, IT, Finance, Product) when the topic is clear. " +
            "If the knowledge base returns no relevant result, say so clearly rather than guessing. " +
            "Cite the source document when you use retrieved information.",
        name: "CompanyAssistant",
        tools: [AIFunctionFactory.Create(kb.SearchAsync, name: "search_knowledge_base")]);

var session = await agent.CreateSessionAsync();

Console.WriteLine($"Company assistant ready (model: {chatModelId}).");
Console.WriteLine("Ask anything about company policies, IT, or HR. Type 'exit' to quit.\n");


while (true)
{
    Console.Write("You: ");
    var input = Console.ReadLine();
    if (string.IsNullOrWhiteSpace(input)) continue;
    if (input.Equals("exit", StringComparison.OrdinalIgnoreCase)) break;

    Console.Write("Assistant: ");
    await foreach (var update in agent.RunStreamingAsync(input, session))
        Console.Write(update.Text);

    Console.WriteLine("\n");
}


static string InferCategory(string filename) => filename switch
{
    var f when f.StartsWith("leave")
            || f.StartsWith("benefit")
            || f.StartsWith("onboarding") => "HR",
    var f when f.StartsWith("it-")        => "IT",
    var f when f.StartsWith("expense")    => "Finance",
    var f when f.StartsWith("product")    => "Product",
    _                                     => "General",
};


class Chunk
{
    [VectorStoreKey]
    public Guid Id { get; set; } = Guid.NewGuid();

    [VectorStoreData]
    public string Text { get; set; } = "";

    [VectorStoreData]
    public string Source { get; set; } = "";

    [VectorStoreData(IsIndexed = true)]
    public string Category { get; set; } = "";

    public ReadOnlyMemory<float> Vector { get; set; }
}

static class Chunker
{
    public static IEnumerable<string> Split(string text, int chunkSize, int overlap)
    {
        int start = 0;
        while (start < text.Length)
        {
            int end = Math.Min(start + chunkSize, text.Length);
            yield return text[start..end];
            if (end == text.Length) break;
            start += chunkSize - overlap;
        }
    }
}

class KnowledgeBase(
    VectorStoreCollection<Guid, Chunk> collection,
    OpenAI.Embeddings.EmbeddingClient embeddingClient)
{
    [Description(
        "Search the company knowledge base for information about HR policies, IT support, " +
        "benefits, onboarding, expenses, or products. " +
        "Always call this tool before answering questions about internal company topics.")]
    public async Task<string> SearchAsync(
        [Description("The user's question, rephrased as a focused search query")] string query,
        [Description("Optional category filter: HR, IT, Finance, Product. Leave empty to search all.")] string? category = null,
        [Description("Number of results to return. Default is 3.")] int topK = 3)
    {
        var result      = await embeddingClient.GenerateEmbeddingAsync(query);
        var queryVector = new ReadOnlyMemory<float>(result.Value.ToFloats().ToArray());

        var options = new VectorSearchOptions<Chunk>
        {
            Filter = !string.IsNullOrEmpty(category)
                ? r => r.Category == category
                : null,
        };

        var results = new List<string>();
        await foreach (var hit in collection.SearchAsync(queryVector, topK, options))
            results.Add($"[Source: {hit.Record.Source} | Score: {hit.Score:P0}]\n{hit.Record.Text}");

        return results.Count > 0
            ? string.Join("\n\n---\n\n", results)
            : "No relevant information found in the knowledge base for this query.";
    }
}