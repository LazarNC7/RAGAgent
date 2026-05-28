using System.ClientModel;
using System.Text.Json;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel.Connectors.InMemory;
using OpenAI;

var ollamaEndpoint = Environment.GetEnvironmentVariable("OLLAMA_ENDPOINT") ?? "http://localhost:11434";
var docsPath       = Path.Combine(AppContext.BaseDirectory, "Docs");
var outputPath     = Path.Combine(AppContext.BaseDirectory, "vectors.json");

var models = new[]
{
    new ModelConfig("nomic-embed-text",        Dims: 768,  ChunkSize: 1500, Overlap: 200),
    new ModelConfig("mxbai-embed-large",       Dims: 1024, ChunkSize: 400,  Overlap: 80),
    new ModelConfig("snowflake-arctic-embed2", Dims: 1024, ChunkSize: 1500, Overlap: 200),
};

Console.WriteLine("╔══════════════════════════════════════════════════════════════════╗");
Console.WriteLine("║              Vector Exporter — Acme Docs                        ║");
Console.WriteLine("╚══════════════════════════════════════════════════════════════════╝");
Console.WriteLine($"Docs   : {docsPath}");
Console.WriteLine($"Output : {outputPath}");
Console.WriteLine();

// Top-level export: list of { model, chunks: [ { source, category, text_preview, vector } ] }
var export = new List<ModelExport>();

foreach (var model in models)
{
    Console.WriteLine($"Embedding with {model.Name}...");

    var client = new OpenAIClient(
        new ApiKeyCredential("ollama"),
        new OpenAIClientOptions { Endpoint = new Uri($"{ollamaEndpoint.TrimEnd('/')}/v1") });

    var embeddingClient = client.GetEmbeddingClient(model.Name);
    var chunks          = new List<ChunkExport>();

    foreach (var file in Directory.GetFiles(docsPath, "*.md").OrderBy(f => f))
    {
        var text     = await File.ReadAllTextAsync(file);
        var source   = Path.GetFileName(file);
        var category = InferCategory(source);
        var textChunks = Chunker.Split(text, model.ChunkSize, model.Overlap).ToList();

        for (int i = 0; i < textChunks.Count; i++)
        {
            var chunk  = textChunks[i];
            var result = await embeddingClient.GenerateEmbeddingAsync(chunk);
            var vector = result.Value.ToFloats().ToArray();

            chunks.Add(new ChunkExport(
                Source:      source,
                Category:    category,
                ChunkIndex:  i,
                TextPreview: chunk.Length > 80 ? chunk[..80].Replace("\n", " ") + "…" : chunk.Replace("\n", " "),
                Vector:      vector));

            Console.Write(".");
        }
    }

    export.Add(new ModelExport(model.Name, model.Dims, chunks));
    Console.WriteLine($"  {chunks.Count} chunks");
}

Console.WriteLine();
Console.WriteLine($"Writing {outputPath}...");

var json = JsonSerializer.Serialize(export, new JsonSerializerOptions { WriteIndented = false });
await File.WriteAllTextAsync(outputPath, json);

Console.WriteLine($"Done. {new FileInfo(outputPath).Length / 1024} KB written.");
Console.WriteLine();
Console.WriteLine("Next step:");
Console.WriteLine("  pip install umap-learn plotly pandas numpy");
Console.WriteLine("  python visualize.py");


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


record ModelConfig(string Name, int Dims, int ChunkSize, int Overlap);

record ModelExport(string Model, int Dims, List<ChunkExport> Chunks);

record ChunkExport(
    string   Source,
    string   Category,
    int      ChunkIndex,
    string   TextPreview,
    float[]  Vector);

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