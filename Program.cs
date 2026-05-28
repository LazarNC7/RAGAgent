// ============================================================================
//    ollama pull nomic-embed-text
//    ollama pull mxbai-embed-large
//    ollama pull snowflake-arctic-embed2
// ============================================================================

using System.ClientModel;
using System.Diagnostics;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel.Connectors.InMemory;
using OpenAI;


var ollamaEndpoint = Environment.GetEnvironmentVariable("OLLAMA_ENDPOINT") ?? "http://localhost:11434";

var models = new[]
{
    new ModelConfig("nomic-embed-text",        Dims: 768,  ChunkSize: 1500, Overlap: 200),
    new ModelConfig("mxbai-embed-large",       Dims: 1024, ChunkSize: 400,  Overlap: 80),
    new ModelConfig("snowflake-arctic-embed2", Dims: 1024, ChunkSize: 1500, Overlap: 200),
};



var questions = new[]
{
    new Question(
        Query:          "How many days of annual leave do employees get?",
        ExpectedSource: "leave-policy.md",
        Category:       "HR",
        Hint:           "25"),

    new Question(
        Query:          "What is the hotel expense limit per night?",
        ExpectedSource: "expense-policy.md",
        Category:       "Finance",
        Hint:           "150"),

    new Question(
        Query:          "How do I reset my VPN?",
        ExpectedSource: "it-helpdesk-faq.md",
        Category:       "IT",
        Hint:           "certificate"),

    new Question(
        Query:          "What pension contribution does the company make?",
        ExpectedSource: "benefits-handbook.md",
        Category:       "HR",
        Hint:           "6%"),

    new Question(
        Query:          "What should I do in my first week at Acme?",
        ExpectedSource: "onboarding-guide.md",
        Category:       "HR",
        Hint:           "compliance"),

    new Question(
        Query:          "Does Acme CRM support REST API integration?",
        ExpectedSource: "product-faq.md",
        Category:       "Product",
        Hint:           "REST"),

    new Question(
        Query:          "How far in advance must I submit a leave request?",
        ExpectedSource: "leave-policy.md",
        Category:       "HR",
        Hint:           "2 weeks"),

    new Question(
        Query:          "What is the daily meal allowance when travelling?",
        ExpectedSource: "expense-policy.md",
        Category:       "Finance",
        Hint:           "50"),

    new Question(
        Query:          "Who do I contact for urgent IT problems?",
        ExpectedSource: "it-helpdesk-faq.md",
        Category:       "IT",
        Hint:           "5000"),

    new Question(
        Query:          "When do employee benefits start?",
        ExpectedSource: "benefits-handbook.md",
        Category:       "HR",
        Hint:           "probation"),

    new Question(
        Query:          "What learning budget am I entitled to?",
        ExpectedSource: "benefits-handbook.md",
        Category:       null,
        Hint:           "500"),

    new Question(
        Query:          "How do I submit expenses?",
        ExpectedSource: "expense-policy.md",
        Category:       null,
        Hint:           "Finance portal"),
};

var docsPath = Path.Combine(AppContext.BaseDirectory, "Docs");

Console.WriteLine("╔══════════════════════════════════════════════════════════════════╗");
Console.WriteLine("║          Ollama Embedding Model Benchmark — Acme Docs            ║");
Console.WriteLine("╚══════════════════════════════════════════════════════════════════╝");
Console.WriteLine($"Docs path : {docsPath}");
Console.WriteLine($"Questions : {questions.Length}");
Console.WriteLine($"Models    : {string.Join(", ", models.Select(m => m.Name))}");
Console.WriteLine();

var allResults = new List<ModelResult>();

foreach (var model in models)
{
    Console.WriteLine($"▶ Testing  {model.Name}  (dims={model.Dims}, chunkSize={model.ChunkSize})");
    Console.WriteLine(new string('─', 68));

    var result = await RunModelBenchmark(model, ollamaEndpoint, docsPath, questions);
    allResults.Add(result);

    PrintModelDetail(result, questions);
    Console.WriteLine();
}

PrintSummaryTable(allResults, questions.Length);
PrintWinner(allResults);

// ─── benchmark runner ─────────────────────────────────────────────────────

static async Task<ModelResult> RunModelBenchmark(
    ModelConfig model,
    string ollamaEndpoint,
    string docsPath,
    Question[] questions)
{
    var client = new OpenAIClient(
        new ApiKeyCredential("ollama"),
        new OpenAIClientOptions { Endpoint = new Uri($"{ollamaEndpoint.TrimEnd('/')}/v1") });

    var embeddingClient = client.GetEmbeddingClient(model.Name);

    var vectorStore = new InMemoryVectorStore();
    var collection  = vectorStore.GetCollection<Guid, Chunk>(
        $"bench-{model.Name}",
        new VectorStoreCollectionDefinition
        {
            Properties =
            [
                new VectorStoreKeyProperty(nameof(Chunk.Id),       typeof(Guid)),
                new VectorStoreDataProperty(nameof(Chunk.Text),     typeof(string)),
                new VectorStoreDataProperty(nameof(Chunk.Source),   typeof(string)),
                new VectorStoreDataProperty(nameof(Chunk.Category), typeof(string)) { IsIndexed = true },
                new VectorStoreVectorProperty(nameof(Chunk.Vector), model.Dims)
                {
                    Type             = typeof(ReadOnlyMemory<float>),
                    DistanceFunction = DistanceFunction.CosineSimilarity,
                },
            ]
        });

    await collection.EnsureCollectionExistsAsync();

    var ingestSw = Stopwatch.StartNew();
    int totalChunks = 0;

    foreach (var file in Directory.GetFiles(docsPath, "*.md"))
    {
        var text     = await File.ReadAllTextAsync(file);
        var category = InferCategory(Path.GetFileName(file));
        var chunks   = Chunker.Split(text, model.ChunkSize, model.Overlap).ToList();

        foreach (var chunk in chunks)
        {
            var result = await embeddingClient.GenerateEmbeddingAsync(chunk);
            var vector = new ReadOnlyMemory<float>(result.Value.ToFloats().ToArray());
            await collection.UpsertAsync(new Chunk
            {
                Text     = chunk,
                Source   = Path.GetFileName(file),
                Category = category,
                Vector   = vector,
            });
            totalChunks++;
        }
    }

    ingestSw.Stop();

    var questionResults = new List<QuestionResult>();
    var querySw         = Stopwatch.StartNew();

    foreach (var q in questions)
    {
        var qEmbedding = await embeddingClient.GenerateEmbeddingAsync(q.Query);
        var qVector    = new ReadOnlyMemory<float>(qEmbedding.Value.ToFloats().ToArray());

        var searchOptions = new VectorSearchOptions<Chunk>
        {
            Filter = !string.IsNullOrEmpty(q.Category)
                ? r => r.Category == q.Category
                : null,
        };

        var hits = new List<(string Source, double Score, string Text)>();
        await foreach (var hit in collection.SearchAsync(qVector,3, searchOptions))
            hits.Add((hit.Record.Source, hit.Score ?? 0, hit.Record.Text));

        var top1 = hits.FirstOrDefault();

        bool sourceMatch = top1.Source == q.ExpectedSource;
        bool hintMatch   = hits.Any(h => h.Text.Contains(q.Hint, StringComparison.OrdinalIgnoreCase));

        questionResults.Add(new QuestionResult(
            Question:    q,
            Top1Source:  top1.Source ?? "—",
            Top1Score:   top1.Score,
            SourceMatch: sourceMatch,
            HintMatch:   hintMatch,
            AllHits:     hits));
    }

    querySw.Stop();

    return new ModelResult(
        Model:           model,
        QuestionResults: questionResults,
        IngestMs:        ingestSw.ElapsedMilliseconds,
        TotalQueryMs:    querySw.ElapsedMilliseconds,
        TotalChunks:     totalChunks);
}


static void PrintModelDetail(ModelResult r, Question[] questions)
{
    foreach (var qr in r.QuestionResults)
    {
        var icon = qr.SourceMatch && qr.HintMatch ? "ok" :
                   qr.SourceMatch || qr.HintMatch ? "!" : "x";

        Console.WriteLine($"  {icon}  {qr.Question.Query,-52}");
        Console.WriteLine($"      Expected: {qr.Question.ExpectedSource,-30} Got: {qr.Top1Source,-30} Score: {qr.Top1Score:F4}");

        if (!qr.SourceMatch)
            Console.WriteLine($"      Source mismatch — hint found: {qr.HintMatch}");
    }

    int correct = r.QuestionResults.Count(q => q.SourceMatch && q.HintMatch);
    Console.WriteLine();
    Console.WriteLine($"  Accuracy     : {correct}/{questions.Length}  ({100.0 * correct / questions.Length:F0}%)");
    Console.WriteLine($"  Avg score    : {r.QuestionResults.Average(q => q.Top1Score):F4}");
    Console.WriteLine($"  Ingest time  : {r.IngestMs} ms  ({r.TotalChunks} chunks)");
    Console.WriteLine($"  Query time   : {r.TotalQueryMs} ms total  ({r.TotalQueryMs / (double)questions.Length:F0} ms/query avg)");
}

static void PrintSummaryTable(List<ModelResult> results, int totalQuestions)
{
    Console.WriteLine("╔══════════════════════════════════════════════════════════════════════════════╗");
    Console.WriteLine("║                              SUMMARY                                        ║");
    Console.WriteLine("╠════════════════════════════╦═══════════╦══════════╦══════════╦═════════════╣");
    Console.WriteLine("║ Model                      ║ Accuracy  ║ Avg Score║ Ingest ms║ ms/query    ║");
    Console.WriteLine("╠════════════════════════════╬═══════════╬══════════╬══════════╬═════════════╣");

    foreach (var r in results.OrderByDescending(r => r.Accuracy(totalQuestions)))
    {
        int    correct  = r.QuestionResults.Count(q => q.SourceMatch && q.HintMatch);
        double accuracy = 100.0 * correct / totalQuestions;
        double avgScore = r.QuestionResults.Average(q => q.Top1Score);
        double msPerQ   = r.TotalQueryMs / (double)totalQuestions;

        Console.WriteLine(
            $"║ {r.Model.Name,-26} ║ {correct}/{totalQuestions} ({accuracy:F0}%)  ║ {avgScore:F4}   ║ {r.IngestMs,8} ║ {msPerQ,11:F0} ║");
    }

    Console.WriteLine("╚════════════════════════════╩═══════════╩══════════╩══════════╩═════════════╝");
    Console.WriteLine();
}

static void PrintWinner(List<ModelResult> results)
{
    // ── Scoring rationale ────────────────────────────────────────────────
    //  Accuracy (80%) — did the right doc come back with the right answer?
    //                   Dominates because a wrong retrieval can't be fixed downstream.
    //
    //  Similarity (10%) — how confident is the model on its hits?
    //                      Uses rank (1st=1.0, 2nd=0.5, 3rd=0.0) so tiny raw
    //                      cosine differences don't distort the result.
    //
    //  Speed (10%)      — ms per query (not ingest — ingest is once-off at startup).
    //                      Fastest = 1.0, slowest = 0.0.
    // ─────────────────────────────────────────────────────────────────────

    var simRanks = results
        .OrderByDescending(r => r.QuestionResults.Average(q => q.Top1Score))
        .Select((r, i) => (r.Model.Name, SimRank: 1.0 - i * (1.0 / results.Count)))
        .ToDictionary(x => x.Name, x => x.SimRank);

    double maxQueryMs = results.Max(r => r.TotalQueryMs);
    double minQueryMs = results.Min(r => r.TotalQueryMs);
    double queryRange = maxQueryMs - minQueryMs;

    var scored = results.Select(r =>
    {
        double acc   = r.QuestionResults.Count(q => q.SourceMatch && q.HintMatch)
                       / (double)r.QuestionResults.Count;
        double sim   = simRanks[r.Model.Name];
        double speed = queryRange > 0
                       ? 1.0 - (r.TotalQueryMs - minQueryMs) / queryRange
                       : 1.0;
        double total = acc * 0.80 + sim * 0.10 + speed * 0.10;
        return (r.Model.Name, total, acc, sim, speed);
    }).OrderByDescending(x => x.total).ToList();

    Console.WriteLine("Composite score  (accuracy 80% · similarity rank 10% · query speed 10%)");
    Console.WriteLine($"  {"Model",-28} {"Composite",10} {"Accuracy",10} {"Sim rank",10} {"Speed",10} {"ms/q",8}");
    Console.WriteLine("  " + new string('-', 74));
    foreach (var s in scored)
    {
        double msPerQ = results.First(r => r.Model.Name == s.Name).TotalQueryMs
                        / (double)results.First(r => r.Model.Name == s.Name).QuestionResults.Count;
        Console.WriteLine($"  {s.Name,-28} {s.total,10:P0} {s.acc,10:P0} {s.sim,10:P0} {s.speed,10:P0} {msPerQ,8:F0}");
    }
    Console.WriteLine();

    var winner = scored.First();
    Console.WriteLine($"Recommended for your docs: {winner.Name}");
    Console.WriteLine($"  Composite: {winner.total:P0}  (accuracy {winner.acc:P0}, sim rank {winner.sim:P0}, speed {winner.speed:P0})");
    Console.WriteLine();
    Console.WriteLine("  Update Program.cs:");
    Console.WriteLine($"  OLLAMA_EMBED_MODEL={winner.Name}");
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


record ModelConfig(string Name, int Dims, int ChunkSize, int Overlap);

record Question(
    string  Query,
    string  ExpectedSource,
    string? Category,
    string  Hint);

record QuestionResult(
    Question                            Question,
    string                              Top1Source,
    double                              Top1Score,
    bool                                SourceMatch,
    bool                                HintMatch,
    List<(string Source, double Score, string Text)> AllHits);

record ModelResult(
    ModelConfig          Model,
    List<QuestionResult> QuestionResults,
    long                 IngestMs,
    long                 TotalQueryMs,
    int                  TotalChunks)
{
    public double Accuracy(int total) =>
        QuestionResults.Count(q => q.SourceMatch && q.HintMatch) / (double)total;
}

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