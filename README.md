# RAGAgent

A lightweight, modular C# console application that implements a Retrieval-Augmented Generation (RAG) pipeline using the modern `.NET AI` ecosystem (`Microsoft.Extensions.AI` and `Microsoft.Extensions.VectorData`). 

By default, this project uses **GitHub Models** (OpenAI's `gpt-4o-mini` and `text-embedding-3-small`) to act as an intelligent internal assistant for "Acme Software Ltd." It automatically ingests local Markdown documentation, stores it in an in-memory vector database, and uses autonomous tool-calling to provide grounded, highly accurate answers to user queries.

---

## Features

* **Automated Document Ingestion:** Automatically reads, chunks (1500 chars / 200 overlap), and embeds Markdown files from a local `Docs` folder.
* **Semantic Vector Search:** Uses `InMemoryVectorStore` to query document chunks via Cosine Similarity.
* **Autonomous Tool Calling:** The AI agent automatically decides when to invoke the `search_knowledge_base` tool to retrieve context before answering.
* **Dynamic Categorization:** Infers document categories (HR, IT, Finance, Product) based on filenames for optimized filtering.
* **Real-time Streaming:** Streams the LLM's response directly to the console for a fast, conversational user experience.
