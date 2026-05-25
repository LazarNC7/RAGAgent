# RAGAgent
A C# console RAG assistant built with Microsoft.Extensions.AI. It ingests local Markdown docs into an in-memory vector store and uses an AI agent with tool-calling to answer company questions. Built on standard .NET abstractions, it easily ports to Ollama by swapping the OpenAI clients for local ones and updating vector dimensions.
