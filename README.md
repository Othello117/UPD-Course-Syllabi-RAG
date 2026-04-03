---
title: UPD CRS Assistant
emoji: 👀
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 6.11.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Chat with a helpful and knowledgeable UPD CRS assistant
---

# UPD CRS Assistant (RAG Pipeline)

This repository contains the RAG (Retrieval-Augmented Generation) pipeline configurations and specifications for the UPD CRS Assistant, designed to help students with UP Diliman Math Department course information.

## Specifications and Process

### 1. Data Sources and Chunking

- **UPD Math Syllabi**: 
  - Raw syllabi text processed using `RecursiveCharacterTextSplitter`.
- **UPD CRS Math Data**: 
  - Chunked row by row.
  - Performed feature engineering.
  - Aggregated by specific class and professor.
  - Formatted into chunks. Any chunks that exceeded the token limit of the embedding model were subsequently split.

### 2. Embeddings

- **Model**: `BAAI/bge-m3`
- **Vector Store**: Pinecone (`PineconeVectorStore`)
- All processed chunks and metadata are embedded and indexed here to enable semantic search.

### 3. Routing Logic

The system utilizes an intelligent routing chain to determine the intent and scope of the user's question before deciding how to retrieve context:
- The router analyzes the chat history and the user's latest question, outputting a structured JSON decision.
- **Is Math Course Related**: If the query is unrelated to Math courses, it gets routed directly to the LLM's general knowledge.
- **Broad vs. Specific Query (`is_broad`)**: 
  - If the query is asking for a general overview, list of classes, or multiple courses (`is_broad: true`), the retriever applies a metadata filter to only search Pinecone for `summary_type` matching `"Class"` or `"Professor"`.
  - If the query is asking about specific details like prerequisites or syllabus rules, the retriever filters for specific document chunks (`summary_type` does not exist).
- The router also rewrites the query into a standalone `search_query` for optimized vector search (`k=5`).

### 4. LLM Model

- **Model**: `Qwen/Qwen2.5-7B-Instruct`
- Accessed via Hugging Face Endpoints (`HuggingFaceEndpoint` / `ChatHuggingFace`).
- Configured with a low temperature (0.1) to ensure consistent JSON routing and factual RAG responses.

### 5. Other Technical Details

- **Framework**: LangChain & Gradio
- **Memory Handling**: Retains and formats the chat history, passing it both to the Router for query disambiguation and to the final RAG Prompt for contextual answering.
- **Deployment**: Hosted on Hugging Face Spaces using the Gradio SDK.
