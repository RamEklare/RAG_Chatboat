# RAG_Chatboat

           ┌─────────────────────────┐
           │      User Uploads       │
           │     Factsheet PDF        │
           └───────────┬─────────────┘
                       │
                       ▼
           ┌─────────────────────────┐
           │   PDF Processing Layer  │
           │ pdfplumber + OCR + tables│
           └───────────┬─────────────┘
                       │ Extracted text & tables
                       ▼
         ┌──────────────────────────────┐
         │   Chunking / Snippet Maker   │
         └───────────┬──────────────────┘
                     │  chunks
                     ▼
      ┌──────────────────────────────────────┐
      │ Embedding Generation (HuggingFace)   │
      └──────────────────┬───────────────────┘
                          │  vectors
                          ▼
               ┌──────────────────┐
               │   FAISS Index    │
               └──────┬──────────┘
                      │  top-k vectors
                      ▼
       ┌────────────────────────────────┐
       │   Retrieval + Rule-based Answer│
       │ (No API, No LLM)               │
       └─────────────────┬──────────────┘
                         ▼
                ✔ Final Answer using
                retrieved facts only
