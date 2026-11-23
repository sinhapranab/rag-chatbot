# PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) application that lets users ask
questions about PDF documents. It extracts text from PDFs, splits the
text into smaller chunks, converts those chunks into vector embeddings,
stores them in ChromaDB, retrieves relevant content, and uses Google
Gemini to generate answers grounded in the retrieved document context.

## Features

-   Load PDF documents with LangChain's `PyPDFLoader`.
-   Split document text into chunks using
    `RecursiveCharacterTextSplitter`.
-   Generate embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
-   Store and retrieve vectors using ChromaDB.
-   Retrieve relevant chunks using Maximum Marginal Relevance (MMR).
-   Generate context-based answers using Google Gemini.
-   Use either a command-line chat flow or a Streamlit PDF-upload
    interface.
-   The Streamlit interface displays the source page numbers for
    retrieved answers.

## Tech Stack

-   **Python**
-   **LangChain**
-   **Google Gemini** (`gemini-3.5-flash-lite`)
-   **Hugging Face Sentence Transformers**
-   **ChromaDB**
-   **Streamlit** (web interface)
-   **PyPDF**

## Project Structure

The following is a suggested layout based on the files in this project.
Adjust filenames or folders if your local project uses a different
layout.

``` text
pdf-rag-chatbot/
│
├── documents/
│   └── node_js.pdf
│
├── create_database.py
├── main.py
├── app.py
├── .env
├── .gitignore
└── README.md
```

The vector database folders are created when the application runs:

-   `chroma_db/` --- used by `create_database.py` and the command-line
    `main.py`.
-   `chroma_db_ui/` --- used by the Streamlit `app.py`.
-   `uploaded_pdfs/` --- stores PDFs uploaded through the Streamlit app.

These folders contain generated local data and should generally not be
committed to GitHub.

## How It Works

1.  **PDF loading:** `PyPDFLoader` reads the PDF and extracts its pages.
2.  **Text chunking:** `RecursiveCharacterTextSplitter` splits the page
    content into chunks of 500 characters with 50 characters of overlap.
3.  **Embedding generation:** `HuggingFaceEmbeddings` converts each
    chunk into a numerical vector using
    `sentence-transformers/all-MiniLM-L6-v2`.
4.  **Vector storage:** ChromaDB stores the chunks and their embeddings
    locally.
5.  **Retrieval:** The retriever uses MMR to find relevant and diverse
    chunks for a question. The configured settings are `k=3`,
    `fetch_k=10`, and `lambda_mult=0.5`.
6.  **Answer generation:** The retrieved text is inserted into a prompt
    and sent to Gemini. The prompt instructs the model to answer only
    from the supplied context and to say when the answer cannot be
    found.

## Setup and Installation

### 1. Clone the repository

``` bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Replace the URL and folder name with your GitHub repository details.

### 2. Create and activate a virtual environment

**Windows PowerShell:**

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**

``` bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install langchain langchain-community langchain-core langchain-text-splitters langchain-huggingface langchain-chroma langchain-google-genai sentence-transformers chromadb pypdf python-dotenv streamlit
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

``` env
GOOGLE_API_KEY=your_google_gemini_api_key
```

For `app.py`, you can optionally choose the Gemini model with:

``` env
GEMINI_MODEL=gemini-3.5-flash-lite
```

Keep your real API key private. Do not commit `.env` to GitHub.

## Running the Project

This repository contains two ways to interact with the RAG pipeline.
They use separate ChromaDB directories, so their indexes are not
automatically shared.

### Option A: Command-line application

This flow uses `create_database.py` to index the PDF at
`documents/node_js.pdf`, then `main.py` to chat with that indexed
document.

**Step 1 --- Add your PDF**

Place your PDF at:

``` text
documents/node_js.pdf
```

The path is defined in `create_database.py`. If your PDF has a different
name or location, update `PDF_PATH`.

**Step 2 --- Create the vector database**

``` bash
python create_database.py
```

This loads the PDF, creates chunks and embeddings, and stores them in
the `chroma_db/` directory in the `rag_collection` collection.

**Step 3 --- Start the command-line chat**

``` bash
python main.py
```

Enter a question when prompted. Enter `0` to exit.

### Option B: Streamlit web application

The Streamlit app lets you upload a PDF through the browser. It performs
indexing when you click **Process PDF**, then enables the chat input.

Run:

``` bash
streamlit run app.py
```

Open the local URL printed by Streamlit in your browser, upload a PDF,
and click **Process PDF**. Once processing is complete, ask questions in
the chat box.

The app stores uploaded PDF files in `uploaded_pdfs/` and their vectors
in `chroma_db_ui/`. It creates a separate ChromaDB collection for each
distinct PDF file, based on the file's SHA-256 hash. Re-uploading the
same PDF reuses its existing collection.

> **Note:** The Streamlit app has its own indexing and retrieval flow.
> It does not read the `chroma_db/` collection created by
> `create_database.py`; it uses `chroma_db_ui/` instead.

## Configuration

  ------------------------------------------------------------------------------------------
  Setting                 Value                                      Purpose
  ----------------------- ------------------------------------------ -----------------------
  Chunk size              `500`                                      Maximum chunk size
                                                                     configured in the text
                                                                     splitter

  Chunk overlap           `50`                                       Overlap between
                                                                     adjacent chunks

  Embedding model         `sentence-transformers/all-MiniLM-L6-v2`   Converts text chunks
                                                                     into vectors

  Vector database         ChromaDB                                   Persists vectors
                                                                     locally

  Collection (CLI)        `rag_collection`                           Collection used by
                                                                     `main.py`

  Retriever               MMR                                        Retrieves relevant and
                                                                     diverse chunks

  `k`                     `3`                                        Number of chunks
                                                                     returned to the
                                                                     application

  `fetch_k`               `10`                                       Candidate chunks
                                                                     considered by MMR

  `lambda_mult`           `0.5`                                      Relevance/diversity
                                                                     balance for MMR

  Chat model              `gemini-3.5-flash-lite`                    Generates answers from
                                                                     retrieved context
  ------------------------------------------------------------------------------------------

## Security and GitHub Notes

Before pushing the project to GitHub, add a `.gitignore` file. For
example:

``` gitignore
# Secrets
.env

# Python
venv/
.venv/
__pycache__/
*.py[cod]

# Local vector databases and uploaded documents
chroma_db/
chroma_db_ui/
uploaded_pdfs/

# Streamlit
.streamlit/secrets.toml
```

Do not upload API keys, private PDFs, virtual environments, or generated
vector database files. A user cloning the project should create their
own `.env` file and supply their own PDF.

## Current Project Scope

-   The command-line flow indexes the configured PDF file and answers
    questions using retrieved chunks.
-   The Streamlit flow supports PDF uploads, local indexing, chat
    history during the current Streamlit session, and source-page
    labels.
-   Both flows use the same Hugging Face embedding model and Gemini
    model configuration by default, but maintain separate ChromaDB
    storage directories.

## Future Improvements

-   Add support for multiple document formats.
-   Add OCR support for scanned or image-only PDFs.
-   Display retrieved source excerpts alongside answers.
-   Add tests for PDF loading, indexing, retrieval, and response
    generation.
-   Add configurable chunking and retrieval settings.
-   Add conversation persistence and document-management features.
-   Add deployment instructions for a hosted version.

## Author

**Your Name**

Add your GitHub profile or portfolio link here.
