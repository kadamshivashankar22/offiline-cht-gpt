# Project Documentation

## 1. Project Title

**MiniChatGPT: Offline Intelligent AI Assistant with Local LLM, Document Retrieval, OCR, and Image Tools**

## 2. Project Overview

This project implements an offline-first AI assistant that runs locally on a machine using a GGUF large language model through `GPT4All`. The application is delivered through a Streamlit interface and supports:

- normal text chat
- document-based question answering from uploaded files
- OCR support for scanned PDFs and images
- image description and image-based questioning
- offline image generation using a local diffusion model

The central file is `app.py`. The other Python scripts are support utilities for indexing, retrieval experiments, and sample-file testing.

## 3. Objective

The objective of this project is to provide a local AI assistant that can continue working without cloud APIs for common assistant tasks such as:

- answering user questions
- reading uploaded documents
- extracting text from scans
- describing uploaded images
- generating images offline

This makes the system suitable for demos, hackathons, low-connectivity environments, and privacy-sensitive local experimentation.

## 4. Problem Statement

Most AI assistants depend on internet-hosted APIs. That creates several practical problems:

- internet access may be unavailable
- private files may not be safe to upload externally
- latency and cloud cost may be undesirable during demos or internal use

This project addresses that gap by keeping inference, retrieval, and most media processing on the local machine.

## 5. Scope of the Current Repository

Based on the code in this repository, the implemented scope is broader than a plain RAG chatbot.

### Included in the current implementation

- local LLM chat through `GPT4All`
- file upload for `PDF`, `DOCX`, `TXT`, `PNG`, `JPG`, `JPEG`, and `WEBP`
- document chunking and vector search using Chroma
- OCR for scanned or weak-text PDFs
- fallback OCR for uploaded images
- local image captioning when a caption model is already cached
- offline image generation through a local `sd-turbo` model folder
- multilingual response alignment for several Indian scripts and romanized input

### Not fully implemented as production features

- model settings persisted in a config file
- automated dependency bootstrap
- formal test coverage
- deployment packaging
- persistent indexing for user-uploaded documents in the Streamlit app

## 6. Technology Stack

### Programming language

- Python

### UI layer

- Streamlit

### LLM and generation

- GPT4All
- local Mistral GGUF model
- Diffusers
- Torch

### Retrieval and embeddings

- LangChain
- Chroma / ChromaDB
- HuggingFace sentence embeddings
- `sentence-transformers/all-MiniLM-L6-v2`

### Document and OCR processing

- PyPDFLoader
- Docx2txtLoader
- TextLoader
- pytesseract
- pdf2image
- Pillow

### Image understanding

- Transformers pipeline
- BLIP image captioning model when locally cached

## 7. System Architecture

### High-level architecture

```text
User
  -> Streamlit interface
  -> request router
     -> local chat path
     -> document QA path
     -> image QA path
     -> image generation path
  -> response/output
```

### Document QA architecture

```text
Uploaded document
  -> file loader
  -> optional OCR for scanned PDFs
  -> text splitting
  -> embedding generation
  -> Chroma vector store
  -> similarity retrieval
  -> answer synthesis from retrieved context
```

### Image understanding architecture

```text
Uploaded image
  -> local caption model if available
  -> otherwise image-property + OCR fallback
  -> question answering over generated image description
```

### Image generation architecture

```text
Prompt
  -> local sd-turbo diffusion pipeline
  -> generated image preview
  -> PNG download
```

## 8. Main Modules and Their Roles

### 8.1 `app.py`

This is the main product file and contains:

- Streamlit page layout and chat interface
- local LLM loading through `GPT4All`
- embedding model loading
- OCR helper functions
- image captioning fallback logic
- document retrieval and summarization routines
- session-state chat history management
- sidebar image-generation tool

### 8.2 Local chat module

The function `ask_model()` builds a prompt using:

- a system instruction
- a language-alignment rule
- limited recent chat history
- the latest user message

It then sends the prompt to the locally loaded model through cached inference wrappers.

### 8.3 Retrieval module

The document retrieval flow in `app.py` does the following:

1. Load uploaded text content from PDF, DOCX, or TXT.
2. Run OCR on PDFs when direct extraction appears weak.
3. Split the text into chunks with `RecursiveCharacterTextSplitter`.
4. Create embeddings using `all-MiniLM-L6-v2`.
5. Build a Chroma vector store.
6. Retrieve the most relevant chunk for document-based questions.
7. Generate an answer strictly from retrieved context.

An additional fast overview path is used for broad file-analysis questions.

### 8.4 OCR module

The OCR pipeline is implemented through `extract_text_with_ocr()` and helper functions.

It includes:

- automatic discovery of `tesseract.exe`
- automatic search for Poppler binaries
- PDF page conversion through `pdf2image`
- OCR text extraction through `pytesseract`

This is especially useful for scanned PDFs where direct text extraction fails.

### 8.5 Image understanding module

The image-analysis path first tries a local captioning model through Transformers. If that model is unavailable, the app falls back to:

- image size and orientation
- tone estimation
- OCR text detected inside the image

This generated description is then used to answer image-related questions in the chat.

### 8.6 Offline image generation module

The sidebar includes an image-generation tool that:

- loads a local `sd-turbo` model
- runs fully offline once the model is present locally
- uses CUDA when available, otherwise CPU fallback
- lets the user preview and download the generated image

### 8.7 Helper scripts

The repository also contains small standalone scripts:

- `indexer.py` indexes sample files into a persisted Chroma database
- `query.py` performs a direct similarity search on that database
- `chatbot.py` offers a terminal-based retrieval loop
- `loader_test.py` validates PDF, DOCX, and OCR extraction
- `ocr_test.py` performs a simple OCR smoke test

These scripts are useful for experimentation, but they are not the main user-facing application.

## 9. Detailed Working Flow

### Scenario A: normal chat

1. User enters a text message.
2. The app detects whether the query is general chat or code-oriented.
3. The local Mistral model answers through `GPT4All`.
4. The response is shown in the Streamlit chat window.

### Scenario B: document question answering

1. User uploads a PDF, DOCX, or TXT file.
2. The file is loaded and split into chunks.
3. If the PDF looks scanned, OCR is attempted.
4. The chunk embeddings are stored in a Chroma vector store.
5. When the user asks a file-related question, the app retrieves relevant chunks.
6. The answer is produced only from uploaded document context.

### Scenario C: image understanding

1. User uploads an image.
2. The app tries to caption it locally.
3. If no caption model is available, fallback summary logic is used.
4. The user can then ask questions about the uploaded image.

### Scenario D: image generation

1. User opens the sidebar image studio.
2. User enters a text prompt and output size.
3. The local diffusion model generates an image.
4. The result is previewed and can be downloaded as PNG.

## 10. File and Folder Description

```text
minichatgpt/
|- app.py                              Main Streamlit application
|- indexer.py                          Sample document indexing script
|- query.py                            Similarity-search test script
|- chatbot.py                          CLI retrieval chatbot
|- loader_test.py                      Loader and OCR validation script
|- ocr_test.py                         OCR smoke test
|- make_sample_pdf.py                  Sample PDF generator
|- make_sample_docx.py                 Sample DOCX generator
|- sample.pdf                          Demo PDF
|- sample.docx                         Demo DOCX
|- sample.png                          Demo image
|- chroma_db/                          Persisted Chroma store used by helper scripts
|- models/sd-turbo/                    Local diffusion model folder
|- mistral-7b-instruct-v0.2...gguf     Local LLM model file
```

## 11. Setup Requirements

### Python dependencies used by the current code

```bash
pip install streamlit langchain langchain-community langchain-core chromadb sentence-transformers gpt4all transformers torch pillow pytesseract pdf2image pypdf docx2txt diffusers accelerate safetensors langchain-chroma langchain-huggingface unstructured python-docx reportlab
```

### External tools

- Install Tesseract OCR
- Install Poppler

### Local model files

- `mistral-7b-instruct-v0.2.Q4_K_M.gguf`
- `models/sd-turbo/` or another locally cached diffusion model

## 12. How to Run the Project

### Main app

```bash
streamlit run app.py
```

### Helper scripts

```bash
python indexer.py
python query.py
python chatbot.py
python loader_test.py
python ocr_test.py
```

### Sample file generation

```bash
python make_sample_pdf.py
python make_sample_docx.py
```

## 13. Implementation Notes From Code Review

The documentation above is based on the actual repository contents, and a few practical points are important:

- `app.py` is the real entry point; there is no `main.py`
- the LLM path is hard-coded to one Windows location
- the Streamlit app creates an in-memory Chroma store for each uploaded document session
- the helper scripts use a persisted `chroma_db` directory
- OCR support depends on external system binaries, not only Python packages
- the checked-in `requirements.txt` does not fully cover the imports used in `app.py`

## 14. Advantages

- runs locally without cloud APIs
- supports both structured document QA and normal chat
- can recover text from scanned PDFs
- can describe uploaded images
- includes an offline image-generation feature
- privacy-friendly because user files stay on the local machine

## 15. Limitations

- hard-coded model path reduces portability
- dependency setup is not yet streamlined
- OCR accuracy depends on scan quality
- image captioning is limited if the local caption model is missing
- helper scripts and main app are not yet unified under one package structure
- there is no automated testing pipeline in the repository

## 16. Future Enhancements

- move model paths and runtime flags into configuration
- refresh `requirements.txt` to match the actual codebase
- add unit and integration tests
- persist uploaded-document indexes for reuse
- add model selection and settings UI
- improve packaging for one-click setup
- add exportable conversation history

## 17. Conclusion

This project is a local multimodal AI assistant rather than a narrow OCR or RAG demo. It combines offline chat, retrieval, OCR, image understanding, and image generation inside one Streamlit application. The repository already demonstrates a strong prototype for offline AI assistance, but it would benefit from dependency cleanup, portability improvements, and testing before production use.

## 18. Short Hackathon Summary

MiniChatGPT is an offline AI assistant built with a local Mistral model, Chroma-based retrieval, OCR for scanned PDFs, image understanding, and offline image generation. The main app runs in Streamlit and allows users to chat, ask questions from uploaded files, inspect images, and generate images without relying on external AI APIs.
