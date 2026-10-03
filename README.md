# MiniChatGPT

MiniChatGPT is an offline-first AI assistant built around a local `GPT4All` model, retrieval over uploaded documents, OCR for scanned PDFs and images, and an optional offline image generation panel inside a Streamlit UI.

This repository is not a simple "offline ChatGPT" demo anymore. The main app in `app.py` combines several features:

- Local text chat using `mistral-7b-instruct-v0.2.Q4_K_M.gguf`
- Document question answering for uploaded `PDF`, `DOCX`, and `TXT` files
- OCR fallback for low-text or scanned PDFs
- Image understanding through local captioning when available, with OCR/property fallback
- Offline image generation through a local `sd-turbo` pipeline
- Language-aware replies, including Indian scripts and romanized input such as Hinglish

## Main Entry Point

The real application entry point is:

```bash
streamlit run app.py
```

The older sample-style command `python main.py` does not apply to this project because there is no `main.py` in the repo.

## What Each File Does

- `app.py`: Main Streamlit application and the actual product entry point
- `indexer.py`: Small standalone script that indexes sample PDF, DOCX, and image content into `./chroma_db`
- `query.py`: Simple retrieval test against the persisted Chroma database
- `chatbot.py`: Minimal command-line loop over the persisted Chroma database
- `loader_test.py`: Quick loader and OCR verification script
- `ocr_test.py`: Simple OCR smoke test
- `make_sample_pdf.py`: Generates `sample.pdf`
- `make_sample_docx.py`: Generates `sample.docx`
- `models/sd-turbo/`: Local diffusion model folder used by the offline image studio
- `mistral-7b-instruct-v0.2.Q4_K_M.gguf`: Local LLM file loaded by `GPT4All`

## Actual Architecture

### 1. Local chat path

```text
User message
  -> Streamlit chat UI
  -> prompt builder with language alignment
  -> GPT4All + local Mistral GGUF
  -> response shown in chat history
```

### 2. Document QA path

```text
Uploaded PDF/DOCX/TXT
  -> LangChain loader
  -> optional OCR fallback for weak/scanned PDFs
  -> RecursiveCharacterTextSplitter
  -> HuggingFace embeddings (all-MiniLM-L6-v2)
  -> Chroma vector store
  -> similarity retrieval
  -> answer built from retrieved context
```

### 3. Image understanding path

```text
Uploaded PNG/JPG/WEBP
  -> BLIP caption model if cached locally
  -> otherwise image-property summary + OCR text fallback
  -> image-aware Q&A in chat
```

### 4. Offline image generation path

```text
Prompt
  -> Diffusers pipeline
  -> local sd-turbo model
  -> generated image preview + download
```

## Libraries and Tools Used

### Core app dependencies

- `streamlit`
- `langchain`
- `langchain-community`
- `langchain-core`
- `chromadb`
- `sentence-transformers`
- `gpt4all`
- `transformers`
- `torch`
- `pillow`
- `pytesseract`
- `pdf2image`
- `pypdf`
- `docx2txt`

### Optional or feature-specific dependencies

- `diffusers`
- `accelerate`
- `safetensors`
- `langchain-chroma`
- `langchain-huggingface`
- `unstructured`
- `python-docx`
- `reportlab`

### External system tools

- `Tesseract OCR` for image/PDF OCR
- `Poppler` for PDF-to-image conversion used by `pdf2image`

## Important Setup Notes

### 1. Local LLM path is currently hard-coded

`app.py` loads the model from this exact Windows path:

```text
C:/Users/munde/OneDrive/Desktop/minichatgpt/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

If you move the project, update the `model_path` inside `app.py`.

### 2. Image generation expects a local model

The sidebar image studio first looks for:

```text
models/sd-turbo
```

If that folder is missing, it falls back to the `OFFLINE_IMAGE_MODEL` environment variable or a cached Hugging Face model ID.

### 3. OCR is optional but important for scanned PDFs

The app can load text PDFs directly, but scanned PDFs need:

- `pytesseract`
- `pdf2image`
- installed `tesseract.exe`
- installed Poppler binaries such as `pdftoppm.exe`

## Recommended Installation

Create and activate a virtual environment, then install the packages used by the current codebase.

```bash
pip install streamlit langchain langchain-community langchain-core chromadb sentence-transformers gpt4all transformers torch pillow pytesseract pdf2image pypdf docx2txt diffusers accelerate safetensors langchain-chroma langchain-huggingface unstructured python-docx reportlab
```

Install external OCR tools separately:

- Tesseract OCR
- Poppler

## How To Run

### Streamlit app

```bash
streamlit run app.py
```

### Helper scripts

Build sample files if needed:

```bash
python make_sample_pdf.py
python make_sample_docx.py
```

Index sample data:

```bash
python indexer.py
```

Query the persisted vector store:

```bash
python query.py
```

Run the simple CLI retrieval chatbot:

```bash
python chatbot.py
```

## How The Main App Behaves

- The app keeps multiple chats in Streamlit session state
- Uploading a new file clears old retrieval context so stale answers are not reused
- PDFs may be reprocessed through OCR when extracted text is too sparse
- Retrieved answers are cached for repeated document questions
- The default configuration favors fast response paths
- Very common coding prompts may return instant hard-coded code snippets

## Current Limitations

- The main LLM path is hard-coded to one machine
- `requirements.txt` in the repo does not fully match what `app.py` imports
- OCR quality depends on scan quality and installed system tools
- The app does not have a formal test suite
- The Streamlit app builds the document vector store in memory for each upload instead of reusing `./chroma_db`
- Image captioning quality depends on whether the BLIP model is already cached locally
- The helper scripts and the Streamlit app use slightly different LangChain package variants

## Suggested Improvements

- Move hard-coded model paths into environment variables or a config file
- Split required dependencies from optional ones
- Add a proper `requirements-app.txt` or refresh `requirements.txt`
- Add unit tests for OCR routing, retrieval routing, and language detection
- Persist uploaded-document embeddings if large files are used often
- Add a real settings panel for model path, OCR behavior, and retrieval depth

## Quick Summary

This project is best described as an offline multimodal assistant, not just a basic RAG chatbot. The main deliverable is the Streamlit app in `app.py`, while `indexer.py`, `query.py`, and `chatbot.py` are small companion scripts for separate Chroma-based experiments.
