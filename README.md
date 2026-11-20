# Ghost PDF Chat

A private, local chat interface for PDFs. Upload a document, ask a question, and get answers grounded in the document—without sending the PDF or your questions to a cloud service.

Ghost PDF Chat runs [Ollama](https://ollama.com) and its search index on your own computer. The index is held in RAM for the active browser session and can be cleared with **Forget this document**.

## Features

- Local PDF question answering with Ollama
- Answers constrained to retrieved document context
- Automatic OCR fallback for scanned PDFs via Tesseract
- In-memory FAISS index—no document database or persisted vector store
- A 50 MB upload guard, configurable for your available RAM
- macOS-friendly Streamlit interface

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com), installed and running
- The `llama3.2:1b` model (or another Ollama model you choose)
- [Homebrew](https://brew.sh/) and Tesseract if you want scanned-PDF OCR

## Quick start

```bash
git clone https://github.com/armaanchoraria/Ghost-PDF-Chat.git
cd Ghost-PDF-Chat
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:1b
streamlit run app.py
```

Open the local address Streamlit prints, normally `http://localhost:8501`.

### Enable OCR for scanned PDFs

```bash
brew install tesseract
```

OCR is automatic: when a PDF has no selectable text, Ghost renders its pages in memory and passes them to the local Tesseract process. No PDF file is written by the application.

## Configuration

Configuration is optional and uses environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `GHOST_PDF_MODEL` | `llama3.2:1b` | Ollama chat model |
| `GHOST_PDF_EMBED_MODEL` | chat model | Ollama embedding model |
| `GHOST_PDF_MAX_UPLOAD_MB` | `50` | Maximum PDF size retained in RAM |

For improved retrieval, install a dedicated embedding model and launch with it:

```bash
ollama pull nomic-embed-text
GHOST_PDF_EMBED_MODEL=nomic-embed-text streamlit run app.py
```

## Privacy and limits

- This app does not upload PDFs or questions itself. Ollama runs locally by default.
- The index lives in the Streamlit session's memory. Use **Forget this document** to clear it immediately; closing the browser session also clears it.
- OCR quality depends on the source scan. Handwriting, skewed pages, and low-resolution images may need cleanup before upload.
- `llama3.2:1b` is deliberately small for modest hardware. Larger local models may produce better answers.

## Troubleshooting

**“Could not read or index this PDF”** — verify Ollama is running and pull the configured model with `ollama pull <model>`.

**OCR asks for Tesseract** — run `brew install tesseract`, then restart Streamlit.

**The answer is not in the PDF** — Ghost is designed to say so rather than make up an answer. Try a more specific question or use a stronger local model.

## License

[MIT](LICENSE)
