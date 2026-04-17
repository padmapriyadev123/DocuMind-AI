import pdfplumber
import tiktoken
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_file, pdf_name):
    pages = []

    # Method 1: pdfplumber
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text()
                    if text and len(text.strip()) > 20:
                        pages.append({
                            "text": text.strip(),
                            "source": pdf_name,
                            "page": i + 1
                        })
                except Exception:
                    continue
    except Exception:
        pass

    # Method 2: PyPDF2 fallback
    if not pages:
        try:
            if hasattr(pdf_file, 'seek'):
                pdf_file.seek(0)
            reader = PdfReader(pdf_file)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and len(text.strip()) > 20:
                    pages.append({
                        "text": text.strip(),
                        "source": pdf_name,
                        "page": i + 1
                    })
        except Exception as e:
            raise ValueError(f"Could not read {pdf_name}: {str(e)}")

    if not pages:
        raise ValueError(
            f"'{pdf_name}' has no extractable text. "
            f"It may be a scanned image PDF."
        )
    return pages


def chunk_pages(pages, chunk_size=500, overlap=50):
    enc = tiktoken.get_encoding("cl100k_base")
    chunks = []
    for page in pages:
        tokens = enc.encode(page["text"])
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_text = enc.decode(tokens[start:end])
            if len(chunk_text.strip()) > 10:
                chunks.append({
                    "text": chunk_text,
                    "source": page["source"],
                    "page": page["page"]
                })
            if end == len(tokens):
                break
            start += chunk_size - overlap
    return chunks