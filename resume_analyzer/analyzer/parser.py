"""Extract plain text from PDF, DOCX and TXT resumes."""
import io
import re
from pypdf import PdfReader
from docx import Document


def _clean(text: str) -> str:
    text = text.replace("\u2022", "- ").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    elif name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:  # resumes often use tables for layout
            for row in table.rows:
                parts.extend(cell.text for cell in row.cells)
        text = "\n".join(parts)
    elif name.endswith(".txt"):
        text = data.decode("utf-8", errors="ignore")
    else:
        raise ValueError("Unsupported file type. Use PDF, DOCX or TXT.")
    text = _clean(text)
    if len(text) < 50:
        raise ValueError(
            "Almost no text found. The file may be a scanned image; "
            "ATS systems can't read those either."
        )
    return text
