# step02_pdf_chunking.py
import os
import re
from dataclasses import dataclass, asdict
from typing import Iterator, List
from pypdf import PdfReader
import tiktoken

from config import Settings

ENCODER = tiktoken.get_encoding("cl100k_base")  # kompatibel zu OpenAI-Embeddings

@dataclass
class Chunk:
    document_id: str             # z. B. Dateiname ohne Pfad
    chunk_index: int             # fortlaufend pro Dokument
    text: str
    source_path: str             # absoluter Pfad
    page_start: int              # 1-basiert
    page_end: int                # 1-basiert

# ---------- Hilfsfunktionen ----------

def find_pdfs(root_dir: str) -> List[str]:
    pdfs = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.abspath(os.path.join(root, f)))
    return sorted(pdfs)

def extract_pages(path: str) -> List[str]:
    """Liest PDF-Seiten als Text (eine Liste: ein Eintrag pro Seite)."""
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        txt = p.extract_text() or ""
        txt = normalize_text(txt)
        if txt.strip():
            pages.append(txt)
        else:
            pages.append("")  # leere Seite trotzdem markieren
    return pages

def normalize_text(text: str) -> str:
    # weiche Zeilenumbrüche/Hypens glätten, Whitespace komprimieren
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)     # Silbentrennung Zeilenende
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text_by_tokens(text: str, max_tokens: int, overlap: int) -> Iterator[str]:
    """Chunking über Token-Fenster mit Überlappung; gibt dekodierten Text je Chunk zurück."""
    tokens = ENCODER.encode(text)
    step = max(1, max_tokens - overlap)
    for start in range(0, len(tokens), step):
        end = min(start + max_tokens, len(tokens))
        yield ENCODER.decode(tokens[start:end])
        if end == len(tokens):
            break

def chunk_pages(
    pages: List[str],
    max_tokens: int,
    overlap: int,
    fuse_pages: bool = False
) -> Iterator[tuple[str, int, int]]:
    """
    Erzeugt Text-Chunks und liefert (chunk_text, page_start, page_end).
    - fuse_pages=False: pro Seite chunken (einfach, robuste Seitenangaben)
    - fuse_pages=True: Seiten zusammenfügen und dann chunken (besserer Kontext, schwierigere Seitenspannen)
    """
    if not fuse_pages:
        for i, page in enumerate(pages, start=1):
            if not page.strip():
                continue
            for chunk in chunk_text_by_tokens(page, max_tokens, overlap):
                yield chunk, i, i
    else:
        # Seiten zusammenfügen; wir merken uns die Grenzen grob über Marker
        sep = "\n\n[[PAGE_BREAK]]\n\n"
        joined = sep.join(pages)
        # Index -> Seite grob rekonstruieren
        # (vereinfachter Ansatz, reicht für Doku/Navigation)
        positions = []
        pos = 0
        for i, page in enumerate(pages, start=1):
            positions.append((pos, i))
            pos += len(page) + len(sep)

        for chunk in chunk_text_by_tokens(joined, max_tokens, overlap):
            # grobe Schätzung der Seitenspanne über Stringpositionen
            # (optional; kann später verfeinert werden)
            # Wir verwenden start=0, weil chunk_text_by_tokens auf Tokenbasis schneidet.
            # Fürs Prototyping: wir setzen die Spanne auf (1, 1) wenn nicht eindeutig.
            page_start = positions[0][1] if positions else 1
            page_end = positions[-1][1] if positions else 1
            yield chunk, page_start, page_end

# ---------- Main-Pipeline für Schritt 2 ----------

def build_chunks_for_directory(s: Settings) -> List[Chunk]:
    pdf_paths = find_pdfs(s.pdf_dir)
    if not pdf_paths:
        raise SystemExit(f"Keine PDFs gefunden unter: {s.pdf_dir}")

    all_chunks: List[Chunk] = []
    for path in pdf_paths:
        pages = extract_pages(path)
        doc_id = os.path.splitext(os.path.basename(path))[0]
        idx = 0
        for chunk_text, p_start, p_end in chunk_pages(
            pages,
            max_tokens=s.chunk_tokens,
            overlap=s.chunk_overlap,
            fuse_pages=False,     # bewusst simpel/robust (Seitenangaben exakt)
        ):
            all_chunks.append(
                Chunk(
                    document_id=doc_id,
                    chunk_index=idx,
                    text=chunk_text,
                    source_path=path,
                    page_start=p_start,
                    page_end=p_end,
                )
            )
            idx += 1
        print(f"{doc_id}: {idx} Chunks aus {len(pages)} Seiten")
    print(f"Gesamt: {len(all_chunks)} Chunks aus {len(pdf_paths)} Dateien")
    return all_chunks

def main():
    s = Settings()
    chunks = build_chunks_for_directory(s)

    # kleine Vorschau ausgeben
    preview = min(3, len(chunks))
    for c in chunks[:preview]:
        print("-" * 80)
        print(f"{c.document_id}  [#{c.chunk_index}]  Seiten {c.page_start}-{c.page_end}")
        print(c.text[:400].strip().replace("\n", " ") + ("..." if len(c.text) > 400 else ""))

    # Hier noch kein Persistieren – das machen wir in Schritt 3 (Embeddings) / Schritt 4 (Qdrant Upsert)

if __name__ == "__main__":
    main()
