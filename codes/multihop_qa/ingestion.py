"""Document parsing and exact block-offset chunking; headings remain metadata."""
import hashlib
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET
from .schema import uid, text

SUPPORTED = {".txt", ".md", ".html", ".htm", ".docx", ".pdf", ".xlsx"}


class HTMLText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip += 1
        if not self.skip and tag in {"p", "div", "br", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self.skip:
            self.skip -= 1
        if not self.skip and tag in {"p", "div", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def document_from_text(content, title="", source="memory", source_group_id=None):
    text(content, "document text")
    did = uid("doc", content)
    blocks, section = [], []
    separated = re.sub(r"(?m)^(#{1,6}[ \t]+[^\n]+)$", r"\n\n\1\n\n", content)
    for number, paragraph in enumerate(re.split(r"\n\s*\n", separated)):
        for part in re.split(r"(?m)(?=^#{1,6}\s)", paragraph):
            value = part.strip()
            if not value:
                continue
            heading = re.fullmatch(r"(#{1,6})\s+([^\n]+)", value)
            if heading:
                depth = len(heading[1])
                section = section[:depth - 1] + [heading[2]]
            blocks.append(dict(block_id=uid("block", [did, len(blocks)]), text=value,
                               kind="heading" if heading else "text", section=list(section),
                               position=dict(paragraph=number, pages=[]), needs_review=False))
    return finish_document(did, title, source, blocks, source_group_id)


def finish_document(did, title, source, blocks, group=None):
    if not blocks or not any(b["text"].strip() for b in blocks):
        raise ValueError("document contains no text; OCR may be required")
    normalized = "".join(re.sub(r"\s+", "", b["text"]) for b in blocks if b["kind"] != "heading")
    return dict(schema_version="1.0", doc_id=did, title=title, source=source,
                source_group_id=group or uid("source", normalized), blocks=blocks)


def parse_file(path, source_group_id=None, pdf_backend='text', pages=None, ocr='auto', provisional=False):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError("unsupported input format: " + suffix)
    if suffix == '.xlsx':
        from .document_parsers import parse_spreadsheet
        return parse_spreadsheet(path, source_group_id)
    if suffix == '.pdf' and pdf_backend == 'layout':
        from .document_parsers import parse_pdf
        return parse_pdf(path, pages, ocr, provisional, source_group_id)
    if suffix in {".txt", ".md", ".html", ".htm"}:
        content = path.read_text(encoding="utf-8-sig")
        if suffix in {".html", ".htm"}:
            parser = HTMLText()
            parser.feed(content)
            content = "".join(parser.parts)
        return document_from_text(content, path.stem, str(path.resolve()), source_group_id)
    did = "doc:" + hashlib.sha256(path.read_bytes()).hexdigest()[:24]
    blocks = []
    if suffix == ".pdf":
        import pymupdf
        with pymupdf.open(path) as pdf:
            for page_number, page in enumerate(pdf, 1):
                page_blocks = page.get_text("blocks", sort=True)
                if not any(b[4].strip() for b in page_blocks if b[6] == 0):
                    raise ValueError(f"page {page_number} has no text layer; OCR required")
                for raw in page_blocks:
                    if raw[6] != 0 or not raw[4].strip():
                        continue
                    blocks.append(dict(block_id=uid("block", [did, len(blocks)]), text=raw[4].strip(),
                                       kind="text", section=[], needs_review=True,
                                       position=dict(pages=[page_number], bbox=list(raw[:4]))))
        # PDF text order and tables are not silently treated as verified OCR/layout.
    elif suffix == ".docx":
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        with zipfile.ZipFile(path) as archive:
            root = ET.fromstring(archive.read("word/document.xml"))
        section = []
        body = root.find("w:body", ns)
        for item in body:
            if item.tag.endswith("}p"):
                value = "".join(n.text or "" for n in item.findall(".//w:t", ns))
                if not value.strip():
                    continue
                style = item.find("w:pPr/w:pStyle", ns)
                name = style.get("{" + ns["w"] + "}val", "") if style is not None else ""
                is_heading = name.lower().startswith("heading")
                if is_heading:
                    section = [value]
                kind = "heading" if is_heading else "text"
            elif item.tag.endswith("}tbl"):
                rows = [["".join(t.text or "" for t in cell.findall(".//w:t", ns))
                         for cell in row.findall("w:tc", ns)] for row in item.findall("w:tr", ns)]
                value = "\n".join(" | ".join(row) for row in rows)
                kind = "table"
            else:
                continue
            blocks.append(dict(block_id=uid("block", [did, len(blocks)]), text=value, kind=kind,
                               section=list(section), position=dict(paragraph=len(blocks), pages=[]),
                               needs_review=kind == "table"))
    return finish_document(did, path.stem, str(path.resolve()), blocks, source_group_id)


def chunk_documents(documents, max_chars=1200):
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    chunks, seen = [], set()
    for doc in documents:
        if doc["doc_id"] in seen:
            continue
        seen.add(doc["doc_id"])
        for block in doc["blocks"]:
            if block["kind"] == "heading" or block.get("excluded"):
                continue
            content = block["text"]
            start = 0
            while start < len(content):
                end = min(start + max_chars, len(content))
                if block["kind"] == "table":
                    end = len(content)
                elif end < len(content):
                    boundaries = [m.end() + start for m in re.finditer(r"[。！？；\n]", content[start:end])]
                    if boundaries:
                        end = boundaries[-1]
                value = content[start:end]
                cid = uid("chunk", [doc["doc_id"], block["block_id"], start, end])
                if value.strip():
                    chunks.append(dict(schema_version="1.0", chunk_id=cid, doc_id=doc["doc_id"],
                                       source_group_id=doc["source_group_id"], title=doc["title"],
                                       section=block.get("section", []), text=value,
                                       source_position=block["position"],
                                       spans=[dict(block_id=block["block_id"], start=start, end=end)],
                                       needs_review=bool(block.get("needs_review")), kind=block["kind"]))
                    chunks[-1]['quality_flags'] = block.get('quality_flags', [])
                    chunks[-1]['extraction_method'] = block.get('extraction_method', 'text')
                start = end
    return chunks


def token_rechunk(documents, tokenizer, max_tokens):
    """No silent truncation: reduce character budget until every prose span fits."""
    if max_tokens < 1:
        raise ValueError('positive token budget required')
    chunks = chunk_documents(documents)
    output = []
    for chunk in chunks:
        if chunk['kind'] == 'table':
            chunk['token_count'] = len(tokenizer.encode(chunk['text'], add_special_tokens=False))
            chunk['oversized'] = chunk['token_count'] > max_tokens
            output.append(chunk)
            continue
        content, offset = chunk['text'], 0
        while offset < len(content):
            end = min(len(content), offset+max_tokens)
            while end > offset and len(tokenizer.encode(content[offset:end], add_special_tokens=False)) > max_tokens:
                end -= 1
            if end == offset:
                raise ValueError('single character exceeds token budget')
            part = dict(chunk, text=content[offset:end])
            original = chunk['spans'][0]
            part['spans'] = [dict(original, start=original['start']+offset, end=original['start']+end)]
            part['chunk_id'] = uid('chunk', [chunk['doc_id'], original['block_id'], original['start']+offset, original['start']+end])
            part['token_count'] = len(tokenizer.encode(part['text'], add_special_tokens=False))
            output.append(part)
            offset = end
    return output
