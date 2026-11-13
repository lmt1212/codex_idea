"""Layout-aware PDF/OCR and spreadsheet adapters with explicit quality flags."""
import hashlib
import re
from pathlib import Path
from .schema import uid
from .ingestion import finish_document


def page_interval(value, total):
    if not value:
        return 0, total
    parts = value.split(':')
    if len(parts) != 2:
        raise ValueError('pages must be first:last, inclusive and 1-based')
    first, last = map(int, parts)
    if not 1 <= first <= last <= total:
        raise ValueError('page range outside document')
    return first-1, last


def quality_flags(value):
    flags = []
    if '\ufffd' in value or '\x00' in value:
        flags.append('garbled_text')
    if re.search(r'(?:[=<>±∑√]|\b(?:sin|cos|log)\s*\()', value):
        flags.append('formula_requires_review')
    if not value.strip():
        flags.append('empty_text')
    return flags


def parse_pdf(path, pages=None, ocr='auto', provisional=False, group=None):
    import pymupdf
    path = Path(path)
    did = 'doc:' + hashlib.sha256(path.read_bytes()).hexdigest()[:24]
    blocks, warnings, engine = [], [], None
    with pymupdf.open(path) as pdf:
        first, last = page_interval(pages, len(pdf))
        toc = pdf.get_toc()
        for page_no in range(first, last):
            page = pdf[page_no]
            section = [entry[1] for entry in toc if entry[2] <= page_no+1][-1:]
            raw = [b for b in page.get_text('blocks', sort=True) if b[6] == 0 and b[4].strip()]
            tables = []
            try:
                tables = [dict(bbox=list(t.bbox), cells=t.extract()) for t in page.find_tables().tables]
            except Exception as ex:
                warnings.append(dict(page=page_no+1, issue='table_detection_failed', error_type=type(ex).__name__))
            used_ocr = ocr == 'always' or (ocr == 'auto' and not raw)
            if used_ocr:
                if engine is None:
                    from rapidocr import RapidOCR
                    engine = RapidOCR()
                import numpy as np
                pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                result = engine(image[:, :, :3])
                raw = []
                if result.txts is not None:
                    for box, value, confidence in zip(result.boxes, result.txts, result.scores):
                        xs, ys = [float(v[0])/2 for v in box], [float(v[1])/2 for v in box]
                        raw.append((min(xs), min(ys), max(xs), max(ys), value, 0, 0, float(confidence)))
            if not raw:
                raise ValueError(f'page {page_no+1} contains no extracted text; enable OCR')
            for table in tables:
                value = '\n'.join(' | '.join('' if cell is None else str(cell) for cell in row) for row in table['cells'])
                blocks.append(dict(block_id=uid('block', [did, page_no, 'table', table['bbox']]), text=value,
                                   kind='table', table=table['cells'], section=section,
                                   position=dict(pages=[page_no+1], bbox=table['bbox']), needs_review=True,
                                   quality_flags=['table_structure_requires_review']))
            for b in raw:
                rect = pymupdf.Rect(b[:4])
                if not used_ocr and any(rect.intersects(pymupdf.Rect(t['bbox'])) and (rect & pymupdf.Rect(t['bbox'])).get_area() / max(rect.get_area(), 1) > .5 for t in tables):
                    continue
                value = b[4].strip()
                flags = quality_flags(value)
                confidence = b[7] if used_ocr else None
                if used_ocr:
                    flags.append('ocr_requires_review')
                margin = rect.y1 < 35 or rect.y0 > page.rect.height-25
                blocks.append(dict(block_id=uid('block', [did, page_no, list(b[:4]), value]), text=value,
                                   kind='text', section=section, position=dict(pages=[page_no+1], bbox=list(b[:4])),
                                   needs_review=not provisional or bool(flags), quality_flags=flags,
                                   excluded=margin, extraction_method='rapidocr' if used_ocr else 'native',
                                   ocr_confidence=confidence, source_status='automated_unreviewed'))
        document = finish_document(did, path.stem, str(path.resolve()), blocks, group)
        document.update(parser='pymupdf_layout_rapidocr', selected_pages=[first+1,last], total_pages=len(pdf),
                        parsing_warnings=warnings, cross_page_tables_merged=False)
        return document


def parse_spreadsheet(path, group=None):
    import openpyxl
    path = Path(path)
    did = 'doc:' + hashlib.sha256(path.read_bytes()).hexdigest()[:24]
    blocks = []
    workbook = openpyxl.load_workbook(path, read_only=False, data_only=False)
    try:
        for sheet in workbook:
            values = [[cell.value for cell in row] for row in sheet]
            value = '\n'.join(' | '.join('' if cell is None else str(cell) for cell in row) for row in values)
            if not value.strip():
                continue
            blocks.append(dict(block_id=uid('block', [did, sheet.title]), kind='table', text=value,
                               table=values, merged_cells=[str(r) for r in sheet.merged_cells.ranges],
                               section=[sheet.title], position=dict(sheet=sheet.title,pages=[]), needs_review=True))
    finally:
        workbook.close()
    return finish_document(did, path.stem, str(path.resolve()), blocks, group)
