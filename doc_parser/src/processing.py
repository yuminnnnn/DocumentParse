import re, traceback
from io import BytesIO
from pathlib import Path
from typing import List

import pypdfium2 as pdfium
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import DocumentStream

from .config import OUTPUT_DIR, BATCH_SIZE

def postprocess_markdown(content: str) -> str:
    content = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', '이미지)', content)
    content = re.sub(r'(^|\n)\s*\* ', r'\1- ', content)

    lines = []
    for line in content.split('\n'):
        s = line.strip()
        if ('×' in s or '+' in s) and s and not (s.startswith('$') and s.endswith('$')):
            lines.append(f"${s}$")
        else:
            lines.append(line)
    return "\n".join(lines).replace("```", "").strip()


def _load_pdf(path: Path):
    try:
        pdf = pdfium.PdfDocument(path)
        print(f"총 페이지 수: {len(pdf)}")
        return pdf
    except Exception as e:
        print(f"PDF 열기 실패: {e}")
        return None


def _process_batch(source_pdf, pages: list[int], batch_num: int, path: Path, converter: DocumentConverter) -> str:
    page_numbers = [i+1 for i in pages]
    print(f"\n 배치 처리: 페이지 {page_numbers[0]}-{page_numbers[-1]}...")

    try:
        temp_pdf = pdfium.PdfDocument.new()
        temp_pdf.import_pages(source_pdf, pages=pages)
        buffer = BytesIO()
        temp_pdf.save(buffer)
        buffer.seek(0)
        temp_pdf.close()

        stream = DocumentStream(name=path.name, stream=buffer)
        result = converter.convert(stream)

        if result.status.name == "SUCCESS":
            return postprocess_markdown(result.document.export_to_markdown())

        err = f"  배치 실패: {result.status.name} - {result.status.value}"
        print(err)
        return f"\n\n---\n\n### {err}\n\n---\n\n"

    except Exception as e:
        err = f" 배치 처리 중 예외 발생: {e}"
        print(err)
        traceback.print_exc()
        return f"\n\n---\n\n### {err}\n\n---\n\n"


def process_pdf_batch(input_pdf_path: Path, converter: DocumentConverter, batch_size: int = BATCH_SIZE) -> List[str]:
    if converter is None:
        print("Converter가 준비되지 않았습니다.")
        return []
    if not input_pdf_path.exists():
        print(f"파일을 찾을 수 없습니다: {input_pdf_path}")
        return []

    print(f"배치 파이프라인 시작: {input_pdf_path.name}")

    source_pdf = _load_pdf(input_pdf_path)
    if not source_pdf:
        return []

    results = []
    for start in range(0, len(source_pdf), batch_size):
        pages = list(range(start, min(start + batch_size, len(source_pdf))))
        results.append(_process_batch(source_pdf, pages, (start//batch_size)+1, input_pdf_path, converter))

    return results
