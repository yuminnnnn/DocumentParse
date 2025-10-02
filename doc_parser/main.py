import sys
import os
from pathlib import Path
import argparse

sys.path.insert(0, os.path.abspath(str(Path(__file__).resolve().parent / "src")))

from src.config import INPUT_DIR, OUTPUT_DIR
from src.docling_setup import create_ocr_options, create_vlm_options, setup_document_converter
from src.processing import process_pdf_batch

def main():
    # 명령줄 인자 파싱 설정
    parser = argparse.ArgumentParser(description="Doc Parser 파이프라인을 실행하여 PDF 문서를 마크다운으로 변환합니다.")
    parser.add_argument(
        "input_file",
        nargs="?", # 인자가 없으면 None
        type=str,
        help="처리할 PDF 파일의 경로 (예: data/my_document.pdf)"
    )
    args = parser.parse_args()

    # 입력 파일 경로 결정
    if args.input_file:
        input_pdf_path = Path(args.input_file)
    else:
        # 인자가 없으면 기본 입력 디렉토리(doc_parser/data) 내의 첫 번째 PDF 파일을 기본으로 사용
        pdf_files = list(INPUT_DIR.glob("*.pdf"))
        if not pdf_files:
            print(f"❌ 기본 입력 디렉토리 '{INPUT_DIR}'에 PDF 파일이 없습니다. 'input_file' 인자를 지정하거나 'doc_parser/data'에 PDF 파일을 넣어주세요.")
            return
        input_pdf_path = pdf_files[0] # 첫 번째 PDF 파일을 기본으로 사용

    if not input_pdf_path.exists():
        print(f"❌ 지정된 입력 파일을 찾을 수 없습니다: {input_pdf_path}")
        return
    print("🚀 Doc Parser 파이프라인 시작")

    # 출력 디렉토리 생성
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. OCR 및 VLM 옵션 설정
    ocr_options = create_ocr_options()
    vlm_options = create_vlm_options()

    if vlm_options is None:
        print("❌ VLM 옵션 설정에 실패하여 프로그램을 종료합니다.")
        return

    # 2. DocumentConverter 설정
    converter = setup_document_converter(ocr_options, vlm_options)

    if converter is None:
        print("❌ DocumentConverter 설정에 실패하여 프로그램을 종료합니다.")
        return

    # 3. PDF 배치 처리 및 결과 병합
    results = process_pdf_batch(input_pdf_path, converter)

    # 4. 최종 결과 저장
    if results:
        final_markdown = "\n\n".join(results)
        file_name_stem = input_pdf_path.stem
        output_file = OUTPUT_DIR / f"{file_name_stem}_output.md"
        output_file.write_text(final_markdown, encoding='utf-8')
        

        print(f"\n\n✅✅✅ 최종 결과가 성공적으로 저장되었습니다: {output_file}")
    else:
        print("❌ 처리된 내용이 없어 최종 파일을 생성하지 않았습니다.")

if __name__ == "__main__":
    main()
