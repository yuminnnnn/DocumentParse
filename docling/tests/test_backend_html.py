from io import BytesIO
from pathlib import Path

from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import (
    ConversionResult,
    DoclingDocument,
    InputDocument,
    SectionHeaderItem,
)
from docling.document_converter import DocumentConverter

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA


def test_heading_levels():
    in_path = Path("tests/data/html/wiki_duck.html")
    in_doc = InputDocument(
        path_or_stream=in_path,
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
    )
    backend = HTMLDocumentBackend(
        in_doc=in_doc,
        path_or_stream=in_path,
    )
    doc = backend.convert()

    found_lvl_1 = found_lvl_2 = False
    for item, _ in doc.iterate_items():
        if isinstance(item, SectionHeaderItem):
            if item.text == "Etymology":
                found_lvl_1 = True
                # h2 becomes level 1 because of h1 as title
                assert item.level == 1
            elif item.text == "Feeding":
                found_lvl_2 = True
                # h3 becomes level 2 because of h1 as title
                assert item.level == 2
    assert found_lvl_1 and found_lvl_2


def test_ordered_lists():
    test_set: list[tuple[bytes, str]] = []

    test_set.append(
        (
            b"<html><body><ol><li>1st item</li><li>2nd item</li></ol></body></html>",
            "1. 1st item\n2. 2nd item",
        )
    )
    test_set.append(
        (
            b'<html><body><ol start="1"><li>1st item</li><li>2nd item</li></ol></body></html>',
            "1. 1st item\n2. 2nd item",
        )
    )
    test_set.append(
        (
            b'<html><body><ol start="2"><li>1st item</li><li>2nd item</li></ol></body></html>',
            "2. 1st item\n3. 2nd item",
        )
    )
    test_set.append(
        (
            b'<html><body><ol start="0"><li>1st item</li><li>2nd item</li></ol></body></html>',
            "0. 1st item\n1. 2nd item",
        )
    )
    test_set.append(
        (
            b'<html><body><ol start="-5"><li>1st item</li><li>2nd item</li></ol></body></html>',
            "1. 1st item\n2. 2nd item",
        )
    )
    test_set.append(
        (
            b'<html><body><ol start="foo"><li>1st item</li><li>2nd item</li></ol></body></html>',
            "1. 1st item\n2. 2nd item",
        )
    )

    for idx, pair in enumerate(test_set):
        in_doc = InputDocument(
            path_or_stream=BytesIO(pair[0]),
            format=InputFormat.HTML,
            backend=HTMLDocumentBackend,
            filename="test",
        )
        backend = HTMLDocumentBackend(
            in_doc=in_doc,
            path_or_stream=BytesIO(pair[0]),
        )
        doc: DoclingDocument = backend.convert()
        assert doc
        assert doc.export_to_markdown() == pair[1], f"Error in case {idx}"


def test_unicode_characters():
    raw_html = "<html><body><h1>Hello World!</h1></body></html>".encode()  # noqa: RUF001
    in_doc = InputDocument(
        path_or_stream=BytesIO(raw_html),
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
        filename="test",
    )
    backend = HTMLDocumentBackend(
        in_doc=in_doc,
        path_or_stream=BytesIO(raw_html),
    )
    doc: DoclingDocument = backend.convert()
    assert doc.texts[0].text == "Hello World!"


def get_html_paths():
    # Define the directory you want to search
    directory = Path("./tests/data/html/")

    # List all HTML files in the directory and its subdirectories
    html_files = sorted(directory.rglob("*.html"))
    return html_files


def get_converter():
    converter = DocumentConverter(allowed_formats=[InputFormat.HTML])

    return converter


def test_e2e_html_conversions():
    html_paths = get_html_paths()
    converter = get_converter()

    for html_path in html_paths:
        # print(f"converting {html_path}")

        gt_path = (
            html_path.parent.parent / "groundtruth" / "docling_v2" / html_path.name
        )

        conv_result: ConversionResult = converter.convert(html_path)

        doc: DoclingDocument = conv_result.document

        pred_md: str = doc.export_to_markdown()
        assert verify_export(pred_md, str(gt_path) + ".md", generate=GENERATE), (
            "export to md"
        )

        pred_itxt: str = doc._export_to_indented_text(
            max_text_len=70, explicit_tables=False
        )
        assert verify_export(pred_itxt, str(gt_path) + ".itxt", generate=GENERATE), (
            "export to indented-text"
        )

        assert verify_document(doc, str(gt_path) + ".json", GENERATE)
