from langchain_text_splitters import MarkdownHeaderTextSplitter
from pathlib import Path
import re

def normalize_headings(md):
    return re.sub(
        r"(?m)^\*\*(.+?)\*\*\s*$",
        r"# \1",
        md
    )

def normalize_metadata(docs):
    for doc in docs:
        if not doc.metadata:
            doc.metadata = {"source": "unknown"}
    return docs

def parse_markdown_sections(file_directory):

    sections = []

    markdown_dir_path = Path(f"{file_directory}")

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)

    for file in markdown_dir_path.iterdir():
        with open(f"{file}", "r") as md_file:
            data = md_file.read()
            normalized_data = normalize_headings(data)
            md_header_splits = markdown_splitter.split_text(normalized_data)
        sections.append(md_header_splits)

    for doc_list in sections:
        normalize_metadata(doc_list)

    return sections

if __name__ == "__main__":
    path = "markdown_docs"
    doc_sections = parse_markdown_sections(path)
    print(doc_sections[0][0].page_content)
    print(doc_sections[0][0].metadata)
