from langchain_text_splitters import MarkdownHeaderTextSplitter
from pathlib import Path
import re

def normalize_headings(md):
    return re.sub(
        r"(?m)^\*\*(.+?)\*\*\s*$",
        r"# \1",
        md
    )

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
    return sections

if __name__ == "__main__":
    path = "markdown_docs"
    parse_markdown_sections(path)