from mrkdwn_analysis import MarkdownAnalyzer
from pathlib import Path

def parse_markdown_sections(file_directory):

    markdown_dir_path = Path(f"{file_directory}")

    for file in markdown_dir_path.iterdir():
        analyzer = MarkdownAnalyzer(f"{file}")
        headers = analyzer.identify_headers()
        print("Headers:", headers)
        paragraphs = analyzer.identify_paragraphs()
        print("Paragraphs:", paragraphs)

if __name__ == "__main__":
    path = "markdown_docs"
    parse_markdown_sections(path)