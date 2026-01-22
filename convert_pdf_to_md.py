from vision_parse import VisionParser

def convert_pdf_to_md(input_dir, output_dir, api_key):
    parser = VisionParser(
        model_name="gpt-4o",
        api_key=api_key,
        temperature=0.7,
        top_p=0.4,
        image_mode="url",
        detailed_extraction=False,
        enable_concurrency=True,
    )


    for pdf_file in input_dir.iterdir():
        pdf_path = f"{input_dir}/{pdf_file.name}"
        markdown_pages = parser.convert_pdf(pdf_path)

        with open(f"{output_dir}/{pdf_file.name}.md", "w", encoding="utf-8") as f:
            for i, page_content in enumerate(markdown_pages):
                f.write(page_content)
