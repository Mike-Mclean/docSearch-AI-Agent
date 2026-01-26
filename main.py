from dotenv import load_dotenv
import chromadb
import uuid
from pathlib import Path
from openai import OpenAI
from split_mds import *
from convert_pdf_to_md import *
import os

load_dotenv()

openAI_client = OpenAI()

chroma_client = chromadb.Client()
directory_path = Path("test_docs")

openAI_api_key = os.getenv("OPENAI_API_KEY")

pdfs_dir = Path("harder_test_docs")

md_dir = Path("markdown_docs")
md_dir.mkdir(exist_ok=True)


def create_collection():
    try:
        collection = chroma_client.get_collection("test")
        if collection is not None:
            chroma_client.delete_collection("test")
    except Exception:
        pass
    finally:
        chroma_collection = chroma_client.create_collection("test")

    return chroma_collection


def split_documents(chroma_collection):

    convert_pdf_to_md(input_dir=pdfs_dir, output_dir=md_dir, api_key=openAI_api_key)
    document_sections = parse_markdown_sections(file_directory=md_dir)

    for docs in document_sections:
        chroma_collection.add(
            documents = [doc.page_content for doc in docs],
            metadatas=[doc.metadata for doc in docs],
            ids = [str(uuid.uuid4()) for _ in docs],
        )


def get_results(query):
    results = chroma_collection.query(
        query_texts=[query],
        n_results=8,
    )
    return results["documents"][0]

def construct_prompt(query):
    documents = get_results(query)
    context = "\n\n".join(documents)

    prompt = f"""
        You answer questions about financial documents.

        Answer the question using ONLY the context below.
        If the answer is not present, say "I don't know."

        Context:
        {context}

        Question:
        {query}
    """

    return prompt

def send_prompt(query):
    prompt = construct_prompt(query)
    response = openAI_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
        {"role": "system", "content": "You are a helpful financial assistant."},
        {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content

if __name__ == '__main__':

    query = input("What would you like to know: ")
    chroma_collection = create_collection()
    split_documents(chroma_collection)
    print(send_prompt(query))
