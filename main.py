import json
from dotenv import load_dotenv
import chromadb
import uuid
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

chroma_client = chromadb.Client()
directory_path = Path("test_docs")

query = "What is the client fee schedule?"

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

    for txt_file in directory_path.iterdir():
        with open(f"{directory_path}/{txt_file.name}", 'r') as file:
            data = file.read()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=150,
                chunk_overlap=50,
                length_function=len,
                is_separator_regex=False,
            )
        texts = text_splitter.create_documents([data])
        data = [item.page_content for item in texts]
        document_ids = [str(uuid.uuid4()) for _ in range(len(data))]

        chroma_collection.add(
        documents = data,
        metadatas=[{"title": f"{txt_file.name}"} for _ in range(len(data))],
        ids = document_ids,
    )

def get_results(query):
    results = chroma_collection.query(
        query_texts=[query],
        n_results=2,
    )
    results = json.dumps(results, indent=4)
    print(results)


if __name__ == '__main__':

    chroma_collection = create_collection()
    split_documents(chroma_collection)
    get_results(query)
