import os
import numpy as np
import pandas as pd
import json
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=key)

# chroma_key = os.getenv("CHROMA_API_KEY")
# chroma_tenant = os.getenv("CHROMA_TENANT")
# chroma_client = chromadb.CloudClient(
#   api_key=chroma_key,
#   tenant=chroma_tenant,
#   database='Fin_Doc Search'
# )

chroma_client = chromadb.Client()

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


def split_document():
    with open("Test Docs/doc3.txt", 'r') as file:
        data= file.read()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=150,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
    texts = text_splitter.create_documents([data])
    data = [item.page_content for item in texts]
    return data

def add_collections(chroma_collection):
    data = split_document()
    document_ids = [str(uuid.uuid4()) for _ in range(len(data))]

    chroma_collection.add(
        documents = data,
        ids = document_ids,
    )

def get_results(query):
    results = chroma_collection.query(
        query_texts=[query],
        n_results=2,
    )
    results = json.dumps(results, indent=4)
    print(results)


# def generate_embeddings(text):
#     response = client.embeddings.create(
#         input=text,
#         model="text-embedding-3-small"
#     )
#     response = json.loads(response.model_dump_json())
#     embeddings = response['data'][0]['embedding']
#     return np.array(embeddings)

# def cosine_similarity(vec1, vec2):
#     dot_product = np.dot(vec1, vec2)
#     norm_vec1 = np.linalg.norm(vec1)
#     norm_vec2 = np.linalg.norm(vec2)
#     similarity = dot_product / (norm_vec1 * norm_vec2)
#     return similarity


if __name__ == '__main__':

    chroma_collection = create_collection()
    add_collections(chroma_collection)
    get_results(query)
