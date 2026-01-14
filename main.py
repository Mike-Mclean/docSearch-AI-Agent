from dotenv import load_dotenv
import chromadb
import uuid
from pathlib import Path
from openai import OpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

openAI_client = OpenAI()

chroma_client = chromadb.Client()
directory_path = Path("test_docs")


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
                chunk_size=600,
                chunk_overlap=100,
                length_function=len,
                is_separator_regex=False,
            )

        documents = text_splitter.create_documents(
            [data],
            metadatas=[{"title": txt_file.name}]
        )

        chroma_collection.add(
            documents = [doc.page_content for doc in documents],
            metadatas=[doc.metadata for doc in documents],
            ids = [str(uuid.uuid4()) for _ in documents],
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
