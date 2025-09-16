import os
from pinecone import Pinecone, ServerlessSpec


def main():
    pc = Pinecone(api_key=os.getenv("PINECONE_API"))

    index_name = "developer-quickstart-py"

    if not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model":"llama-text-embed-v2",
                "field_map":{"text": "chunk_text"}
            }
        )


if __name__ == "__main__":
    main()