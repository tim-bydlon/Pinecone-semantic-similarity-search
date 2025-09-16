import os
from pinecone import Pinecone, ServerlessSpec
from pinecone_datasets import list_datasets, load_dataset
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def main():
    # Check if we have the API key first
    api_key = os.getenv("PINECONE_API")
    if not api_key:
        print("No PINECONE_API environment variable found.")
        return

    try:
        # Load the Quora dataset
        print("Loading Quora dataset...")
        dataset = load_dataset("quora_all-MiniLM-L6-bm25")
        print(f"Dataset loaded successfully!")
        print(f"Dataset contains {len(dataset)} question records")

        # Initialize Pinecone
        print("\nInitializing Pinecone...")
        pc = Pinecone(api_key=api_key)
        index_name = "quora-questions"

        if not pc.has_index(index_name):
            print(f"Creating index '{index_name}'...")
            pc.create_index(
                name=index_name,
                dimension=384,  # MiniLM embedding dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print("Index created!")
        else:
            print(f"Index '{index_name}' already exists.")

        # Get the index object
        index = pc.Index(index_name)

        # Upsert sample data using the recommended method from docs
        print("\nUpserting sample data to index...")

        # Method 1: Using iter_documents (more control)
        batch_count = 0
        for batch in dataset.iter_documents(batch_size=100):
            index.upsert(vectors=batch)
            batch_count += 1
            if batch_count >= 10:  # Limit to ~1000 records for testing
                break
            print(f"Upserted batch {batch_count}")

        print("Sample data upserted successfully!")

        # Show some sample questions from the dataset
        print("\nSample questions from dataset:")
        sample_batch = next(dataset.iter_documents(batch_size=5))
        for i, record in enumerate(sample_batch[:3]):  # Show first 3 questions
            # Based on Pinecone docs, records have id, values, metadata structure
            record_id = record.get('id', 'No ID')
            metadata = record.get('metadata', {})
            # Check if there's text in metadata or if we need to access blob differently
            print(f"ID: {record_id}")
            print(f"Metadata: {metadata}")
            if i == 0:  # Show full structure for first record
                print(f"Full record structure: {list(record.keys())}")
            print("---")

    except Exception as e:
        print(f"Error: {e}")
        print("Failed to load dataset or upsert data.")


if __name__ == "__main__":
    main()