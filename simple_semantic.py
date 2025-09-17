import os
from pinecone import Pinecone
from pinecone_datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SimpleSemanticSearch:
    def __init__(self):
        print("Initializing Simple Semantic Search...")

        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API"))
        self.index_name = "quora-simple-semantic"

        # Setup and load data
        self._setup_index_and_data()

        print("Simple Semantic Search ready!")
        print()

    def _setup_index_and_data(self):
        """Setup index with integrated embedding and load data."""
        # Check if index exists
        if not self.pc.has_index(self.index_name):
            print(f"Creating index '{self.index_name}' with integrated embedding...")

            # Create index with integrated embedding for text search
            self.pc.create_index_for_model(
                name=self.index_name,
                cloud="aws",
                region="us-east-1",
                embed={
                    "model": "llama-text-embed-v2",
                    "field_map": {"text": "question_text"}
                }
            )
            print("Index with integrated embedding created!")

            # Get the index
            self.index = self.pc.Index(self.index_name)

            # Load and upsert the Quora dataset
            self._load_quora_data()
        else:
            print(f"Index '{self.index_name}' already exists.")
            # Get the index
            self.index = self.pc.Index(self.index_name)

            # Check if index is empty and load data if needed
            stats = self.index.describe_index_stats()
            if stats['total_vector_count'] == 0:
                print("Index is empty, loading Quora data...")
                self._load_quora_data()
            else:
                print(f"Index contains {stats['total_vector_count']} vectors")

    def _load_quora_data(self):
        """Load Quora dataset and upsert with text for integrated embedding."""
        print("Loading Quora dataset...")
        dataset = load_dataset("quora_all-MiniLM-L6-bm25")
        docs = dataset.documents

        print("Upserting all Quora questions with text for integrated embedding...")

        batch_size = 96  # Maximum batch size for integrated embedding
        records_to_upsert = []
        count = 0

        for _, row in docs.iterrows():
            # Get question text from blob
            blob_data = row['blob']
            question_text = blob_data.get('text', '').strip() if isinstance(blob_data, dict) else ''

            if not question_text:
                continue

            # Create record with text (integrated embedding will handle the vectors)
            record = {
                "_id": str(row['id']),
                "question_text": question_text,  # This field will be embedded automatically
                "source": "quora"
            }

            records_to_upsert.append(record)
            count += 1

            # Upsert in batches (use upsert_records for integrated embedding)
            if len(records_to_upsert) >= batch_size:
                self.index.upsert_records("__default__", records_to_upsert)
                print(f"Upserted batch of {len(records_to_upsert)} records")
                records_to_upsert = []

        # Upsert remaining records
        if records_to_upsert:
            self.index.upsert_records("__default__", records_to_upsert)
            print(f"Upserted final batch of {len(records_to_upsert)} records")

        print(f"Successfully upserted {count} questions with integrated embedding!")

    def search_questions(self, user_question, top_k=5):
        """Search for similar questions using integrated text search."""
        try:
            results = self.index.search(
                namespace="__default__",
                query={
                    "inputs": {"text": user_question},
                    "top_k": top_k
                },
                fields=["question_text"]
            )
            return results
        except Exception as e:
            print(f"Error searching: {e}")
            return None

    def display_results(self, user_question, results):
        """Display search results."""
        print(f"Question: '{user_question}'")
        print("=" * 60)

        # Handle integrated embedding search results structure
        if not results or 'result' not in results or 'hits' not in results['result'] or not results['result']['hits']:
            print("No similar questions found.")
            return

        hits = results['result']['hits']
        print(f"Found {len(hits)} similar questions:")
        print()

        for i, hit in enumerate(hits, 1):
            question_id = hit['_id']
            similarity_score = hit['_score']

            # Get question text from fields
            question_text = hit.get('fields', {}).get('question_text', 'Text not available')

            print(f"{i}. Score: {similarity_score:.4f}")
            print(f"   Question: {question_text}")
            print()

    def run(self):
        """Main interactive loop."""
        print("Simple Semantic Search")
        print("=" * 50)
        print("Type 'quit' to exit")
        print()

        while True:
            try:
                user_input = input("Your question: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Thanks for using Simple Semantic Search!")
                    break

                if not user_input:
                    print("Please enter a question.")
                    continue

                print("\nSearching...")
                results = self.search_questions(user_input, top_k=5)

                print()
                self.display_results(user_input, results)
                print("-" * 60)
                print()

            except (KeyboardInterrupt, EOFError):
                print("\n\nThanks for using Simple Semantic Search!")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                break  # Exit on any other error to prevent infinite loop


def main():
    """Main entry point."""
    try:
        searcher = SimpleSemanticSearch()
        searcher.run()
    except Exception as e:
        print(f"Failed to initialize: {e}")


if __name__ == "__main__":
    main()