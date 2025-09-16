import os
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from pinecone_datasets import load_dataset
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class InteractiveQAFinder:
    def __init__(self):
        print("Initializing Smart Q&A Finder...")

        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API"))
        self.index_name = "quora-questions-with-metadata"

        # Load the embedding model (same one used for the dataset)
        print("Loading embedding model...")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Setup index with proper metadata following Pinecone best practices
        self._setup_index_with_metadata()

        print("Smart Q&A Finder ready!")
        print()

    def _setup_index_with_metadata(self):
        """Setup index with proper metadata following Pinecone best practices."""

        # Create index if it doesn't exist
        if not self.pc.has_index(self.index_name):
            print(f"Creating index '{self.index_name}' with metadata support...")
            self.pc.create_index(
                name=self.index_name,
                dimension=384,  # MiniLM embedding dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print("Index created!")
        else:
            print(f"Index '{self.index_name}' already exists.")

        self.index = self.pc.Index(self.index_name)

        # Check if we need to upsert data with metadata
        stats = self.index.describe_index_stats()
        if stats['total_vector_count'] == 0:
            self._upsert_with_metadata()

    def _upsert_with_metadata(self):
        """Upsert data with text stored in metadata (Pinecone best practice)."""
        print("Loading Quora dataset...")
        dataset = load_dataset("quora_all-MiniLM-L6-bm25")
        docs = dataset.documents

        print("Upserting data with metadata (following Pinecone best practices)...")

        batch_size = 100
        max_records = 1000  # Limit for demo
        vectors_to_upsert = []
        count = 0

        for _, row in docs.iterrows():
            if count >= max_records:
                break

            # Get question text from blob
            blob_data = row['blob']
            question_text = blob_data.get('text', '').strip() if isinstance(blob_data, dict) else ''

            if not question_text:
                continue

            # Convert vector to list if needed
            vector = row['values']
            if hasattr(vector, 'tolist'):
                vector = vector.tolist()
            elif not isinstance(vector, list):
                vector = list(vector)

            # Create record with metadata (PINECONE BEST PRACTICE)
            record = {
                "id": str(row['id']),
                "values": vector,
                "metadata": {
                    "question_text": question_text,  # Store text in metadata!
                    "source": "quora",
                    "record_index": count
                }
            }

            vectors_to_upsert.append(record)
            count += 1

            # Upsert in batches
            if len(vectors_to_upsert) >= batch_size:
                self.index.upsert(vectors=vectors_to_upsert)
                print(f"Upserted batch of {len(vectors_to_upsert)} records")
                vectors_to_upsert = []

        # Upsert remaining records
        if vectors_to_upsert:
            self.index.upsert(vectors=vectors_to_upsert)
            print(f"Upserted final batch of {len(vectors_to_upsert)} records")

        print(f"Successfully upserted {count} records with metadata!")

    def embed_question(self, question_text):
        """Convert a question to vector embedding using the same model as the dataset."""
        try:
            # Generate embedding using all-MiniLM-L6-v2 (384 dimensions)
            embedding = self.model.encode(question_text)

            # Convert to list for Pinecone compatibility
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            elif not isinstance(embedding, list):
                embedding = list(embedding)

            return embedding
        except Exception as e:
            print(f"Error embedding question: {e}")
            return None

    def find_similar_questions(self, user_question, top_k=5):
        """Find similar questions using actual semantic search."""
        try:
            # Embed the user's question
            query_vector = self.embed_question(user_question)
            if query_vector is None:
                return None

            # Search for similar questions in Pinecone
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )

            return results

        except Exception as e:
            print(f"Error searching for similar questions: {e}")
            return None

    def display_results(self, user_question, results):
        """Display search results using metadata (Pinecone best practice)."""
        print(f"Question: '{user_question}'")
        print("=" * 60)

        if not results or not results.matches:
            print("No similar questions found.")
            return

        print(f"Found {len(results.matches)} similar questions:")
        print()

        for i, match in enumerate(results.matches, 1):
            question_id = match.id
            similarity_score = match.score

            # Get text directly from metadata (PINECONE BEST PRACTICE - no external lookup needed!)
            question_text = match.metadata.get('question_text', 'Text not available')

            print(f"{i}. Score: {similarity_score:.4f}")
            print(f"   Question: {question_text}")
            print()

    def interactive_demo(self):
        """Main interactive demo loop."""
        print("🔍 Interactive Smart Q&A Finder")
        print("=" * 50)
        print("Ask any question and find semantically similar questions from Quora's dataset!")
        print("Type 'quit' to exit")
        print()

        while True:
            try:
                user_input = input("Your question: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Thanks for using Smart Q&A Finder!")
                    break

                if not user_input:
                    print("Please enter a question.")
                    continue

                print("\nSearching...")
                results = self.find_similar_questions(user_input, top_k=5)

                print()
                self.display_results(user_input, results)
                print("-" * 60)
                print()

            except KeyboardInterrupt:
                print("\n\nThanks for using Smart Q&A Finder!")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                print("Please try again.")
                print()

    def demo_search(self, sample_questions=None):
        """Run demo with sample questions."""
        if sample_questions is None:
            sample_questions = [
                "How do I learn Python programming?",
                "What is machine learning?",
                "How to start a business?",
                "What are the best books to read?",
                "How do I invest in stocks?"
            ]

        print("🔍 Smart Q&A Finder Demo")
        print("=" * 50)
        print("Demonstrating semantic question similarity search...")
        print()

        for i, question in enumerate(sample_questions, 1):
            print(f"Demo {i}/{len(sample_questions)}")
            print("-" * 30)

            results = self.find_similar_questions(question)
            self.display_results(question, results)

            print("=" * 60)
            print()


def main():
    """Main entry point."""
    try:
        finder = InteractiveQAFinder()

        print("Choose mode:")
        print("1. Interactive mode (type your own questions)")
        print("2. Demo mode (pre-defined sample questions)")
        print()

        while True:
            try:
                choice = input("Enter choice (1 or 2): ").strip()

                if choice == "1":
                    finder.interactive_demo()
                    break
                elif choice == "2":
                    finder.demo_search()
                    break
                else:
                    print("Please enter 1 or 2")

            except (EOFError, KeyboardInterrupt):
                # Handle non-interactive environments by defaulting to demo
                print("Running demo mode...")
                finder.demo_search()
                break

    except Exception as e:
        print(f"Failed to initialize: {e}")
        print("Please check your setup and try again.")


if __name__ == "__main__":
    main()