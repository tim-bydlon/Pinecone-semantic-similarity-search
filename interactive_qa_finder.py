import os
import pandas as pd
from pinecone import Pinecone
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
        self.index = self.pc.Index("quora-questions")

        # Load the embedding model (same one used for the dataset)
        print("Loading embedding model...")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Load the dataset to access question text
        print("Loading Quora dataset for text lookup...")
        self.dataset = load_dataset("quora_all-MiniLM-L6-bm25")

        # Create a lookup dictionary for fast question text retrieval
        print("Creating question text lookup...")
        self.question_lookup = self._create_question_lookup()

        print("Smart Q&A Finder ready!")
        print()

    def _create_question_lookup(self):
        """Create a dictionary mapping question IDs to their text for fast lookup."""
        docs = self.dataset.documents
        lookup = {}

        for _, row in docs.iterrows():
            question_id = str(row['id'])
            blob_data = row['blob']
            if isinstance(blob_data, dict) and 'text' in blob_data:
                lookup[question_id] = blob_data['text'].strip()

        print(f"Created lookup for {len(lookup)} questions")
        return lookup

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
        """Display search results with question text and similarity scores."""
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

            # Get the actual question text
            question_text = self.question_lookup.get(str(question_id), "Text not available")

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