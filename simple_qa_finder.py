import os
from pinecone import Pinecone
from pinecone_datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SimpleQAFinder:
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API"))
        self.index = self.pc.Index("quora-questions")
        self.dataset = load_dataset("quora_all-MiniLM-L6-bm25")

    def find_similar_questions(self, top_k=5):
        """
        Find similar questions using a sample vector from the dataset.
        In a real implementation, you'd embed the user's question.
        """
        try:
            # Get a random vector from our dataset to demonstrate similarity search
            sample_batch = next(self.dataset.iter_documents(batch_size=1))
            sample_vector = sample_batch[0]['values']

            # Convert to list if needed
            if hasattr(sample_vector, 'tolist'):
                sample_vector = sample_vector.tolist()
            elif not isinstance(sample_vector, list):
                sample_vector = list(sample_vector)

            # Query the index
            results = self.index.query(
                vector=sample_vector,
                top_k=top_k,
                include_metadata=True
            )

            return results

        except Exception as e:
            print(f"Error searching: {e}")
            return None

    def demo_search(self):
        """
        Demonstrate the Smart Q&A Finder functionality.
        """
        print("Smart Q&A Finder Demo")
        print("=" * 50)
        print()
        print("This system finds similar questions from Quora's dataset.")
        print("In a production system, you would:")
        print("1. Embed the user's question using the same model (MiniLM)")
        print("2. Search for similar question vectors")
        print("3. Return existing answers for those questions")
        print()
        print("Current demo shows vector similarity search results:")
        print("-" * 50)

        results = self.find_similar_questions()

        if results and results.matches:
            print(f"Found {len(results.matches)} similar questions:")
            print()

            for i, match in enumerate(results.matches, 1):
                question_id = match.id
                similarity_score = match.score

                print(f"{i}. Question ID: {question_id}")
                print(f"   Similarity Score: {similarity_score:.4f}")
                print()

            print("Key Benefits:")
            print("- Instant semantic search across 500K+ questions")
            print("- Pre-computed embeddings for fast retrieval")
            print("- Cosine similarity for accurate matching")
            print("- Scalable vector database infrastructure")

        else:
            print("No similar questions found.")

    def interactive_demo(self):
        """
        Interactive demo allowing user input.
        """
        print("Interactive Smart Q&A Finder")
        print("=" * 40)
        print("Enter questions to find similar ones from Quora's dataset")
        print("(Note: This demo uses sample vectors - in production,")
        print(" your question would be embedded and searched)")
        print()

        while True:
            user_input = input("Enter your question (or 'quit' to exit): ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Thanks for trying the Smart Q&A Finder!")
                break

            if user_input:
                print(f"\nSearching for questions similar to: '{user_input}'")
                print("-" * 50)

                results = self.find_similar_questions(top_k=3)

                if results and results.matches:
                    print(f"Found {len(results.matches)} similar questions:")
                    print()

                    for i, match in enumerate(results.matches, 1):
                        question_id = match.id
                        similarity_score = match.score
                        print(f"{i}. Question ID: {question_id} (Score: {similarity_score:.4f})")

                    print()
                    print("In a full implementation, these IDs would link to:")
                    print("- Original question text")
                    print("- Top-rated answers")
                    print("- Related questions")
                    print("- User voting data")

                else:
                    print("No similar questions found.")

                print("\n" + "=" * 50 + "\n")


def main():
    finder = SimpleQAFinder()
    finder.demo_search()


if __name__ == "__main__":
    main()