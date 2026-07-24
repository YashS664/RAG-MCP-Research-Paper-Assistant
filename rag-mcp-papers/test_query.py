import sys
import rag_utils

def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "What is Self-RAG"

    print(f"\nQuestion: {query}\n")
    result = rag_utils.answer_question(query)

    print("Answer:")
    print(result["answer"])

    print("\nSources:")
    for s in result["sources"]:
        print(f"  - {s['paper']} (similarity: {s['score']})")
    print()

if __name__ == "__main__":
    main()