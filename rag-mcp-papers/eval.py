import rag_utils

# Each entry: (question, expected paper filename, without .pdf)
# Adjust these paper names to match whatever you named your PDFs in data/papers/

EVAL_SET = [
    ("What is Self-RAG and how does it use reflection token?", "self_rag"),
    ("How does ReAct combine reasoning and acting?", "react"),
    ("How does Toolformer decide when to call an API?", "toolformer"),
    ("What is chain-of-thought prompting?","cot"),
    ("What is retrieval-augmented generation?", "rag"),
]

TOP_K = 5 

def run_eval():
    correct = 0 
    print(f"Running {len(EVAL_SET)} evaluation questions (top_k-{TOP_K})...\n")

    for question, expected_paper in EVAL_SET:
        results = rag_utils.retrieve(question, top_k=TOP_K)
        retrieved_papers = [r["paper"] for r in results]

        hit = expected_paper in retrieved_papers
        correct += hit

        status = "PASS" if hit else "FAIL"
        print(f"[{status}] \"{question}\"")
        print(f"    expected: {expected_paper}")
        print(f"    retrieved: {retrieved_papers}")
        print()

    accuracy = correct / len(EVAL_SET) * 100
    print(f"Retrieval accuracy: {correct}/{len(EVAL_SET)} ({accuracy:.0f}%)")


if __name__  == "__main__":
    run_eval()