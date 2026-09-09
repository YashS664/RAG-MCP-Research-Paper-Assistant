import sys 
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_utils import _build_prompt

def test_build_prompt_includes_question():
    chunks = [{"paper": "self_rag", "chunk_index": 0, "text": "Self-RAG uses reflection tokens."}]
    prompt = _build_prompt("What is Self-RAG?", chunks)
    assert "What is Self-RAG?" in prompt

def test_build_prompt_includes_source_paper_name():
    chunks = [{"paper": "self_rag", "chunk_index": 3, "text": "Some excerpt text."}]
    prompt = _build_prompt("test query", chunks)
    assert "self_rag" in prompt

def  test_build_prompt_includes_all_chunks():
    chunks = [
        {"paper": "rag", "chunk_index": 0, "text": "First excerpt."},
        {"paper": "self_rag", "chunk_index": 1, "text": "Second excerpt."},
    ]
    prompt = _build_prompt("test query", chunks)
    assert "First excerpt." in prompt
    assert "Second excerpt." in prompt

def test_build_prompt_instructs_citation():
    """The prompt should explicitly instruct the model to cite sources."""
    chunks = [{"paper": "react", "chunk_index": 0, "text": "excerpt"}]
    prompt = _build_prompt("test query", chunks)
    assert "cite" in prompt.lower()