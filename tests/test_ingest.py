import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingest import chunk_text

def test_chunk_test_split_basic():
    """A string longer than chunk_size should be split into multiple chunks."""
    text = "a" * 1000
    chunks = chunk_text(text, chunk_size=400, overlap=50)
    assert len(chunks) > 1 

def test_chunk_text_overlap():  # At the end of chunk 1 should reappear at the start of chunk 2
    """Consecutive chunks should share overlapping content at the boundary."""
    text = "0123456789" * 20
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert chunks[0][-20] == chunks[1][:20]

def test_chunk_text_short_input_single_chunk():
    """Text shorter than chunk_size should come back as exactly one chunk."""
    text = "a short paper abstract"
    chunks = chunk_text(text, chunk_size=800, overlap=150)
    assert len(chunks) == 1
    assert chunks[0] == text

def test_chunk_text_empty_input():
    """Empty text should produce no chunks, not an error."""
    chunks = chunk_text("", chunk_size=800, overlap=150)
    assert chunks == []

def test_chunk_text_no_empty_chunks():
    """No chunk in the output should ever be an empty/whitespace-only string."""
    text = "word " * 500
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert all(chunk.strip() != "" for chunk in chunks)
