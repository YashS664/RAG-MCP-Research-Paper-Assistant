import sys
import os
import logging
import traceback
import functools

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests
import rag_utils
from mcp.server.fastmcp import FastMCP

LOG_PATH = os.path.join(os.path.dirname(__file__), "server.log")
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("rag-papers-mcp")

# To run inspector on stdio tansport 
# mcp = FastMCP("rag-papers")
# To run inspector on sse tansport
mcp = FastMCP("rag-papers", host="127.0.0.1", port=8000)

def logged_tool(func):
    """Decorator: logs every tool call (args, success/failure, and errors)."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"CALL {func.__name__} args={args} kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"OK  {func.__name__}")
            return result
        except Exception:
            logger.error(f"FAIL {func.__name__}\n{traceback.format_exc()}")
            raise 
    return wrapper

@mcp.tool()
@logged_tool
def search_papers(query: str) -> str:
    """
    Search the indexed research papers for chunks relevant to the query.
    Returns raw matching excerpts with their source paper -- no LLM generation,
    just retrieval. Useful for quickly checking what the knowledge base contains.
    """
    results = rag_utils.retrieve(query, top_k=5)
    if not results:
        return "No relevant excerpts found."
    
    lines = []
    for r in results:
        lines.append(
            f"[{r['paper']} | chunk {r['chunk_index']} | score {r['score']:.3f}]\n{r['text']}\n"
        )
    return "\n---\n".join(lines)

    
@mcp.tool()
@logged_tool
def answer_question(query: str) -> str:
    """
    Answer a question using RAG over the indexed papers: retrieves relevant
    excerpts and generates a synthesized answer with citations to the source papers.
    """
    result = rag_utils.answer_question(query)
    sources_str = ", ".join(s["paper"] for s in result["sources"])
    return f"{result['answer']}\n\nSources: {sources_str}"


@mcp.tool()
@logged_tool
def list_papers() -> str:
    """List which papers are currently indexed in the knowledge base."""
    rag_utils._load_resources()
    papers = sorted(set(m["paper"] for m in rag_utils._metadata))
    return "Indexed papers:\n" + "\n".join(f"- {p}" for p in papers)

@mcp.tool()
@logged_tool
def get_arxive_metadata(paper_id: str) -> str:
    """
    Look up LIVE metadata for a paper directly from the real arxiv API
    (title, authors, abstract, published date) -- given an arxiv ID like
    '2310.11511'. This calls an external, real-world system, not local data.
    """
    url = f"http://export.arxiv.org/api/query?id_list={paper_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    import xml.etree.ElementTree as ET
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(response.text)
    entry = root.find("atom:entry", ns)
    if entry is None:
        return f"No arXiv entry found for id {paper_id}"
    
    title = entry.find("atom:title", ns).text.strip()
    published = entry.find("atom:published", ns).text.strip()
    authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
    summary = entry.find("atom:summary", ns).text.strip()

    return (
        f"Title: {title}\n"
        f"Authors: {', '.join(authors)}\n"
        f"Published: {published}\n\n"
        f"Abstract: {summary}"
    )


if __name__ == "__main__":
    if "--http" in sys.argv:
        logger.info("Starting Server over HTTP (SSE teansport) on port 8000")
        mcp.run(transport="sse")
    else:
        logger.info("Starting server over stdio")
        mcp.run()