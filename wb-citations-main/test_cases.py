from retriever import CustomGoogleVertexAISearchRetriever
import pytest
from config import SAMPLE_QUERIES

@pytest.mark.parametrize('query',
                list(set([(x[0]) for x in SAMPLE_QUERIES])))
def test_search(query):
    structured_approach = 'vector' # One of 'hardcoded', 'dynamic', 'experimental', 'endpoint', 'vector'
    Retriever = CustomGoogleVertexAISearchRetriever(structured_approach=structured_approach)
    results = Retriever.get_relevant_documents(query, return_dict=True)
    summary = Retriever._get_most_relevant_summary()
    assert results is not None, "No results returned"
    assert summary is not None, "No summary produced"
