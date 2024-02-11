
import re
import timeit
from vertexai.language_models import TextEmbeddingModel
from google.cloud import bigquery, aiplatform
from typing import Any, List, Optional, Union
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import MatchNeighbor
import logging

logger = logging.getLogger('Vector Search')
logger.setLevel(logging.INFO)
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

PROJECT_ID = 'rajat-demo-354311'
EMBEDDING_MODEl = "textembedding-gecko@001"
BUCKET_URI = 'vector-search-index-wb-people'
REGION = "us-central1"
client = bigquery.Client()
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

def encode_texts_to_embeddings(sentences: List[str]) -> List[Optional[List[float]]]:
    """Process and upload embedding data for a row in a 
    database table"""
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEl)
    try:
        embeddings = model.get_embeddings(sentences)
        return [embedding.values for embedding in embeddings]
    except Exception:
        logger.error("Could not encode query")
        return [None for _ in range(len(sentences))]
    
def fetch_nearest_neighbours(sentences: Union[str, List[str]], 
                             endpoint_ref: Union[str, aiplatform.MatchingEngineIndexEndpoint],
                             endpoint_unique_id: str,
                             n_neighbours: int=10) -> List[List[MatchNeighbor]]:
    """Fetch nearest n neighbours for a sentence or an array of sentences"""
    sentences = [sentences] if not isinstance(sentences, list) else sentences        
    embeddings = encode_texts_to_embeddings(sentences=sentences)
    endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_ref) if isinstance(endpoint_ref, str) else endpoint_ref
    logger.info(f"Finding nearest {n_neighbours} neighbours")
    response = endpoint.find_neighbors(
        deployed_index_id=endpoint_unique_id,
        queries=embeddings,
        num_neighbors=n_neighbours,
    )
    return response



def extract_project_ids(query: str) -> List[str]:
    """Extracts pid from a query"""
    try:
        pattern = r'p\d+'
        project_ids= re.findall(pattern, query,re.IGNORECASE)
        logger.info(f"Extracted project ids: {project_ids}")
        return project_ids
    except Exception:
        return []

def extract_upi(query: str) -> List[str]:
    """Extracts upi from a query"""
    try:
        #14621438 this is a upi number. how to extract 8 sequence of number?
        pattern = r'\d{8}'
        upi= re.findall(pattern, query,re.IGNORECASE)
        logger.info(f"Extracted upi: {upi}")
        return upi
    except Exception:
        return []


def return_rows_for_sentences(sentences: Union[str, List[str]], 
                              table_name: str,
                              endpoint_ref: Union[str, aiplatform.MatchingEngineIndexEndpoint],
                              endpoint_unique_id: str,
                              id_column: str='id',
                              n_neighbours: int=10) -> List[List[Any]]:
    """Fetches nearest neighbours for a query and returns corresponding rows from a table"""
    st=timeit.default_timer()
    sentences = [sentences] if not isinstance(sentences, list) else sentences 
    result = fetch_nearest_neighbours(sentences, endpoint_ref, endpoint_unique_id, n_neighbours)
    fetch_time = timeit.default_timer()-st
    output = []
    extracted_pids=[]
    extracted_upis=[]
    for sentence in sentences:
        extracted_pids+=extract_project_ids(sentence)
        extracted_upis+=extract_upi(sentence)
    for sentence in result:
        st=timeit.default_timer()
        ids = [x.id for x in sentence]
        idstring = ','.join([f'"{x.id}"' for x in sentence])
        for pid in extracted_pids:
            if pid in ids:
                ids.remove(pid)
            ids.insert(0,pid)
        for pid in extracted_pids:
            idstring+=f',"{pid}"'
        for upi in extracted_upis:
            if upi in ids:
                ids.remove(upi)
            ids.insert(0,upi)
        for upi in extracted_upis:
            idstring+=f',"{upi}"' 
        querystring = f"SELECT * FROM {table_name} WHERE {id_column} IN ({idstring})"
        logger.info(f"Sending results to BQ with filter: {querystring}")
        r = client.query(querystring)
        bq_results = [row for row in r.result()]
        bigquery_time = timeit.default_timer()-st
        try:
            results = {row.get(id_column): row for row in bq_results}
            sorted_results = [results.get(k) for k in ids if k in results]
            output.append(sorted_results)
        except Exception as e:
            logger.error(str(e))
    return output, fetch_time, bigquery_time
