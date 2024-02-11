# test
import asyncio, httpx, json, uuid, vertexai
from config import MetadataConfig, DocumentConfig, DatastoreConfig, DATASTORE_CONFIG, LLMParams, TAllowedIntents
from google import auth as gauth
from google.cloud import bigquery, dialogflowcx_v3beta1 as dialogflow, discoveryengine_v1alpha as discoveryengine_v1, aiplatform
from google.protobuf.json_format import MessageToDict
from langchain.llms import VertexAI
from langchain.schema.document import Document
import logging
from vertexai.preview.generative_models import GenerativeModel
from vertexai.language_models import CodeGenerationModel
from typing import List, Union, Optional, Dict, Tuple, Literal
from vector_search import return_rows_for_sentences
import re
import timeit

logger = logging.getLogger('Retriever')
# Set some constants

TEXT_MODEL = "text-bison@002"
CODE_MODEL = 'code-bison@001'
PROJECT_ID = 'rajat-demo-354311'
LOCATION = 'us-central1'
DATA_STORE = 'wb-unstructured-with-metad_1690319343422'
DEFAULT_LLM_PARAMS: LLMParams = {"max_output_tokens": 1024, "temperature": 0.2}
DIALOGFLOW_AGENT = 'aab90aca-87ee-46e6-913e-037c8116a2d4'

# Experimental
TEXT2SQL_ENDPOINT = "1185898057347104768"
TEXT2SQL_LOCATION = "us-west2"
bigquery_client=bigquery.Client(project=PROJECT_ID)
llm_model=GenerativeModel(TEXT_MODEL)
code_model=CodeGenerationModel.from_pretrained(CODE_MODEL)
search_client=discoveryengine_v1.SearchServiceClient()
dialogflow_client=dialogflow.SessionsClient()
httpx_client=httpx.Client()
gauth_default=gauth.default()
vertexai.init(project=PROJECT_ID, location=LOCATION)
llm = VertexAI(model_name=TEXT_MODEL, temperature=0)

class AgentIntentException(Exception):
  "Raised when no Intent is matched"
  pass

class AgentSQLException(Exception):
  "Raised when some error occurs in BigQuery"
  pass

class AgentLLMException(Exception):
  "Raised when some error occurs from an LLM call"
  pass

TStructuredApproaches = Literal['hardcoded', 'dynamic', 'endpoint', 'experimental', 'vector']

# Naive implementation of a multi-document retriever
class CustomGoogleVertexAISearchRetriever():
  """A custom implementation of an object compatible with langchain's retriever class.
  This is designed to fulfil one primary function: for a given query, return
  an array of langchain `Documents` to be used for summarization and conversation.
  We also allow the use of native summaries produced by vertex AI search.

  There are several components to this

  """
  def __init__(self,
               project:str=PROJECT_ID,
               location:str=LOCATION,
               datastore_config:Dict[TAllowedIntents, DatastoreConfig]=DATASTORE_CONFIG,
               max_query_rows:int=300,
               summary_result_count:int=3,
               single_source_confidence:float=0.95, # Above this we will not run multiple searches
               single_minimum_source_confidence:float=0.05, # below this we will not run multiple searches
               structured_approach:TStructuredApproaches = 'vector',
               max_results_per_query:int=10
               ):
    st=timeit.default_timer()
    self.bq_client = bigquery_client
    self.llm = llm_model
    self.code_model = code_model
    self.search_client = search_client
    self.df_client = dialogflow_client
    self.max_query_rows = max_query_rows
    self.project = project
    self.location = location
    self.single_source_confidence = single_source_confidence
    self.single_minimum_source_confidence = single_minimum_source_confidence
    self.datastore_config = datastore_config
    self.structured_approach = structured_approach
    self.max_results_per_query = max_results_per_query
    self.timestamps={}

    self.dialogflow_agent = DIALOGFLOW_AGENT
    self._llm_param_defaults = DEFAULT_LLM_PARAMS
    logger.info(f"Initialised in {timeit.default_timer()-st} seconds")
    
    self._setup_http_client()
    self._reset_cache()
    self._set_search_defaults(project, summary_result_count)

  # Utility functions
  def _setup_http_client(self):
    self.httpx_client = httpx_client
    self.httpx_creds, _project = gauth_default
    self.httpx_client.headers.update({'X-Goog-User-Project': self.project, 'Content-Type': "application/json; charset=utf-8",})
    self._refresh_credentials()

  def _refresh_credentials(self):
    """Refresh the auth token"""
    auth_req = gauth.transport.requests.Request()
    self.httpx_creds.refresh(auth_req)
    self.httpx_client.headers.update({'Authorization': f"Bearer {self.httpx_creds.token}"})

  def _set_search_defaults(self,
                           project,
                           summary_result_count,
                           include_citations=True):
    self._serving_config_defaults = {
      "project": project,
      "location": "global",
      "serving_config": "default_config"
    }
    self.content_search_spec = {
      "summary_spec": {
          "summary_result_count": summary_result_count,
          "include_citations": include_citations,
           "model_prompt_spec": {"preamble" : "Give a summary in detail."}
            },
      "extractive_content_spec": {
        "max_extractive_answer_count": 3
        },
       
      "snippet_spec": {
        "return_snippet": True
      }
      }


  def _reset_cache(self):
    self.intent_cache = {}
    self.result_cache = {}
    self.history = set()
    self.errors = []

  def _make_llm_params(self, **kwargs):
    return self._llm_param_defaults | {**kwargs}

  def _make_serving_config(self, config:DocumentConfig):
    params = self._serving_config_defaults | {"data_store": config.get('datastore_id')}
    return self.search_client.serving_config_path(**params)

  def _make_search_history_key(self, query_str, datastore_id, datastore_type):
    return hash(f"{query_str}-{datastore_id}-{datastore_type}")
  
  def _get_sorted_intents(self):
    """Return a Dict[intent_name, intent_scores] sorted descending by score"""
    return dict(sorted(self.intent_cache.items(), key=lambda x: x[1][0], reverse=True))
  
  def _get_most_relevant_summary(self):
    """Return the most relevant summary, or None"""
    for k in self._get_sorted_intents().keys():
      if k in self.result_cache:
        summary = self.result_cache[k].get('summary')
        if summary is not None:
            citation_dict = self.result_cache[k].get('citation_dict')
            matches = re.finditer(r'\[(\d+(?:,\s*\d+)*)\]', summary) #updated expression as now the summary may include more than one number in a single bracket
            new_summary = "\n\nSOURCES:"
            numbers_added = set()
            for match in matches:
                numbers = [int(num.strip()) for num in match.group(1).split(',')]
                for number in numbers:
                    if number not in numbers_added:
                      link = citation_dict.get(number)
                      if link:
                          link = link.replace("gs://", "https://storage.cloud.google.com/")
                          new_summary += f"\n[{number}] {link}"
                          numbers_added.add(number)
            summary += new_summary 
        return summary
    return None
        
  async def _afetch_results_all(self, query: str) -> Tuple[List[Tuple[str, str]], Union[str, None]]:
    """Run a query against all datastores and return combined results.
    Returns an array of results in in Tuple[content, document_name] format, then
    either a str summary or None if no summary is provided.
    """
    tasks = []
    logger.info(f"Running query '{query}'")
    st=timeit.default_timer()
    query = self._detect_intents_for_query(query)
    logger.info(f"expanded query: {query}")
    sorted_intents =  self._get_sorted_intents()
    self.timestamps['detect_intent'] = timeit.default_timer()-st
    for i_type, i_name, i_score in sorted_intents.values():
      # if i_score < self.single_minimum_source_confidence:
      #   logger.info(f"Confidence score {i_score} is below {self.single_minimum_source_confidence} threshold. Not running any further searches")
      #   break
      self.result_cache[i_name] = {"summary": None, "results": {"metadata": [], "documents": []}}
      if i_type._name_ == 'NO_MATCH':
        msg = f"Query '{query}' reported a 'No match' intent with a score of {i_score}"
        logger.warning(msg)
      elif i_type._name_ == 'INTENT':
        data = self.datastore_config.get(i_name)
        if data is not None:
          metadata_config_target = 'sql' if self.structured_approach in ['hardcoded', 'dynamic'] else self.structured_approach # Both dynamic and hardcoded use the 'SQL' config
          metadata = data.get(metadata_config_target)
          documents = data.get('documents')

          # Run Document search
          if documents is not None:
            k = self._make_search_history_key(query, i_name, 'documents')
            if k not in self.history:
              tasks.append(asyncio.create_task(self._arun_document_search(i_name, query, documents)))
              self.history.add(k)
          # Run Structured search
          if metadata is not None:
            k = self._make_search_history_key(query, i_name, self.structured_approach)
            if k not in self.history:
              try:
                if self.structured_approach == 'experimental':
                  tasks.append(asyncio.create_task(self._arun_experimental_search(i_name, query, metadata)))
                elif self.structured_approach == 'vector':
                  tasks.append(asyncio.create_task(self._arun_vector_search(i_name, query, metadata)))
                else:
                  tasks.append(asyncio.create_task(self._arun_metadata_search(i_name, query, metadata)))
              except AgentLLMException as e:
                logger.error(e)
                self.errors.append(str(e))
              self.history.add(k)
          
          if i_score >= self.single_source_confidence:
            logger.info(f"Confidence score exceeds {self.single_source_confidence} threshold. Not running any further searches")
            break
        else:
          logger.warning(f"No table found matching intent {i_name}")
    await asyncio.wait(tasks)

  async def _arun_vector_search(self,
                               intent_key:TAllowedIntents,
                               query:str,
                               metadata:MetadataConfig,):
    logger.info(f"Querying vector search engine {metadata.get('endpoint_unique_id')}")
    try:
      results,fetch_time, bigquery_time = return_rows_for_sentences(query, 
                                          metadata.get('name'),
                                          metadata.get('endpoint_ref'),
                                          metadata.get('endpoint_unique_id'),
                                          metadata.get('id_column'),
                                          self.max_results_per_query
                                          )
      if self.timestamps.get('vector_search') is None:
        self.timestamps['vector_search']=0
      self.timestamps['vector_search'] += fetch_time
      if self.timestamps.get('bigquery') is None:
        self.timestamps['bigquery']=0
      self.timestamps['bigquery'] += bigquery_time
      if len(results) > 0:
        results = results[0] # Only supports single query right now
        logger.info(f"{len(results)} results found")
      else:
        logger.warn("No results found")
      self.result_cache[intent_key]["results"]["metadata"] = results
    except Exception as e:
      msg = str(e)
      logger.error(f"Error in vector search: {msg}")
      self.errors.append(msg)
  
  async def _arun_experimental_search(self,
                                     intent_key:TAllowedIntents,
                                     query:str,
                                     metadata:MetadataConfig,
                                     retry:int=1):
    MAX_RETRIES = 3
    if retry > MAX_RETRIES:
      msg = 'Exceeded retries'
      logger.warning(msg)
      self.errors.append(msg)
      return None
    query = {
      "query": query,
      "pageSize": self.max_results_per_query,
      "natural_language_query_understanding_spec": {"filter_extraction_condition": 2},
    }
    try:
      url = f"https://staging-discoveryengine.sandbox.googleapis.com/v1alpha/projects/{metadata.get('project')}/locations/global/collections/default_collection/dataStores/{metadata.get('datastore_id')}/servingConfigs/default_config:search"
      r = self.httpx_client.post(url, data=json.dumps(query))
      r.raise_for_status()
      r = r.json()
      logger.debug(r.get('debugInfo'))
      if r.get('guidedSearchResult') == {}:
        logger.warning(f"No results returned for datastore `{metadata.get('datastore_id')}`")
      # else:
        # logger.info(f"{r.total_size} results returned")
      self.result_cache[intent_key]["results"]["metadata"] = r.get('guidedSearchResult')
      self.result_cache[intent_key]["summary"] = r.get('summary', {'summary_text': None}).get('summary_text')
    except httpx.HTTPStatusError as e:
      if e.response.status_code == 401:
        self._refresh_credentials()
        return self._arun_experimental_search(intent_key, query, metadata, retry + 1)
      else:
        msg = str(e)
        logger.warning(msg)
        self.errors.append(msg)
        return None

  async def _arun_metadata_search(self,
                           intent_key:TAllowedIntents,
                           query:str,
                           metadata:MetadataConfig) -> None:
    """Construct SQL from a query and run it against a table
    Append results to self._results_cache
    """
    logger.info(f"Running metadata search for {intent_key}")

    project, dataset, _table = metadata['table_name'].split('.')

    if self.structured_approach == 'endpoint':
      sql = await self._aconvert_query_to_sql_endpoint(query, f"{project}.{dataset}", table_filter=[metadata['table_name']]) # Endpoint function
    elif self.structured_approach == 'dynamic':
      sql = await self._aconvert_query_to_sql_dynamic(query, metadata, temperature=0) # Alveiro's version
    else:
      sql = await self._convert_query_to_sql(query, metadata, temperature=0) # Manual function
    if sql == '':
        msg = ("Error converting query to SQL")
        raise AgentLLMException(msg)
    processed_results = []
    row_ids = []
    try:
      logger.info(f"SQL:\n{sql}")
      results = await self._aquery_bigquery(sql)
    except AgentSQLException as e:
      self.errors.append(str(e))
      logger.error(e)
      results = None
    if results is None or results.total_rows == 0:
      logger.warning(f"No results returned from BQ table {metadata['table_name']}")
      return None
    logger.info(f"{results.total_rows:,} results from BQ table {metadata['table_name']}")
    idx = 1
    id_warning = False
    for row in results:
      if hasattr(row, 'id'):
        row_ids.append(row.id)
      else:
        if not id_warning:
          logger.info("No ID column in results. Cannot filter documents using metadata.")
          id_warning = True
      processed_results.append(row)
      idx += 1
      if idx == self.max_query_rows:
        break

    self.result_cache[intent_key]["results"]["metadata"] = processed_results

  async def _arun_document_search(self,
                           intent_key:TAllowedIntents,
                           query: str,
                           documents:DocumentConfig) -> None:
    logger.info(f"Running Document search against datastore `{documents.get('datastore_id')}`")
    include_summary = documents.get('datastore_type') in ['unstructured', 'website']
    search_response = await self._amake_vertex_search_call(query, documents, include_summary)
    
    if search_response.total_size == 0:
      logger.warning(f"No results returned for datastore `{documents.get('datastore_id')}`")
    else:
      logger.info(f"{search_response.total_size} results returned")
    self.result_cache[intent_key]["results"]["documents"] = search_response.results
    link_list = [result.document.derived_struct_data.get("link") for result in search_response.results if result.document and result.document.derived_struct_data]
    link_dict = {i + 1: link for i, link in enumerate(link_list[:10])} #slicing to keep top 10 result    
    if include_summary:
      self.result_cache[intent_key]["summary"] = search_response.summary.summary_text
      self.result_cache[intent_key]["citation_dict"] = link_dict

  def _calc_score(self, index, confidence):
    """Calculate a new ranking score based on a result index and confidence score"""
    # This is a very naive implementation - scores are not normalised
    # and this is not thoroughly tested
    # For v1 we rank based on (1/(index + 1) * confidence)
    # Assumption is index is zero indexed and a high index = a higher result
    return ((1 / (index + 1)) * confidence)

  def _blend_results(self, return_dict:bool=False) -> List[Union[Document, Dict]]:
    """Rerank and convert raw results to documents"""
    blended_results = []
    for k, v in self.result_cache.items():
      confidence = self.intent_cache.get(k)
      confidence = 0 if confidence is None else confidence[2]
      document_results = self._convert_unstructured_search_response(
        results=v["results"]["documents"],
        collection=k,
        return_dict=return_dict
        )
      metadata_results = self._convert_metadata_response(
        results=v["results"]["metadata"],
        collection=k,
        return_dict=return_dict
        )
      # TO DO - how  to sort / compare documents vs metadata results for the same source
      for idx, d in enumerate(document_results):
        score = self._calc_score(idx, confidence)
        blended_results.append((score, d))
        if idx + 1 == self.max_results_per_query:
          break
      for idx, m in enumerate(metadata_results):
        score = self._calc_score(idx, confidence)
        blended_results.append((score, m))
        if idx + 1 == self.max_results_per_query:
          break  
    return sorted(blended_results, key=lambda x: x[0], reverse=True)

  def _convert_metadata_response(self, results, collection:str, return_dict:bool=False) -> List[Union[Document, Dict]]:
    """Convert array of BQ results into langchain Documents"""
    documents = []
    source = ''
    for r in results:
      content = '\n'.join([f"{k}: {v}" for k, v in r.items()])
      for k,v in r.items():
        if k in ["url", "profile_url"]:
          source = v
        elif k in ["modified_upi"]:
          source = f"https://intranet.worldbank.org/people/profile/{v}"
      if source: 
        metadata = {"wb_collection": collection, "wb_source": "bigquery", "source": source}
      else:
        metadata = {"wb_collection": collection, "wb_source": "bigquery", "source":"Big Query"}
      if return_dict:
        out = {"page_content": content, "metadata": metadata}
      else:
        out = Document(page_content=content, metadata=metadata)
      documents.append(out)
    return documents

# Taken from public langchain class https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/retrievers/google_vertex_ai_search.py
  def _convert_unstructured_search_response(self,
                                            results,
                                            collection:str, 
                                            chunk_type: str='extractive_answers',
                                            return_dict:bool=False) -> List[Union[Document, Dict]]:
        """Converts a sequence of search results to a list of LangChain documents."""
        documents: List[Document] = []

        for result in results:
            document_dict = MessageToDict(
                result.document._pb, preserving_proto_field_name=True
            )
            derived_struct_data = document_dict.get("derived_struct_data")
            # logger.info(f"Derived struct data: {derived_struct_data}")
            if not derived_struct_data:
                continue

            doc_metadata = document_dict.get("struct_data", {})
            doc_metadata["id"] = document_dict["id"]
            doc_metadata["wb_collection"] = collection
            doc_metadata["wb_source"] = "vertex search"

            if chunk_type not in derived_struct_data:
                continue

            for chunk in derived_struct_data[chunk_type]:
                doc_metadata["source"] = derived_struct_data.get("link", "")
                if chunk_type == "extractive_answers":
                    doc_metadata["source"] += f":{chunk.get('pageNumber', '')}"
                
                if return_dict:
                  out = {"page_content": chunk.get("content", ""), "metadata": doc_metadata}
                else:
                  out = Document(page_content=chunk.get("content", ""), metadata=doc_metadata)
                documents.append(out)

        return documents

  def get_relevant_documents(self, query: str, return_dict:bool=False) -> List[Union[Document, Dict]]:
    """Retriever implementation returning a list of Documents"""
    self._reset_cache()
    asyncio.run(self._afetch_results_all(query))
    return self._blend_results(return_dict=return_dict)

  def _expand_query(self, query: str, intent_response: dict):
    try: 
      detected_synonym = intent_response["matches"][0]["parameters"]["project_entity"]
      p = re.compile("\\b[a-zA-Z]")
      term_in_phrase = detected_synonym if re.search(detected_synonym, query) else "".join([ m1.group(0) for m1 in re.finditer(p,detected_synonym)])
      return query.replace(term_in_phrase, f"{detected_synonym} -{''.join([ m1.group(0) for m1 in re.finditer(p,detected_synonym)])}-")
    except Exception as e:
      return query 
    
  def _detect_intents_for_query(self,
                               query: str,
                               session_id:str=None):
    """Returns the result of detect intent with query as inputs.
    Using the same `session_id` between requests allows continuation
    of the conversation."""
    logger.info("Detecting intents")
    session_id = uuid.uuid4() if session_id is None else session_id
    session = self.df_client.session_path(project=self.project,
                                          session=session_id,
                                          location='global',
                                          agent=self.dialogflow_agent)
    text = dialogflow.TextInput(text=query)
    query_input = dialogflow.QueryInput(text=text, language_code='en-US')
    res = self.df_client.match_intent(request={"session": session, "query_input": query_input})
    expanded_query = self._expand_query(query, MessageToDict(res._pb, preserving_proto_field_name=True))
    self.intent_cache = {i.intent.display_name: (i.match_type, i.intent.display_name, i.confidence) for i in res.matches}
    if len(self.intent_cache) == 1 and '' in self.intent_cache: # NO_MATCH intents do not have display names. Ref https://cloud.google.com/dialogflow/cx/docs/reference/rest/v3/Match
      # self.intent_cache[''] = (dialogflow.Match.MatchType.INTENT, 'default', self.intent_cache.get('')[2])
      logger.info(f"Routing to default intent")
      hardcoded_intent = {
        'people':(dialogflow.Match.MatchType.INTENT, 'people', 0.63),
        'projects': (dialogflow.Match.MatchType.INTENT, 'projects', 0.66),
        'documents': (dialogflow.Match.MatchType.INTENT, 'documents', 0.67)}
      self.intent_cache = hardcoded_intent
      # self._query_bigquery(f"INSERT INTO  `rajat-demo-354311.worldbank_sql.no_matched_intents` VALUES ('{expanded_query}');")
      msg = "Query did not match an intent"
      self.errors.append(msg)
    logger.info(f"Intents: {self.intent_cache}")
    return expanded_query

  def _identify_sql_filters(self, query: str, table_data: MetadataConfig, **llm_kwargs):
      logger.info("Identify filters in SQL")
      prompt = f"""
      Given the table {table_data['table_name']} that contains the fields {','.join(table_data['columns'])}.
      Identify the string fields that need to be used as a filter in a SQL query to resolve the next user question (Ignore numeric fields):
      {query}
      Return the identified fields separated by comma, for example "countryname, regionname, theme1".
      Do not include fields that are not part of the field list provided before. For example, return 'countryname' instead of just 'country'
      Don't explain the answer, just return the identified fields.
    """
      logger.info(f"Using prompt\n{prompt}")
      # NOTE: code-bison still producing better responses than Gemini
      response = self.code_model.predict(prefix=prompt, **self._make_llm_params(**llm_kwargs))
      # response = self.llm.generate_content(contents=prompt, generation_config=self._make_llm_params(**llm_kwargs))
      return [r.strip() for r in response.text.split(',') if r is not None]

  async def _aextract_dimension_values(self,
                                field: str,
                                table_data: MetadataConfig,
                                row_limit:int=200,
                                max_char_len:int=200,
                                max_allowed_tokens:int=(2048 - 200) * 4):
    sql = f""" SELECT DISTINCT {field} AS value FROM `{table_data['table_name']}` LIMIT {row_limit} """
    results = await self._aquery_bigquery(sql)
    outp = []
    total_len = 0
    for row in results:
      val = row.value
      if val is not None:
        if (len(val) + total_len) >= max_allowed_tokens: # Do not allow total length of string to exceed a max
          break
        if len(val) <= max_char_len and val.strip() != '': # Do not allow any particularly long values from filters
          outp.append(val)
          total_len += len(val)
    return outp

  async def _aconvert_query_to_sql_dynamic(self, query: str, table_data: MetadataConfig, **llm_kwargs) -> str:
    """Convert a query to SQL"""
    logger.info("Converting query to SQL using LLM")
    ks = self._identify_sql_filters(query, table_data)
    vs = await asyncio.gather(*[asyncio.create_task(self._aextract_dimension_values(k, table_data)) for k in ks])
    filterable_values = (dict(zip(ks, vs)))

    prompt = f"""Given the table {table_data['table_name']} that contains the fields {','.join(table_data['columns'])}.
Transform the next statement to SQL:
{query}
"""
    prompt += '\n'.join([f"Consider that values available in `{c}` are:\n{','.join(v)}""" for c, v in filterable_values.items()])
    prompt += '\nDo not attempt to join any additional tables, or filter using any values other than those provided here.\n```sql:'

    # NOTE: code-bison still producing better responses than Gemini
    response = self.code_model.predict(prefix=prompt, **self._make_llm_params(**llm_kwargs))
    # response = self.llm.generate_content(contents=prompt, generation_config=self._make_llm_params(**llm_kwargs))
    return response.text.replace("```sql", "").replace("```", "")

  async def _convert_query_to_sql(self, query: str, table_data: MetadataConfig, **llm_kwargs) -> str:
    """Convert a query to SQL"""
    logger.info("Converting query to SQL")
    prompt = f"""Given the table {table_data['table_name']} that contains the fields {','.join(table_data['columns'])}.
Transform the next statement to SQL:
{query}
"""
    prompt += '\n'.join([f"""Consider that values available in `{c}` are:\n"{'","'.join(v)}\"""" for c, v in table_data['filter_values'].items()])
    prompt += '\nLimit the results to 30. Do not attempt to join any additional tables, or filter using any values other than those provided here.\n```sql:'

    # NOTE: code-bison still producing better responses than Gemini
    response = self.code_model.predict(prefix=prompt, **self._make_llm_params(**llm_kwargs))
    # response = self.llm.generate_content(contents=prompt, generation_config=self._make_llm_params(**llm_kwargs))
    return response.text.replace("```sql", "").replace("```", "").replace("''", "|").replace("'", "\"").replace("|","'")

  async def _aconvert_query_to_sql_endpoint(self,
                                query:str,
                                dataset: str,
                                table_filter: Optional[List[str]] = [],
                                column_filter: Optional[Dict[str, List[str]]] = {},
                                nb_samples: int = 2,
                                temperature: float = 0.8) -> str:
      logger.info("Converting query to SQL")
      llm_params = {"temperature":temperature, "nb_samples": nb_samples}

      example_input = {"text": query, "bq_dataset": dataset, "project_id": self.project}
      if table_filter:
        example_input["bq_tables_filter"] = table_filter
      if column_filter:
        example_input["bq_columns_filter"] = column_filter

      endpoint = aiplatform.Endpoint(endpoint_name=f"projects/{self.project}/locations/{TEXT2SQL_LOCATION}/endpoints/{TEXT2SQL_ENDPOINT}")
      res = endpoint.predict(instances=[example_input], parameters=llm_params)
      sql_query = res.predictions[0]['sql_query']
      clean_q = sql_query.replace('\n', '')
      logger.info(f"Query: {clean_q}")
      return sql_query

  async def _amake_vertex_search_call(self,
                               query:str,
                               config:DocumentConfig,
                               include_summary:bool=True):
    """Return a vertex search call with summary"""
    logger.info(f"Querying Vertex engine {config.get('datastore_id')}")
    st=timeit.default_timer()
    request = discoveryengine_v1.SearchRequest(
        serving_config=self._make_serving_config(config),
        query=query,
        page_size=self.max_results_per_query,
        content_search_spec=self.content_search_spec if include_summary else None
    )
    results=self.search_client.search(request)
    # logger.info(f"Search response metada: {results.summary.summary_with_metadata}")
    if self.timestamps.get('vertex_search') is None:
      self.timestamps['vertex_search'] = 0
    self.timestamps['vertex_search'] += timeit.default_timer()-st
    return results

  async def _aquery_bigquery(self, sql: str):
    """Return results of a bigquery query"""
    logger.info("Querying BQ")
    try:
      r = self.bq_client.query(sql, project=self.project)
      return r.result()
    # Catch and reraise as a custom exception type
    except Exception as e:
      msg = f"Error querying BQ: {e}"
      raise AgentSQLException(msg)

  def _query_bigquery(self, sql: str):
    """Return results of a bigquery query"""
    logger.info("Querying BQ")
    try:
      r = self.bq_client.query(sql, project=self.project)
      return r.result()
    # Catch and reraise as a custom exception type
    except Exception as e:
      msg = f"Error querying BQ: {e}"
      raise AgentSQLException(msg)


if __name__ == '__main__':
  retriever = CustomGoogleVertexAISearchRetriever()
