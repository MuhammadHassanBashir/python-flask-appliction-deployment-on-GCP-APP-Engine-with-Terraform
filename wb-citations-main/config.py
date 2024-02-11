from typing import Dict, Optional, Literal, TypedDict, List

class LLMParams(TypedDict):
  max_output_tokens: int
  temperature: float
  max_p: Optional[float]
  max_k: Optional[int]


#  Dialogflow intent detection values.
TAllowedIntents = Literal['projects', 'people', 'documents']

# Create some types
class MetadataConfig(TypedDict):
  table_name: str
  columns: List[str]
  filter_values: Dict[str, List[str]]

class DocumentConfig(TypedDict):
  datastore_id: str
  datastore_type: Literal["website", "structured", "unstructured"]
  project: Optional[str]
  location: Optional[str]
  filter_values: Dict[str, List[str]]

class VectorConfig(TypedDict):
  name: str
  id_column: str
  endpoint_unique_id: str
  endpoint_ref: str

class DatastoreConfig(TypedDict):
  sql: Optional[MetadataConfig]
  vector: Optional[VectorConfig]
  documents: Optional[DocumentConfig]
  experimental: Optional[DocumentConfig]

# NOTE that the keys of this table must match dialogflow intents
DATASTORE_CONFIG: Dict[TAllowedIntents, DatastoreConfig] = {
    "projects": {
        # "documents": {
        #     "datastore_id": "wb-unstructured-with-metad_1690319343422",
        #     "datastore_type": "unstructured",
        #     "project": "rajat-demo-354311",
        #     "location": "us-central1"
        # },
        "vector": {
          'name': 'rajat-demo-354311.world_bank.wb_projects',
          'id_column': 'id',
          'endpoint_ref': '5431697392376217600',
          'endpoint_unique_id': 'wb_projects_002_1702039447947'
          },
        "sql": {
          "table_name": 'rajat-demo-354311.worldbank_sql.wb_projects',
          "columns": ["id : string ",
                "regionname : string ",
                "countryname : string  # represents the country, examples: Indonesia, India.",
                "projectstatusdisplay : string : # current status of the project ",
                "last_stage_reached_name : string ",
                "project_name : string ",
                "pdo : string ",
                "impagency : string ",
                "cons_serv_reqd_ind : string ",
                "url : string ",
                "boardapprovaldate : date ",
                "closingdate : date",
                "projectfinancialtype : string",
                "curr_project_cost : numeric",
                "curr_ibrd_commitment : numeric",
                "curr_ida_commitment : numeric",
                "curr_total_commitment : numeric",
                "grantamt : numeric ",
                "borrower : string",
                "lendinginstr : string",
                "envassesmentcategorycode : string",
                "esrc_ovrl_risk_rate: string",
                "sector1 : string",
                "sector2 : string",
                "sector3 : string",
                "theme1 : string",
                "theme2 : string"],
          "filter_values": {}
          }
        },
    "people": {
        # "documents": {
        #     "datastore_id": "wb-unstructured-with-metad_1690319343422",
        #     "datastore_type": "unstructured",
        #     "project": "rajat-demo-354311",
        #     "location": "us-central1"
        # },
        "experimental": {
          "datastore_id": "test_wb_peep-structured",
          "datastore_type": "structured",
          "project": "rajat-demo-354311",
          "location": "us-central1"
        },
        "sql":{
        "table_name": "rajat-demo-354311.world_bank.external_people_v2",
        "columns":[
          "modified_upi : string",
          "lastname : string",
          "firstname : string",
          "nickname : string",
          "fullname : string",
          "businesstitle : string",
          "dutystation : string",
          "jobtitle : string",
          "roomnumber : string",
          "officialunitcode : string",
          "institution : string",
          "email : string",
          "practicename : string",
          "positionmapping : string",
          "vpucode : string",
          "profffamilymap : string",
          "vpu : string",
          "officialunit : string",
          "language : string",
          "specialization : string",
          "skills : string",
          "countryexperience : string",
          "coursesinstitution : string",
          "educationuniversity : string",
          "educationdegreename : string",
          "organization : string",
          "SPECIALASSISTANT : string",
          "UnitSupport : string"
        ],
        "filter_values": {}
      },
        "vector": {
          'name': 'rajat-demo-354311.world_bank.external_people_v2',
          'id_column': 'modified_upi',
          'endpoint_unique_id': 'wb_peoplw_table_v2_1704920146478',
          'endpoint_ref': '4457477313608548352'  
          },
    },
    "documents": {
        "documents": {
            "datastore_id": "wb-unstructured-with-metad_1690319343422",
            "datastore_type": "unstructured",
            "project": "rajat-demo-354311",
            "location": "us-central1"
        },
        "vector": {
          'name': 'rajat-demo-354311.world_bank.lessons_learned',
          'id_column': 'id',
          'endpoint_unique_id': 'wb_lessons_002_1702039350794',
          'endpoint_ref': '466478803200245760'
          },
    },
    "default": {
        "documents": {
            "datastore_id": "wb-unstructured-with-metad_1690319343422",
            "datastore_type": "unstructured",
            "project": "rajat-demo-354311",
            "location": "us-central1"
        }
    }
  }

SAMPLE_QUERIES = [
  ("Lessons learned from projects implemented in Nepal", "project"),
  ("relocation challenges with renewable energy projects in West Africa", "project"),
  # ("list the mitigation strategies that we have used for addressing these (follow-up question to 1)", "project"),
  ("good project to review before starting a maternal health project in Nigeria", "project"),
  (" current status of P12345", "project"),
  ("Total amount disbursed among open education projects in Bangladesh", "project"),
  ("List projects that have had an overall satisfactory rating over the past 5 years in digital development", "project"),
  ("Common lessons learned across projects that have been rated unsatisfactory in south asia", "project"),
  ("What could the key objectives of a new child nutrition project in Vietnam", "project"),
  ("Help with a draft procurement plan", "project"),
  ("suggest someone who can guide on an infrastructure project in sahel region", "people"),
  ("transport specialist in East Asia who can speak Bahasa", "people"),
  ("how to close a project", "project"),
  ("how to improve disbursement in bank projects?", "project"),
  ("how long is disbursement after closing date of project", "project"),
  ("how do we identify at risk projects?", "project"),
  ("how to calculate milestone date of project preparation", "project"),
  ("what are the characteristics of a project objective", "project"),
  ("how to prepare a security management plan for a project", "project"),
  ("what are the possible sources of funding available for a client to prepare a project?", "project"),
  ("how a program for results project has encouraged procurement reforms in vietnam", "project"),
  ("what ttls need to know about the gender tag for statistics projects", "project"),
  ("how to disclose a deliverable before the closure of the project", "project"),
  ("how to access all project for a single country", "project"),
  ("how to know the quantity of ida/ibrd projects approved per year", "project"),
  ("discounting costs and benefits in economic analysis of world bank projects", "project"),
  ("guidelines on preventing and combating fraud and corruption in projects financed by ibrd loans and ida credits and grants", "project"),
  ("implementation and supervision of bank projects in insecure areas including security and third-party monitoring", "project"),
  ("trade facilitation challenges for women traders and freight forwarders", "project"),
  ("what are the relevant policies and procedures for processing the mdtf grants to the ida project", "project")
]
