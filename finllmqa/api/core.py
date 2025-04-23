from dotenv import load_dotenv
import os

load_dotenv()
LOCAL_HOST = os.environ.get('LOCAL_HOST', '')
if LOCAL_HOST == '':
    raise ValueError("LOCAL_HOST environment variable is not set. Please set it to your local host address.")
# change local and server url if needed
LOCAL_API_URL = 'http://' + LOCAL_HOST
SERVER_API_URL = os.environ.get('SERVER_API_URL', '')
if SERVER_API_URL == '':
    raise ValueError("SERVER_API_URL environment variable is not set. Please set it to your server address.")

# port
NEO4J_API_PORT = ':7687'
NEO4J_API_PORT_BACKUP = ':7474'
MILVUS_API_PORT = '19530'
LLM_API_PORT = ':8000'

# url
NEO4J_API_URL = 'bolt://' + LOCAL_HOST + NEO4J_API_PORT
NEO4J_API_URL_BACKUP = LOCAL_API_URL + NEO4J_API_PORT_BACKUP
LLM_API_URL = SERVER_API_URL + '/v1'
EMBEDDING_API_URL = LLM_API_URL + '/embeddings'
CHAT_API_URL = LLM_API_URL + '/chat/completions'

# login info
STOCK_KG_USER = os.environ.get('STOCK_KG_USER', 'neo4j')
STOCK_KG_PW = os.environ.get('STOCK_KG_PW', 'neo4j')
