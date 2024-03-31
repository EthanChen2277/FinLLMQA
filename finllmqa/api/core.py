# change local and server url if needed 
LOCAL_HOST = '192.168.44.1'
LOCAL_API_URL = 'http://' + LOCAL_HOST
SERVER_API_URL = 'http://gemini2.sufe.edu.cn'

# port
NEO4J_API_PORT = ':7474'
MILVUS_API_PORT = '19530'
LLM_API_PORT = ':8000'
STREAM_API_PORT = ':8006'

# server port
SERVER_LLM_API_PORT = ':27282'

# url
NEO4J_API_URL = LOCAL_API_URL + NEO4J_API_PORT
LLM_API_URL = SERVER_API_URL + SERVER_LLM_API_PORT + '/v1'
EMBEDDING_API_URL = LLM_API_URL + '/embeddings'
CHAT_API_URL = LLM_API_URL + '/chat/completions'
STREAM_API_URL = LOCAL_API_URL + STREAM_API_PORT + '/autogen/stream'

# login info
STOCK_KG_USER = 'neo4j'
STOCK_KG_PW = 'finglm-base-on-kg'
