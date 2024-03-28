# change local and server url if needed 
LOCAL_HOST = '192.168.30.115'
LOCAL_API_URL = 'http://' + LOCAL_HOST
SERVER_API_URL = 'http://gemini2.sufe.edu.cn'

# port
MILVUS_API_PORT = '19530'
LLM_API_PORT = ':8000'
STREAM_API_PORT = ':8006'

# server port
SERVER_LLM_API_PORT = ':27282'

# url
LLM_URL = SERVER_API_URL + SERVER_LLM_API_PORT + '/v1'
EMBEDDING_URL = LLM_URL + '/embeddings'
CHAT_URL = LLM_URL + '/chat/completions'
STREAM_URL = LOCAL_API_URL + STREAM_API_PORT + '/autogen/stream'
