import os
import logging
import sys 
from tqdm import tqdm

os.environ['NEBULA_USER'] = "root"
os.environ['NEBULA_PASSWORD'] = "nebula" # default password
os.environ['NEBULA_ADDRESS'] = "127.0.0.1:9669" 

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output

from llama_index.core import (
    KnowledgeGraphIndex,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    PromptTemplate)
from llama_index.graph_stores.nebula import NebulaGraphStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from finllmqa.api.core import LLM_API_URL
from llama_index.core import Settings

llm = OpenAI(model="gpt-3.5-turbo", api_base='http://gemini2.sufe.edu.cn:47489/v1', api_key='null', timeout=300)
embed_model = OpenAIEmbedding(api_base='http://gemini2.sufe.edu.cn:47489/v1', api_key='null', timeout=300)

Settings.llm = llm
Settings.embed_model = embed_model

# change path to where you save the teaching resources
document_path = 'books/'
file_name_ls = os.listdir(document_path)
file_ls = [document_path + file_name for file_name in file_name_ls]

reader = SimpleDirectoryReader(input_files=file_ls)
documents = reader.load_data()

from llama_index.core.node_parser import SentenceSplitter

useless_text = ['', '', '']
# chunk_size = 512
# chunk_overlap = 64

# splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
# kg_nodes = splitter.get_nodes_from_documents(documents=documents)
# for node in kg_nodes:
#     node.text = node.text.replace(useless_text[0], '').replace(useless_text[1], '').replace(useless_text[2], '')

chunk_size = 256
chunk_overlap = 16

splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
vec_nodes = splitter.get_nodes_from_documents(documents=documents)
for node in vec_nodes:
    node.text = node.text.replace(useless_text[0], '').replace(useless_text[1], '').replace(useless_text[2], '')

from threading import Thread

def create_and_store_kg_index(nodes_group, nodes):
    kg_extract_template = """
    下面提供了一些文本。根据文本，提取不超过{max_knowledge_triplets} 个三元组的知识，形式为(实体,关系,实体)，具体可以是(主语,谓语,宾语)或者其他类型，注意避开停用词。
    请忽略page_label和file_path;
    提取的实体长度不能超过10个字, 控制在5个字左右;
    提取的三元组不要重复;
    如果你无法抽取出有效的三元组请回复'无有效三元组'
    举例：
    ---------------------
    请根据文本抽取三元组：
    文本:

    page_label: 0
    file_path: 'a\\b.pdf'

    小红是小明的母亲.

    三元组：
    (小红,是母亲,小明)
    ---------------------
    请根据文本抽取三元组：
    文本:
    
    page_label: 0
    file_path: 'a\\b.pdf'

    瑞幸是2017年在厦门创立的咖啡店。

    三元组：
    (瑞幸,是,咖啡店)
    (瑞幸,创立于,厦门)
    (瑞幸,创立于,2017)
    ---------------------
    请根据文本抽取三元组：
    文本:

    {text}

    三元组："""
    kg_extract_template = PromptTemplate(kg_extract_template)

    print(f'\n\nstart extract {nodes_group} nodes...\n\n')
    space_name = f"book_{nodes_group}"
    edge_types, rel_prop_names = ["关系"], ["关系"] # default, could be omit if create from an empty kg
    tags = ["实体"] # default, could be omit if create from an empty kg

    graph_store = NebulaGraphStore(
        space_name=space_name,
        edge_types=edge_types,
        rel_prop_names=rel_prop_names,
        tags=tags
    )
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    kg_index = KnowledgeGraphIndex(
        nodes=nodes,
        storage_context=storage_context,
        max_triplets_per_chunk=10,
        space_name=space_name,
        edge_types=edge_types,
        rel_prop_names=rel_prop_names,
        tags=tags,
        include_embeddings=True,
        kg_triple_extract_template=kg_extract_template
    )

    # store index
    kg_index.storage_context.persist(persist_dir=f'../storage/storage_graph/{nodes_group}')

def create_and_store_vector_index(nodes_group, nodes):
    vector_index = VectorStoreIndex(nodes=nodes)
    vector_index.storage_context.persist(persist_dir=f'../storage/storage_vector/{nodes_group}')

# thread = Thread(target=create_and_store_kg_index, args=('all', kg_nodes))
# thread.start()

thread = Thread(target=create_and_store_vector_index, args=('all', vec_nodes))
thread.start()
