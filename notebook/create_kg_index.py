import os
import logging
import sys

os.environ['NEBULA_USER'] = "root"
os.environ['NEBULA_PASSWORD'] = "nebula" # default password
os.environ['NEBULA_ADDRESS'] = "192.168.30.158:9669" 

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

llm = OpenAI(model="gpt-3.5-turbo", api_base=LLM_API_URL, api_key='null')
embed_model = OpenAIEmbedding(api_base=LLM_API_URL, api_key='null')

Settings.llm = llm
Settings.embed_model = embed_model

# change path to where you save the teaching resources
document_path = 'books/'
file_name_ls = ['微观经济学.pdf']
file_name_ls = [document_path + file_name for file_name in file_name_ls]

reader = SimpleDirectoryReader(input_files=file_name_ls)
documents = reader.load_data()

from llama_index.core.node_parser import SentenceSplitter


chunk_size_ls = [256, 512, 1024]
chunk_overlap_pct_ls = [1/8, 1/4]
split_document_dc = {}
for chunk_size in chunk_size_ls:
    for chunk_overlap_pct in chunk_overlap_pct_ls:
        if chunk_size == 256 and chunk_overlap_pct == 1/8:
            continue
        chunk_overlap = int(chunk_size * chunk_overlap_pct)
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_document = splitter.get_nodes_from_documents(documents=documents)
        split_document_dc[f'size_{chunk_size}_overlap_{chunk_overlap}'] = split_document
        print(f'chunk_size: {chunk_size}; chunk_overlap: {chunk_overlap} len_chunks: {len(split_document)}')
print(split_document_dc.keys())
from threading import Thread

def create_and_store_kg_index(nodes_group, nodes):
    kg_extract_template = """
    下面提供了一些文本。根据文本，提取最多 {max_knowledge_triplets} 个三元组的知识，形式为(实体,关系,实体)，具体可以是(主语,谓语,宾语)或者其他类型，注意避开停用词。
    请忽略page_label和file_path
    ---------------------
    示例：
    文本：小红是小明的母亲.
    三元组：
    (小红,是母亲,小明)
    文本:瑞幸是2017年在厦门创立的咖啡店。
    三元组：
    (瑞幸,是,咖啡店)
    (瑞幸,创立于,厦门)
    (瑞幸,创立于,2017)
    文本:在长期中，物价总水平会调整到使货币需求等于货币供给的水平。
    三元组：
    (物价总水平,长期调整使等于,货币需求等于货币供给的水平)
    ---------------------
    文本：{text}
    三元组："""
    kg_extract_template = PromptTemplate(kg_extract_template)

    print(f'\n\nstart extract {nodes_group} nodes...\n\n')
    space_name = f"books_content_{nodes_group}"
    edge_types, rel_prop_names = ["relationship"], ["relationship"] # default, could be omit if create from an empty kg
    tags = ["entity"] # default, could be omit if create from an empty kg

    graph_store = NebulaGraphStore(
        space_name=space_name,
        edge_types=edge_types,
        rel_prop_names=rel_prop_names,
        tags=tags,
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
    kg_index_ls.append(kg_index)

    # store index
    kg_index.storage_context.persist(persist_dir=f'../index/storage_graph/{nodes_group}')

for nodes_group, nodes in split_document_dc.items():
    thread = Thread(target=create_and_store_kg_index, args=(nodes_group, nodes))
    thread.start()
