import os
from abc import ABC
from typing import Optional, List

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.indices.knowledge_graph.retrievers import REL_TEXT_LIMIT, KnowledgeGraphRAGRetriever
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.graph_stores.nebula import NebulaGraphStore
from llama_index.legacy.indices.knowledge_graph import KGTableRetriever
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import StorageContext, load_index_from_storage, QueryBundle, \
    get_response_synthesizer

from finllmqa.api.core import LLM_API_URL


class LlamaIndexTool(ABC):
    def __init__(self,
                 llm: BaseLLM = None,
                 embed_model: BaseEmbedding = None,
                 verbose: bool = True,
                 *args,
                 **kwargs):
        if llm is None:
            self.llm = OpenAI(api_base=LLM_API_URL, api_key='null')
        else:
            self.llm = llm
        if embed_model is None:
            self.embed_model = OpenAIEmbedding(api_base=LLM_API_URL, api_key='null')
        else:
            self.embed_model = embed_model
        self.index = None
        self.retriever = None
        self.query_engine = None
        self._init_query_settings()

    def _init_query_settings(self):
        # NOTE: lazy import
        from llama_index.core import Settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

    def query(self, text):
        assert self.query_engine is not None, 'Query Engine has not been initialize by subclass yet'
        response = self.query_engine.query(text)
        return response.text


class KGRetrieverTool(LlamaIndexTool):
    def __init__(self,
                 llm: BaseLLM = None,
                 embed_model: BaseEmbedding = None,
                 verbose: bool = True,
                 *args,
                 **kwargs):
        super().__init__(llm=llm,
                         embed_model=embed_model,
                         verbose=verbose,
                         *args,
                         **kwargs)
        self.get_retriever()

    def load_index_from_storage(self, persist_folder: str = '../../../storage/storage_graph',
                                nodes_group: str = 'all'):
        persist_dir = persist_folder + nodes_group
        assert os.path.exists(persist_dir)
        space_name = f"{nodes_group}"
        edge_types, rel_prop_names = ["relationship"], ["relationship"]
        tags = ["entity"]

        graph_store = NebulaGraphStore(
            space_name=space_name,
            edge_types=edge_types,
            rel_prop_names=rel_prop_names,
            tags=tags,
        )
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir,
                                                       graph_store=graph_store)
        self.index = load_index_from_storage(
            storage_context=storage_context,
            space_name=space_name,
            edge_types=edge_types,
            rel_prop_names=rel_prop_names,
            tags=tags,
            include_embeddings=True
        )

    def get_retriever(self,
                      max_entities: int = 5,
                      max_synonyms: int = 5,
                      retriever_mode: Optional[str] = "keyword",
                      with_nl2graphquery: bool = False,
                      graph_traversal_depth: int = 2,
                      max_knowledge_sequence: int = REL_TEXT_LIMIT):
        self.load_index_from_storage()
        self.retriever = KGTableRetriever(index=self.index,
                                          max_entities=max_entities,
                                          max_synonyms=max_synonyms,
                                          retriever_mode=retriever_mode,
                                          with_nl2graphquery=with_nl2graphquery,
                                          graph_traversal_depth=graph_traversal_depth,
                                          max_knowledge_sequence=max_knowledge_sequence)


class VectorRetrieverTool(LlamaIndexTool):
    def __init__(self,
                 llm: BaseLLM = None,
                 embed_model: BaseEmbedding = None,
                 verbose: bool = True,
                 *args,
                 **kwargs):
        super().__init__(llm=llm,
                         embed_model=embed_model,
                         verbose=verbose,
                         *args,
                         **kwargs)
        self.get_retriever()

    def load_index_from_storage(self, persist_folder: str = '../../../storage/storage_vector',
                                nodes_group: str = 'all'):
        persist_dir = persist_folder + nodes_group
        assert os.path.exists(persist_dir)
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        self.index = load_index_from_storage(storage_context=storage_context)

    def get_retriever(self,
                      similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
                      vector_store_query_mode: VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT):
        self.load_index_from_storage()
        self.retriever = VectorIndexRetriever(index=self.index,
                                              similarity_top_k=similarity_top_k,
                                              vector_store_query_mode=vector_store_query_mode)


class CustomRetrieverTool(LlamaIndexTool, BaseRetriever):
    """Custom retriever that performs both Vector search and Knowledge Graph search"""

    def __init__(self,
                 llm: BaseLLM = None,
                 embed_model: BaseEmbedding = None,
                 verbose: bool = True,
                 mode: str = "OR",
                 *args,
                 **kwargs):
        LlamaIndexTool.__init__(llm=llm,
                                embed_model=embed_model,
                                verbose=verbose,
                                *args,
                                **kwargs)

        self._vector_retriever = VectorRetrieverTool().retriever
        self._kg_retriever = KGRetrieverTool().retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        kg_nodes = self._kg_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        kg_ids = {n.node.node_id for n in kg_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in kg_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(kg_ids)
        else:
            retrieve_ids = vector_ids.union(kg_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes


class RAGQueryEngineTool(LlamaIndexTool):
    def __init__(self,
                 llm: BaseLLM = None,
                 embed_model: BaseEmbedding = None,
                 verbose: bool = True,
                 *args,
                 **kwargs):
        super().__init__(llm=llm,
                         embed_model=embed_model,
                         verbose=verbose,
                         *args,
                         **kwargs)

    def get_retriever(self, mode: str = 'AND'):
        assert mode in ['KG', 'VEC', 'AND', 'OR']
        if mode == 'KG':
            self.retriever = KGRetrieverTool().retriever
        elif mode == 'VEC':
            self.retriever = VectorRetrieverTool().retriever
        else:
            self.retriever = CustomRetrieverTool(mode=mode)

    def run(self, text, response_mode: str = 'compact'):
        response_synthesizer = get_response_synthesizer(
            response_mode=response_mode
        )
        self.query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=response_synthesizer
        )
        response = self.query(text=text)
        return response



