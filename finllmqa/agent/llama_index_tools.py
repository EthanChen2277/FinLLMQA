from abc import ABC
from pathlib import Path

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from finllmqa.api.core import LLM_API_URL


class LLamaIndexTool(ABC):
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
        self._init_query_settings()

    def _init_query_settings(self):
        # NOTE: lazy import
        from llama_index.core import Settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

    def get_storage_context(self, cache_folder_path: Path):
        pass



class KGQueryTool(LLamaIndexTool):
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



