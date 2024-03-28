from pymilvus import connections, FieldSchema, DataType, CollectionSchema, Collection
from pymilvus.orm import utility
from typing import List
from tqdm import tqdm

from finllmqa.api.embedding import get_embedding
from finllmqa.api.core import LOCAL_HOST, MILVUS_API_PORT


class Milvus:
    def __init__(self,
                 collection_name: str = 'stock'):
        self._vector_field = 'vector'
        self.embeddings_func = get_embedding
        self.embed_dim = 384
        self.connect(collection_name=collection_name)

    def connect(self, collection_name: str = 'stock'):
        connections.connect(host=LOCAL_HOST, port=MILVUS_API_PORT)
        if utility.has_collection(collection_name):
            self.collection = Collection(
                collection_name
            )
        else:
            if collection_name == 'stock':
                self.collection = self._create_stock_collection()
            elif collection_name == 'attribute':
                self.collection = self._create_attribute_collection()
            else:
                raise NotImplementedError

    def _create_stock_collection(self):

        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="code", dtype=DataType.VARCHAR, max_length=65_535),
            FieldSchema(name='name', dtype=DataType.VARCHAR, max_length=65_535),
            FieldSchema(name=self._vector_field, dtype=DataType.FLOAT_VECTOR, dim=self.embed_dim)
        ]
        # 创建Collection
        schema = CollectionSchema(fields, "stock")
        self.collection = Collection("stock", schema, consistency_level='Strong')
        self._create_index()

        self.collection.load()
        return self.collection

    def _create_attribute_collection(self):

        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name='name', dtype=DataType.VARCHAR, max_length=65_535),
            FieldSchema(name=self._vector_field, dtype=DataType.FLOAT_VECTOR, dim=self.embed_dim)
        ]
        # 创建Collection
        schema = CollectionSchema(fields, "attribute")
        self.collection = Collection("attribute", schema, consistency_level='Strong')
        self._create_index()

        self.collection.load()
        return self.collection

    def _create_index(self):
        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64},
        }

        self.collection.create_index(
            self._vector_field,
            index_params=index_params
        )

    def _insert_data(self, data):
        self.collection.insert(data)

    def process_json_data(self, data: List[dict], embedding_field: str = 'name'):
        new_data = []
        for record in tqdm(data):
            embed = self.embeddings_func(record[embedding_field])
            assert len(embed) == self.embed_dim, f'{record[embedding_field]}embedding dim is not correct!'
            record[self._vector_field] = embed
            new_data.append(record)
        return new_data
    
    def insert_json_data(self, data: List[dict], embedding_field: str = 'name'):
        data = self.process_json_data(data=data, embedding_field=embedding_field)
        self._insert_data(data=data)
