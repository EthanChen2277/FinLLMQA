import json
import os
from pathlib import Path
from pymilvus import connections, FieldSchema, DataType, CollectionSchema, Collection, MilvusClient
from pymilvus.orm import utility
from typing import List
from tqdm import tqdm

from api.embedding import get_embedding
from api.core import LOCAL_HOST, MILVUS_API_PORT


class Milvus():
    def __init__(self,
                 collection_name: str = 'stock'):
        self._vector_field = 'vector'
        self.embeddings_func = get_embedding
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
            elif collection_name == 'attributes':
                self.collection = ''
            else:
                raise NotImplementedError

    def _create_stock_collection(self):

        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="code", dtype=DataType.VARCHAR, max_length=65_535),
            FieldSchema(name='name', dtype=DataType.VARCHAR, max_length=65_535),
            FieldSchema(name=self._vector_field, dtype=DataType.FLOAT_VECTOR, dim=384)
        ]
        # 创建Collection
        schema = CollectionSchema(fields, "stock")
        self.collection = Collection("stock", schema, consistency_level='Strong')
        self.create_stock_index()

        self.collection.load()
        return self.collection

    def create_stock_index(self):
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
            record[self._vector_field] = self.embeddings_func(record[embedding_field])
            new_data.append(record)
        return new_data
    
    def insert_json_data(self, data: List[dict], embedding_field: str = 'name'):
        data = self.process_json_data(data=data, embedding_field=embedding_field)
        self._insert_data(data=data)



if __name__ == "__main__":
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    with open(Path(parent_dir, "kg/data/stock.json"), "r") as file:
        stock_name_json = json.load(file)
    print(stock_name_json)
