from pymilvus import FieldSchema, DataType, CollectionSchema, Collection, MilvusClient
from pymilvus.orm import utility

from api.embedding import embedding
from api.core import LOCAL_API_URL, MILVUS_API_PORT


class Milvus():
    def __init__(self):
        self._vector_field = 'vector'
        self.embeddings_func = embedding()
        self.connect()

    def connect(self, collection_name: str = 'stock_info'):
        self.client = MilvusClient(
            uri=LOCAL_API_URL + MILVUS_API_PORT,
            collection_name=collection_name,
        )
        self.alias = self.client.alias
        if utility.has_collection(collection_name, using=self.alias):
            self.collection = Collection(
                collection_name,
                using=self.alias,
            )
        else:
            self.collection = self._create_stock_collection()

    def _create_stock_collection(self):

        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="code", dtype=DataType.VARCHAR, max_length=65_535),
            FieldSchema(name='name', dtype=DataType.VARCHAR, max_length=65_535),
            FieldSchema(name=self._vector_field, dtype=DataType.FLOAT_VECTOR, dim=1536)
        ]
        # 创建Collection
        schema = CollectionSchema(fields, "stock_info")
        self.col = Collection("stock_info", schema, using=self.alias, consistency_level='Strong')
        self.create_stock_index()

        self.col.load()
        return self.col

    def create_stock_index(self):
        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64},
        }

        self.col.create_index(
            self._vector_field,
            index_params=index_params,
            using=self.alias,
        )

    #
    def _insert(self, data):
        self.col.insert(data)

    def insert_stock_info(self, data):
        self.col.insert(data)



if __name__ == "__main__":
    a = Milvus()
    a.connect()
    # a.insert_stock_info()
