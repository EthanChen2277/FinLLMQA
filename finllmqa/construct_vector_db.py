import json
import os
from pathlib import Path

from finllmqa.vector_db.construct.core import *

if __name__ == '__main__':

    current_dir = os.path.dirname(os.path.abspath(__file__))

    with open(Path(current_dir, "kg/data/stock.json"), "r") as file:
        stock_name_json = json.load(file)
    Milvus('stock').insert_json_data(data=stock_name_json,
                                     embedding_field='name')
    
    with open(Path(current_dir, "kg/data/attributes.json"), "r") as file:
        attributes_json = json.load(file)
    Milvus('attribute').insert_json_data(data=attributes_json,
                                         embedding_field='name')
