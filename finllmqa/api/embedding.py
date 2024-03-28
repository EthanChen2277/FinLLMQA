import requests

from finllmqa.api.core import EMBEDDING_URL


def embedding_request(text: str):
    body = {
        'input': text
    }
    url = EMBEDDING_URL
    res = requests.post(url, json=body)
    if res.status_code == 200:
        json = res.json()
        return json
    else:
        return None


def get_embedding(text: str):
    response = embedding_request(text=text)
    if response is None:
        return None
    embed = response['data'][0]['embedding']
    return embed


def get_tokens(text: str):
    response = embedding_request(text=text)
    if response is None:
        return None
    tokens = response['usage']['total_tokens']
    return tokens
