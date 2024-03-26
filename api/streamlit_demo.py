import streamlit as st
import asyncio
import requests
import json
from autogen import AssistantAgent, UserProxyAgent

st.set_page_config(
    page_title="AutoGen Chat Agents",
    page_icon=":robot:",
    layout="wide"
)


selected_model = None
selected_key = None
with st.sidebar:
    st.header("OpenAI Configuration")
    selected_model = st.selectbox("Model", ['chatglm3-6B'], index=0)

with st.chat_message(name="user", avatar="user"):
    input_placeholder = st.empty()
user_input = st.chat_input("Type something...")
if user_input:
    input_placeholder.markdown(user_input)
    if not selected_model:
        st.warning(
            '请先选择模型！', icon="⚠️")
        st.stop()

    llm_config = {
        "config_list": [
            {
                "model": selected_model,
                "base_url": "http://localhost:8000/v1",
                "api_type": "open_ai",
                "api_key": 'NULL'
            }
        ]
    }

    # # Run the asynchronous function within the event loop
    # loop.run_until_complete(initiate_chat())
    display_num = 0
    response = {"stop": False}
    while not response['stop']:
        res = requests.post(url='http://localhost:8006/stream',
                            json={'prompt': user_input}).content
        response = json.loads(res)
        answer_list = response['answer']
        answer_num = len(answer_list)
        if answer_num == 0:
            continue
        answer = answer_list[-1]
        if answer_num > display_num:
            with st.chat_message(name=answer['name'], avatar='assistant'):
                message_placeholder = st.empty()
            display_num = answer_num
        message_placeholder.markdown(answer['response'])
