import streamlit as st
import requests

from finllmqa.api.core import STREAM_API_URL
from finllmqa.agent.langchain_tools import *

llm = ChatGLM3(
    endpoint_url=SERVER_API_URL + SERVER_LLM_API_PORT + CHAT_API_URL,
    max_tokens=8096,
    top_p=0.9
)
tools = [KGRetrieveTool(llm), BaseModel(llm)]
agent = IntentAgent(llm=llm, tools=tools)

st.set_page_config(
    page_title="AutoGen Chat Agents",
    page_icon=":robot:",
    layout="wide"
)

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

    # tool = agent.choose_tools(query=user_input)


    with st.chat_message(name='autogen', avatar='assistant'):
        message_placeholder = st.empty()
    json_res = {"stop": False}
    display_answer_num = 0
    prev_answer = full_answer = ''
    while not json_res['stop']:
        json_res = requests.post(url=STREAM_API_URL,
                                 json={'prompt': user_input}).json()
        answer_list = json_res['answer']
        # 如果第一次请求就得到终止标志，说明存在缓存
        if display_answer_num == 0 and json_res['stop']:
            full_answer = '\n\n'.join([f"{ans['name']}: {ans['response']}" for ans in answer_list])
            message_placeholder.markdown(full_answer)
            break
        answer_num = len(answer_list)
        if answer_num == 0:
            continue
        current_answer = answer_list[-1]['response']
        name = answer_list[-1]['name']
        if answer_num > display_answer_num:
            prev_answer = full_answer
            if display_answer_num != 0:
                prev_answer += '\n\n'
            display_answer_num = answer_num
        else:
            current_answer = f'{name}: {current_answer}'
            full_answer = prev_answer + current_answer
        message_placeholder.markdown(full_answer)
