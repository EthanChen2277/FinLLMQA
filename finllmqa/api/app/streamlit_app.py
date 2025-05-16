from typing import List

import pandas as pd
import streamlit as st

from finllmqa.api.core import LLM
from finllmqa.agent.autogen.oai.client import remove_timeout_buffer
from finllmqa.agent.autogen_tools import get_autogen_stream_answer
from finllmqa.agent.langchain_tools import LangChainTool, SummarizeTool, KGRetrieverTool
from finllmqa.agent.qa_tools import FinInvestmentQA, IntentAgent

st.set_page_config(
    page_title="Financial QA",
    page_icon=":robot:",
    layout="wide"
)
st.title('🤖 Financial QA')

if "conversations" not in st.session_state:
    st.session_state.conversations = []

if "is_answer_summarized" not in st.session_state:
    st.session_state.is_answer_summarized = False

if 'autogen_prompt' not in st.session_state:
    st.session_state.autogen_prompt = None
    
if "is_autogen_start" not in st.session_state:
    st.session_state.is_autogen_start = False


def stream_chat_block(agent: LangChainTool | List[LangChainTool]):
    if isinstance(agent, LangChainTool):
        agent_pool = [agent]
    else:
        agent_pool = agent
    place_holder_pool = []
    description_ls = []
    for i in range(len(agent_pool)):
        description = f'**分析角度:{agent_pool[i].angle}**'
        description_ls.append(description)
        st.markdown(description)
        with st.chat_message(name='llm', avatar='assistant'):
            message_placeholder = st.empty()
            place_holder_pool.append(message_placeholder)
        if agent_pool[i].name == '数据分析':
            display_table_data = agent_pool[i].display_table
            stock_ls = [table['stock'] for table in display_table_data]
            prop_ls = display_table_data[0]['data'].keys()
            selected_prop = st.selectbox("数据类别", prop_ls, index=0)
            tab_ls = st.tabs(stock_ls)
            for idx, tab in enumerate(tab_ls):
                df = pd.DataFrame(display_table_data[idx]['data'][selected_prop])
                with tab:
                    expander = st.expander('查看具体数据')
                    expander.table(df)
    display_agent_pool = agent_pool
    display_ans_generator_pool = [agent.get_stream_response() for agent in display_agent_pool]
    display_answer_ls = ['' for i in range(len(display_ans_generator_pool))]
    finished_agent_index_ls = []
    while len(display_ans_generator_pool) != len(finished_agent_index_ls):
        for i, answer_generator in enumerate(display_ans_generator_pool):
            if i not in finished_agent_index_ls:
                try:
                    display_answer_ls[i] += next(answer_generator).content
                    place_holder = place_holder_pool[i]
                    place_holder.markdown(display_answer_ls[i])
                    # angle = display_agent.angle
                    # place_holder.markdown(f'分析角度: {angle} \n\n')
                except StopIteration:
                    finished_agent_index_ls.append(i)
                    continue
    return description_ls, display_answer_ls, agent_pool


def get_summarized_answer(query=None):
    latest_conversation = st.session_state.conversations[-1]
    query = query or latest_conversation[0]['query']
    if len(latest_conversation) > 0:
        answer_ls = [message['answer'] for message in latest_conversation]
        total_answer = '\n'.join(answer_ls)
        summarize_tool = SummarizeTool(llm=LLM)
        chunks = summarize_tool.get_stream_response(query=query, total_answer=total_answer)
        description = '**总结上一轮对话**'
        st.markdown(description)
        with st.chat_message(name='llm', avatar='assistant'):
            place_holder = st.empty()
        answer = ''
        for chunk in chunks:
            answer += chunk.content
            place_holder.markdown(answer)
        return description, answer, summarize_tool
    else:
        st.markdown('**没有进行对话**')
        return None, None, None


def get_autogen_answer():
    prompt = st.session_state.autogen_prompt
    if prompt:
        autogen_agent = LangChainTool(llm=LLM)
        autogen_agent.name = 'autogen'
        description = '**自动对话**'
        st.markdown(description)
        with st.chat_message(name='llm', avatar='assistant'):
            place_holder = st.empty()
        while True:
            answer = get_autogen_stream_answer(query=prompt)
            if answer == '[DONE]':
                break
            place_holder.markdown(answer)
        return description, answer, autogen_agent
    else:
        st.markdown('**问题类别不属于股票投资，无法给出知识图谱优化建议**')
        return None, None, None


with st.sidebar:
    st.header("QA Agent Configuration")
    selected_qa_type = st.selectbox("问题类别", ['股票投资', '财经百科', '其他'], index=0)

    if selected_qa_type == '股票投资':
        retriever_options = ['text2cypher: pathway']
    else:
        retriever_options = ['KG RAG', 'Vector RAG', 'KG + Vec RAG']
    selected_retriever = st.selectbox("检索方式", retriever_options, index=0)

for conversation in st.session_state.conversations:
    st.chat_message(name="user", avatar="user").write(conversation[-1]['query'])
    for message in conversation:
        st.markdown(message['description'])
        st.chat_message(name='llm', avatar='assistant').write(message['answer'])
        agent = message['agent']
        if agent.name == '数据分析':
            display_table_data = agent.display_table
            stock_ls = [table['stock'] for table in display_table_data]
            prop_ls = display_table_data[0]['data'].keys()
            selected_prop = st.selectbox("数据类别", prop_ls, index=0)
            tab_ls = st.tabs(stock_ls)
            for idx, tab in enumerate(tab_ls):
                df = pd.DataFrame(display_table_data[idx]['data'][selected_prop])
                with tab:
                    expander = st.expander('查看具体数据')
                    expander.table(df)

if st.session_state.is_answer_summarized:
    latest_conversation = st.session_state.conversations[-1]
    summary_description, summary_answer, summary_agent = get_summarized_answer()
    if summary_description is not None and summary_answer is not None and summary_agent is not None:
        one_conversation = [{'query': latest_conversation[0]['query'],
                             'description': summary_description, 'answer': summary_answer, 'agent': summary_agent}]
        st.session_state.conversations.append(one_conversation)
        st.session_state.is_answer_summarized = False
        
if st.session_state.is_autogen_start:
    latest_conversation = st.session_state.conversations[-1]
    autogen_description, autogen_answer, autogen_agent = get_autogen_answer()
    if autogen_description is not None and autogen_answer is not None:
        one_conversation = [{'query': latest_conversation[0]['query'],
                             'description': autogen_description, 'answer': autogen_answer, 'agent': autogen_agent}]
        st.session_state.conversations.append(one_conversation)
        st.session_state.is_autogen_start = False

with st.chat_message(name="user", avatar="user"):
    input_placeholder = st.empty()

user_input = st.chat_input("Type something...")
if user_input:
    input_placeholder.markdown(user_input)
    # 删除过期的buffer
    remove_timeout_buffer()
    # if len(st.session_state.conversations) > 0:
    #     # 如果是股票投资问题，且已经有对话记录，则需要将上次的回答进行总结
    #     latest_conversation = st.session_state.conversations[-1]
    #     if st.session_state.is_answer_summarized:
    #         chat_messages = [
    #             ('human', latest_conversation[0]['query']),
    #             ('ai', latest_conversation[0]['answer']),
    #             ('human', user_input)
    #         ]
    #         base_llm_tool = LangChainTool(llm=LLM)
    #         base_llm_tool.chat_messages = chat_messages
    #         description_ls, answer_ls, agent_pool = stream_chat_block(agent=base_llm_tool)
    #         st.session_state.is_answer_summarized = False
    #     else:
    #         summary_description, summary_answer = get_summarized_answer()
    #         chat_messages = [
    #             ('human', latest_conversation[0]['query']),
    #             ('ai', summary_answer),
    #             ('human', user_input)
    #         ]
    #         base_llm_tool = LangChainTool(llm=LLM)
    #         base_llm_tool.chat_messages = chat_messages
    #         description_ls, answer_ls, agent_pool = stream_chat_block(agent=base_llm_tool)
    # else:
        # if selected_qa_type == '其他':
        #     tool = LangChainTool(llm=LLM)
        #     tool.get_reference(query=user_input)
        #     description_ls, answer_ls, agent_pool = stream_chat_block(agent=tool)
        # else:
    with st.spinner('Initializing agent...'):
        agent = IntentAgent(llm=LLM)
    with st.spinner('Classifying question...'):
        tool = agent.choose_qa_tools(query=user_input)
    if tool.name != selected_qa_type:
        st.chat_message(name='assistant', avatar='assistant').write(
            f'您的问题与已选择的问题类型({selected_qa_type})不匹配, 正在为您转换至{tool.name}类大模型工具进行回答')
        selected_qa_type = tool.name
    if tool.name == '股票投资':
        with st.spinner('Analyzing question...'):
            kg_matched_flag = tool.run(query=user_input)
        if kg_matched_flag:
            agent_pool = tool.knowledge_analysis_pool + tool.pretrain_inference_pool
            description_ls, answer_ls, agent_pool = stream_chat_block(agent=agent_pool)
            question_intent_ls = [str(ka.question_intent) for ka in tool.knowledge_analysis_pool]
            question_intent = "\n".join(question_intent_ls)
            autogen_prompt = f"""
                现在存在一个股票的基本面和行情知识图谱，图谱结构如下
                -------------------------
                {KGRetrieverTool().get_kg_schema()}
                -------------------------
                基于用户提出的问题:{user_input}, 从问题中提取出与该知识图谱匹配的意图有:
                {question_intent}
                请分析图谱结构设计是否合理，同时对于用户的问题分析还可以补充哪些数据到知识图谱中，给出优化后知识图谱的结构。
                请项目主管先给出具体方案。
            """
            st.session_state.autogen_prompt = autogen_prompt
        else:
            description_ls, answer_ls, agent_pool = stream_chat_block(agent=tool)
    elif tool.name == '财经百科':
        description_ls, answer_ls, agent_pool = stream_chat_block(agent=tool)
    else:
        tool.get_reference(query=user_input)
        description_ls, answer_ls, agent_pool = stream_chat_block(agent=tool)
    one_conversation = []
    for description, answer, agent in zip(description_ls, answer_ls, agent_pool):
        one_conversation.append({'query': user_input, 'description': description, 'answer': answer, 'agent': agent})
    st.session_state.conversations.append(one_conversation)

button_column_ls = st.columns(5)
with button_column_ls[1]:
    clear_history_button = st.button('清除历史对话')
    if clear_history_button:
        st.session_state.conversations = []
        st.rerun()
with button_column_ls[2]:
    summary_button = st.button('总结对话')
    if summary_button:
        st.session_state.is_answer_summarized = True
        st.rerun()
if st.session_state.autogen_prompt is not None:
    with button_column_ls[3]:
        autogen_button = st.button('优化知识图谱建议')
        if autogen_button:
            st.session_state.is_autogen_start = True
            st.rerun()
