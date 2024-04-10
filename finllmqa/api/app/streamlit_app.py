import time
from typing import List
import streamlit as st

from finllmqa.agent.autogen.oai.client import remove_timeout_buffer
from finllmqa.agent.autogen_tools import get_autogen_stream_answer
from finllmqa.agent.langchain_tools import LangChainTool, SummarizeTool
from finllmqa.agent.qa_tools import FinInvestmentQA

st.session_state['answer_ls'] = []


def display_chat_block(agent: LangChainTool | List[LangChainTool]):
    if isinstance(agent, LangChainTool):
        agent_pool = [agent]
    else:
        agent_pool = agent
    place_holder_pool = []
    for i in range(len(agent_pool)):
        with st.chat_message(name='llm', avatar='assistant'):
            message_placeholder = st.empty()
            place_holder_pool.append(message_placeholder)
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
    return display_answer_ls


def clean_history_answer():
    st.session_state['answers'] = []


def get_summarized_answer(query, total_answer, input_placeholder):
    # clean_history_answer()
    input_placeholder.markdown(query)
    st.write('111')
    st.write(total_answer)
    st.write('111')
    st.write('\n'.join(st.session_state.get('answer_ls')))
    summarize_tool = SummarizeTool()
    chunks = summarize_tool.get_stream_response(query=query, total_answer=total_answer)
    with st.chat_message(name='llm', avatar='assistant'):
        place_holder = st.empty()
    answer = ''
    for chunk in chunks:
        answer += chunk.content
        place_holder.markdown(answer)
    # st.session_state.history.append(answer)

# def display_autogen_answer(prompt):
#     while len(display_agent_pool) != len(finished_agent_index_pool):
#         for i in range(len(display_agent_pool)):
#             if i not in finished_agent_index_pool:
#                 display_agent = display_agent_pool[i]
#                 place_holder = place_holder_pool[i]
#                 query = display_agent.prompt
#                 # angle = display_agent.angle
#                 # place_holder.markdown(f'分析角度: {angle} \n\n')
#                 answer = get_autogen_stream_answer(query=query)
#                 if answer == '[DONE]':
#                     finished_agent_index_pool.append(i)
#                     continue
#                 place_holder.markdown(answer)


def main():
    st.set_page_config(
        page_title="Financial QA",
        page_icon=":robot:",
        layout="wide"
    )

    with st.sidebar:
        st.header("QA Agent Configuration")
        selected_model = st.selectbox("问题类别", ['金融投资', '财经百科'], index=0)

        if selected_model == '金融投资':
            retriever_options = ['text2cypher: pathway']
        else:
            retriever_options = ['text2cypher', 'KG RAG', 'Vector RAG', 'KG + Vec RAG']
        selected_retriever = st.selectbox("检索方式", retriever_options, index=0)

    with st.chat_message(name="user", avatar="user"):
        input_placeholder = st.empty()
    user_input = st.chat_input("Type something...")
    if user_input:
        input_placeholder.markdown(user_input)
        if not selected_model:
            st.warning(
                '请先选择模型！', icon="⚠️")
            st.stop()

        # 删除过期的buffer
        remove_timeout_buffer()
        # tool = agent.choose_tools(query=user_input)
        with st.spinner('Initializing Agent...'):
            tool = FinInvestmentQA()
        with st.spinner('Analyzing question...'):
            kg_matched_flag = tool.run(query=user_input)
        if kg_matched_flag:
            agent_pool = tool.knowledge_analysis_pool + tool.pretrain_inference_pool
            answer_ls = display_chat_block(agent=agent_pool)
        else:
            answer_ls = display_chat_block(agent=tool)
        st.session_state['answer_ls'] = answer_ls
        st.session_state['user_input'] = user_input
        total_answer = '\n'.join(st.session_state.get('answer_ls'))
        st.write(total_answer)
    summary_button = st.button('Summarize Answer')
    if summary_button:
        query = st.session_state.get('user_input', None)
        total_answer = '\n'.join(st.session_state.get('answer_ls'))
        st.write(total_answer)
        if query is not None:
            get_summarized_answer(query=query, total_answer=total_answer,
                                  input_placeholder=input_placeholder)
    autogen_button = st.button('Start Autogen')


if __name__ == '__main__':
    main()
