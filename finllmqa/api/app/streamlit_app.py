from typing import List
import streamlit as st

from finllmqa.agent.autogen.oai.client import remove_timeout_buffer
from finllmqa.agent.autogen_tools import get_autogen_stream_answer
from finllmqa.agent.langchain_tools import LangChainTool, FinInvestmentQA


def display_chat_block(agent: LangChainTool | List[LangChainTool]):
    if isinstance(agent, LangChainTool):
        agent_pool = [agent]
    else:
        agent_pool = agent
    place_holder_pool = []
    for i in range(len(agent_pool)):
        with st.chat_message(name='autogen', avatar='assistant'):
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


# def get_autogen_answer(display_agent, place_holder):
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

        # 删除过期的buffer
        remove_timeout_buffer()
        # tool = agent.choose_tools(query=user_input)
        tool = FinInvestmentQA()
        kg_matched_flag = tool.run(query=user_input)
        if kg_matched_flag:
            agent_pool = tool.knowledge_analysis_pool + tool.pretrain_inference_pool
            display_chat_block(agent=agent_pool)
        else:
            display_chat_block(agent=tool)


if __name__ == '__main__':
    main()
