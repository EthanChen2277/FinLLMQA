import datetime
import threading
import time

from finllmqa.agent import autogen
from finllmqa.agent.autogen.oai.client import STREAM_BUFFER
from finllmqa.api.core import LLM_API_URL


# autogen stream utils
def stream_chat(prompt):
    config_list_gpt = [
        {
            "model": "chatglm3-6b",
            "base_url": LLM_API_URL,
            "api_type": "openai",
            "api_key": "NULL",
        }
    ]
    llm_config = {"config_list": config_list_gpt,
                  "stream": True, "timeout": 600,
                  "max_tokens": 8192}

    admin = autogen.UserProxyAgent(
        name="组长",
        description='组长负责整个项目的规划和管理，确保项目按时按质完成。组长还需要与其他团队成员沟通，确保信息畅通，'
                    '并协调解决团队内部的问题。问题回答完毕后回复"TERMINATE"',
        system_message='组长负责整个项目的规划和管理，确保项目按时按质完成。组长还需要与其他团队成员沟通，确保信息畅通，'
                       '并协调解决团队内部的问题。问题回答完毕后回复"TERMINATE"',
        is_termination_msg=lambda x: x.get("content", "") and "TERMINATE" in x.get("content", "").rstrip(),
        code_execution_config=False,
        human_input_mode='NEVER',
        llm_config=llm_config
    )
    analyst = autogen.AssistantAgent(
        name="金融分析师",
        description='金融分析师负责对金融市场进行深入研究和分析，包括宏观经济、行业趋势、公司基本面等。',
        system_message='金融分析师负责对金融市场进行深入研究和分析，包括宏观经济、行业趋势、公司基本面等。',
        llm_config=llm_config
    )
    risk_manager = autogen.AssistantAgent(
        name="风险管理师",
        description='风险管理师的任务是评估和控制投资过程中的风险，确保投资组合的风险水平符合既定的风险承受能力。',
        system_message='风险管理师的任务是评估和控制投资过程中的风险，确保投资组合的风险水平符合既定的风险承受能力。',
        llm_config=llm_config
    )
    financial_analyst = autogen.AssistantAgent(
        name="财务分析师",
        description='财务分析师专注于公司的财务报表分析，评估公司的财务状况和未来的盈利能力',
        system_message='财务分析师专注于公司的财务报表分析，评估公司的财务状况和未来的盈利能力',
        llm_config=llm_config
    )

    groupchat = autogen.GroupChat(
        agents=[admin, analyst, risk_manager, financial_analyst], messages=[])
    manager = autogen.GroupChatManager(
        groupchat=groupchat, llm_config=llm_config)
    print(f'initial autogen chat with query: {prompt}')
    admin.initiate_chat(
        manager, message=prompt, silent=True)
    # 如果stream_chat调用完成，给返回加一个停止词[stop]
    STREAM_BUFFER[prompt]["stop"] = True


def get_autogen_stream_answer(query: str):
    # 判断是否已在生成，只有首次才调stream_chat
    if STREAM_BUFFER.get(query) is None:
        # 在线程中调用stream_chat
        try:
            sub_thread = threading.Thread(
                target=stream_chat, args=[query], daemon=True)
            sub_thread.start()
        except Exception as err_msg:
            return err_msg
    while query not in STREAM_BUFFER.keys():
        time.sleep(0.5)
    if STREAM_BUFFER[query]['stop']:
        return '[DONE]'
    else:
        return STREAM_BUFFER[query]["answer"]
