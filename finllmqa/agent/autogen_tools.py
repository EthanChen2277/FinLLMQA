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
    llm_config = {"config_list": config_list_gpt, "stream": True,
                  "timeout": 600, "max_tokens": 8192}

    project_manager = autogen.UserProxyAgent(
        name="项目主管",
        description='项目主管在小组中永远是第一个发言的人，负责整体项目的规划、执行和管理，确保项目按时按质完成。\n协调各个小组成员的工作，分配任务和资源，\n'
                    '确保项目进度和质量。确保团队成员的沟通和协作顺畅，解决项目中的问题和障碍。\n'
                    '提出项目的方案并把对应的工作分配给知识图谱工程师和金融分析师',
        system_message='''
            你负责整体项目的规划、执行和管理，确保项目按时按质完成。协调各个小组成员的工作, 分配任务和资源，
            确保项目进度和质量。确保团队成员的沟通和协作顺畅，解决项目中的问题和障碍。
            你提出项目的方案后需要把对应的工作分配给知识图谱工程师（维护知识图谱）和金融分析师（分析金融数据）
            项目完成后请回复"TERMINATE"
            ''',
        is_termination_msg=lambda x: x.get("content", "") and "TERMINATE" in x.get("content", "").rstrip(),
        code_execution_config=False,
        human_input_mode='NEVER',
        llm_config=llm_config
    )
    kg_engineer = autogen.AssistantAgent(
        name="知识图谱工程师",
        description='知识图谱工程师负责维护和优化知识图谱的结构和数据模型, 编写和优化 Cypher 语句, 确保知识图谱的数据质量和一致性，'
                    '处理数据的更新和清理',
        system_message='你负责维护和优化知识图谱的结构和数据模型, 编写和优化 Cypher 语句, 确保知识图谱的数据质量和一致性，'
                    '处理数据的更新和清理',
        llm_config=llm_config
    )
    fin_analyst = autogen.AssistantAgent(
        name="金融分析师",
        description='金融分析师需要分析当前金融领域的趋势和变化，提出对知识图谱扩展所需的新数据来源和数据类型\n'
                    '提供基于金融分析的指导，确定在股票投资领域需要关注的关键指标和数据\n '
                    '并协助评估现有数据在投资决策中的价值和局限性，提出改进和补充的建议',
        system_message='你需要分析当前金融领域的趋势和变化，提出对知识图谱扩展所需的新数据来源和数据类型\n'
                       '提供基于金融分析的指导，确定在股票投资领域需要关注的关键指标和数据\n '
                       '并协助评估现有数据在投资决策中的价值和局限性，提出改进和补充的建议',
        llm_config=llm_config
    )

    allowed_speaker_transitions_dict = {
        project_manager: [kg_engineer, fin_analyst],
        kg_engineer: [project_manager],
        fin_analyst: [project_manager],
    }
    groupchat = autogen.GroupChat(
        agents=[project_manager, kg_engineer, fin_analyst], messages=[], max_round=5,
        admin_name=project_manager.name, speaker_transitions_type='allowed',
        allowed_or_disallowed_speaker_transitions=allowed_speaker_transitions_dict)
    del llm_config['functions']
    manager = autogen.GroupChatManager(
        groupchat=groupchat, llm_config=llm_config)
    print(f'initial autogen chat with query: {prompt}')

    project_manager.initiate_chat(
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
