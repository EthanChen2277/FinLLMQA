from fastapi import FastAPI
import uvicorn
import datetime
import threading
from pydantic import ConfigDict, BaseModel

from finllmqa.agent import autogen
from finllmqa.agent.autogen.oai.client import stream_buffer
from finllmqa.api.core import LLM_API_URL

autogen_app = FastAPI()


class GetStream(BaseModel):
    prompt: str = None,
    model_config = ConfigDict(json_schema_extra={
        'example': {
            'prompt': '你好'
        }
    })


def autogen_stream(prompt):
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
    admin.initiate_chat(
        manager, message=prompt)
    stream_buffer[prompt]["stop"] = True


def remove_timeout_buffer():
    for key in stream_buffer.copy():
        diff = datetime.datetime.now() - stream_buffer[key]["time"]
        seconds = diff.total_seconds()
        # print(key + ": 已存在" + str(seconds) + "秒")
        if seconds > 10:
            if stream_buffer[key]["stop"]:
                del stream_buffer[key]
                print(key + "：已被从缓存中移除")
            else:
                stream_buffer[key]["stop"] = True
                print(key + "：已被标识为结束")


@autogen_app.post("/autogen/stream")
async def create_item(model: GetStream):
    # 删除过期的buffer
    remove_timeout_buffer()
    # 获取入参
    prompt = model.prompt
    # 判断是否已在生成，只有首次才调stream_chat
    now = datetime.datetime.now()
    if prompt == '你好':
        return {
            "answer": [
                {
                    "name": "chatglm3",
                    "response": "您好，我是Chatglm3智体，基于Autogen构建，可以使用多个角色会话回答您在编程方面的问题。[stop]"
                }
            ],
            "status": 200,
            "stop": True,
            "time": now.strftime("%Y-%m-%d %H:%M:%S")
        }
    if stream_buffer.get(prompt) is None:
        stream_buffer[prompt] = {"answer": [],
                                 "stop": False, "time": now}
        # 在线程中调用stream_chat
        sub_thread = threading.Thread(
            target=autogen_stream, args=[prompt])
        sub_thread.start()
    # 异步返回response
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer_list = stream_buffer[prompt]["answer"]
    # 如果stream_chat调用完成，给返回加一个停止词[stop]
    print(stream_buffer)
    response = {
        "answer": answer_list,
        "status": 200,
        "stop": stream_buffer[prompt]["stop"],
        "time": time
    }

    return response


if __name__ == '__main__':
    uvicorn.run(autogen_app, host='0.0.0.0', port=8006, workers=1)
