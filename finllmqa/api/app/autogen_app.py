from fastapi import FastAPI, Request
import uvicorn
import json
import datetime
import threading

from finllmqa.agent import autogen
from finllmqa.agent.autogen.oai.client import stream_buffer
from finllmqa.api.core import LLM_URL

autogen_app = FastAPI()


def autogen_stream(prompt, history):
    config_list_gpt = [
        {
            "model": "chatglm3-6b",
            "base_url": LLM_URL,
            "api_type": "openai",
            "api_key": "NULL",
        }
    ]
    llm_config = {"config_list": config_list_gpt,
                  "stream": True, "timeout": 600,
                  "max_tokens": 8192}

    user_proxy = autogen.UserProxyAgent(
        name="智能体",
        system_message="智能管理员",
        human_input_mode="NEVER"
    )
    analyst = autogen.AssistantAgent(
        name="金融分析师",
        description='金融方面的专家，善于分析金融问题。问题回答完毕后回复"TERMINATE"',
        llm_config=llm_config
    )
    critic = autogen.AssistantAgent(
        name="评论家",
        system_message="富有批判性，对小组内其他成员的回答进行评论并给出建议",
        llm_config=llm_config,
    )

    groupchat = autogen.GroupChat(
        agents=[user_proxy, analyst, critic], messages=[], max_round=5)
    manager = autogen.GroupChatManager(
        groupchat=groupchat, llm_config=llm_config)
    user_proxy.initiate_chat(
        manager, message=prompt)
    stream_buffer[prompt]["stop"] = True


def remove_timeout_buffer():
    for key in stream_buffer.copy():
        diff = datetime.datetime.now() - stream_buffer[key]["time"]
        seconds = diff.total_seconds()
        # print(key + ": 已存在" + str(seconds) + "秒")
        if seconds > 1200:
            if stream_buffer[key]["stop"]:
                del stream_buffer[key]
                print(key + "：已被从缓存中移除")
            else:
                stream_buffer[key]["stop"] = True
                print(key + "：已被标识为结束")


@autogen_app.post("/autogen/stream")
async def create_item(request: Request):
    # 删除过期的buffer
    remove_timeout_buffer()
    # 获取入参
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    stop = False
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
                                 "stop": stop, "time": now}
        # 在线程中调用stream_chat
        sub_thread = threading.Thread(
            target=autogen_stream, args=(prompt, history))
        sub_thread.start()
    # 异步返回response
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer_list = stream_buffer[prompt]["answer"]
    # 如果stream_chat调用完成，给返回加一个停止词[stop]
    if stream_buffer[prompt]["stop"]:
        stop = True
    response = {
        "answer": answer_list,
        "status": 200,
        "stop": stop,
        "time": time
    }

    return response


if __name__ == '__main__':
    uvicorn.run(autogen_app, host='0.0.0.0', port=8006, workers=1)
