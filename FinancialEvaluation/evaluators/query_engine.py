import os
from typing import Dict, List

from llama_index.core import ChatPromptTemplate
from llama_index.core.base.llms.types import ChatMessage
from tqdm import tqdm
from evaluator import Evaluator
from time import sleep
import re

from llama_index.core.base.base_query_engine import BaseQueryEngine


class QueryEngineEvaluator(Evaluator):
    def __init__(self, query_engine: BaseQueryEngine, prompt_dict: Dict, choices, k, model_name):
        super().__init__(choices, model_name, k)
        self.query_engine = query_engine
        self.prompt_dict = prompt_dict

    def process_prompt(self, line, include_example=True, cot=False):
        if include_example:
            example = f"题目: {line['question']} \n 选项:"
            for choice in self.choices:
                example += f'\n{choice}. {line[f"{choice}"]}'

            example += '\n答案：'
            if cot:
                ans = line["answer"]
                content = "让我们一步一步思考，\n"+line["explanation"]+f"\n所以答案是{ans}。"
                return [
                    {"role": "user", "content": example},
                    {"role": "assistant", "content": content}
                ]
            else:
                return [
                    {"role": "user", "content": example},
                    {"role": "assistant", "content": line["answer"]}
                ]
        else:
            content = '选项:'
            for choice in self.choices:
                content += f'\n{choice}. {line[f"{choice}"]}'
            content += '\n答案:\n'
            if cot:
                content += "\n让我们一步步思考\n"
                return [
                        {"role": "user", "content": content}
                    ]
            else:
                return [
                    {"role": "user", "content": content},
                ]

    def generate_few_shot_prompt(self, subject, dev_df, cot=False):
        prompt = [
            {
                "role": "system",
                "content": f"你是一个中文人工智能助手，以下是中国关于{subject}考试的单项选择题以及相应的参考内容，"
                           f"请根据参考信息选出其中的正确答案。\n 举例："
            }
        ]
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            example = self.process_prompt(dev_df.iloc[i, :], include_example=True, cot=cot)
            if i == 0:
                example[0]["content"] = f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n" + \
                                        example[0]["content"]
            prompt += example
        return prompt

    def concat_query_engine_prompt(self, few_shot_prompt: List, question_prompt: List):
        prompt_dict = self.prompt_dict.copy()
        for prompt_type, prompt in prompt_dict.items():
            prompt_ls = few_shot_prompt + prompt + question_prompt
            chat_msg_ls = [ChatMessage(**prompt) for prompt in prompt_ls]
            prompt_template = ChatPromptTemplate.from_messages(chat_msg_ls)
            prompt_dict[prompt_type] = prompt_template
        return prompt_dict

    def update_query_engine_prompt(self, prompt_dict):
        self.query_engine.update_prompts(**prompt_dict)

    def eval_subject(self, subject_name, test_df, dev_df=None, few_shot=False,
                     save_result_dir=None, cot=False):
        correct_num = 0
        all_answer = {}
        result = []
        score = []
        if few_shot:
            few_shot_prompt = self.generate_few_shot_prompt(subject_name, dev_df, cot=cot)
        else:
            few_shot_prompt = [
                {
                    "role": "system",
                    "content": f"你是一个中文人工智能助手，以下是中国关于{subject_name}考试的单项选择题以及相应的参考内容，"
                               f"请根据参考信息选出其中的正确答案。"
                }
            ]
        answers = list(test_df['answer'])
        for row_index, row in tqdm(test_df.iterrows(), total=len(test_df)):
            question = row['question']
            question_prompt = self.process_prompt(row, include_example=False)
            prompt_dict = self.concat_query_engine_prompt(few_shot_prompt=few_shot_prompt,
                                                          question_prompt=question_prompt)
            self.update_query_engine_prompt(prompt_dict=prompt_dict)
            response = None
            while response is None:
                try:
                    response_str = self.query_engine.query(question)
                except Exception as msg:
                    print(msg)
                    sleep(5)
                    continue
            # if response is None:
            #     response_str = ""
            # else:
            #     response_str = response['choices'][0]['message']['content']
            # print(response_str)
            if cot:
                if not few_shot:
                    response_str=response_str.strip()
                    # if few_shot:
                    #     if len(response_str)>0:
                    #         if self.exact_match(response_str,row["answer"]):
                    #             correct_num+=1
                    #             correct=1
                    #         else:
                    #             correct=0
                    #     else:
                    #         correct=0
                    # else:
                    if len(response_str)>0:
                        ans_list=self.extract_ans(response_str)
                        if len(ans_list)>0 and (ans_list[-1]==row["answer"]):
                            correct_num+=1
                            correct=1
                        else:
                            correct=0
                    else:
                        correct=0
                else:
                    ans_list=re.findall(r"答案是(.+?)。",response_str)
                    if len(ans_list)==0:
                        ans_list=re.findall(r"答案为(.+?)。",response_str)
                    if len(ans_list)==0:
                        ans_list=re.findall(r"选项(.+?)是正确的。",response_str)

                    if len(ans_list)==0:
                        correct=0
                    else:
                        if self.exact_match(ans_list[-1],row["answer"]):
                            correct_num+=1
                            correct=1
                        else:
                            correct=0

            else:
                response_str=response_str.strip()
                if few_shot:
                    if len(response_str)>0:
                        if self.exact_match(response_str,row["answer"]):
                            correct_num+=1
                            correct=1
                        else:
                            correct=0
                    else:
                        correct=0
                else:
                    if len(response_str)>0:
                        ans_list=self.extract_ans(response_str)
                        if len(ans_list)>0 and (ans_list[-1]==row["answer"]):
                            correct_num+=1
                            correct=1
                        else:
                            correct=0
                    else:
                        correct=0
            if save_result_dir:
                result.append(response_str)
                score.append(correct)
            all_answer[str(row_index)] = row["answer"]
        correct_ratio = 100*correct_num/len(answers)

        if save_result_dir:
            test_df['model_output'] = result
            test_df["correctness"] = score
            test_df.to_csv(os.path.join(save_result_dir, f'{subject_name}_val.csv'), encoding="utf-8", index=False)
        return correct_ratio, all_answer

    def extract_ans(self, response_str):
        pattern = [
            r"^选([A-D])",
            r"^选项([A-D])",
            r"答案是\s?选?项?\s?([A-D])",
            r"答案为\s?选?项?\s?([A-D])",
            r"答案应为\s?选?项?\s?([A-D])",
            r"答案选\s?选?项?\s?([A-D])",
            r"答案是:\s?选?项?\s?([A-D])",
            r"答案应该是:\s?选?项?\s?([A-D])",
            r"正确的一项是\s?([A-D])",
            r"答案为:\s?选?项?\s?([A-D])",
            r"答案应为:\s?选?项?\s?([A-D])",
            r"答案:\s?选?项?\s?([A-D])",
            r"答案是：\s?选?项?\s?([A-D])",
            r"答案应该是：\s?选?项?\s?([A-D])",
            r"答案为：\s?选?项?\s?([A-D])",
            r"答案应为：\s?选?项?\s?([A-D])",
            r"答案：\s?选?项?\s?([A-D])",
            r"选项(.+?)是正确的。",
            r"答案为(.+?)。"
        ]
        ans_list = []
        if response_str[0] in ["A", 'B', 'C', 'D']:
            ans_list.append(response_str[0])
        for p in pattern:
            if len(ans_list) == 0:
                ans_list = re.findall(p, response_str)
            else:
                break
        return ans_list
