import os
from tqdm import tqdm
from evaluator import Evaluator
from time import sleep
import re

from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.openai import OpenAI
from finllmqa.api.core import LLM_API_URL


class OpenAI_Evaluator(Evaluator):
    def __init__(self, choices, k, model_name, llm = None):
        super().__init__(choices, model_name, k)
        self.llm = llm or OpenAI(api_key='null', api_base=LLM_API_URL)

    def format_example(self,line,include_answer=True,cot=False):
        example=line['question']
        for choice in self.choices:
            example+=f'\n{choice}. {line[f"{choice}"]}'

        example+='\n答案：'
        if include_answer:
            if cot:
                ans=line["answer"]
                content="让我们一步一步思考，\n"+line["explanation"]+f"\n所以答案是{ans}。"
                return [
                    {"role":"user","content":example},
                    {"role":"assistant","content":content}
                ]
            else:
                return [
                    {"role":"user","content":example},
                    {"role":"assistant","content":line["answer"]}
                ]
        else:
            if cot:
                example += "\n让我们一步步思考,\n"
                return [
                        {"role":"user","content":example}
                        ]
            else:
                return [
                    {"role":"user","content":example},
                ]
    def generate_few_shot_prompt(self, subject, dev_df, cot=False):
        prompt=[
            {
                "role":"system",
                "content":f"你是一个中文人工智能助手，以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。"
            }
        ]
        k=self.k
        if self.k==-1:
            k=dev_df.shape[0]
        for i in range(k):
            tmp=self.format_example(dev_df.iloc[i,:],include_answer=True,cot=cot)
            if i==0:
                tmp[0]["content"]=f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n"+tmp[0]["content"]
            prompt+=tmp
        return prompt


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
            question = self.format_example(row, include_answer=False,)
            full_prompt = few_shot_prompt + question
            if not few_shot:
                full_prompt[-1]["content"]=f"以下是中国关于{subject_name}考试的单项选择题，请选出其中的正确答案。\n\n"+full_prompt[-1]["content"]
            messages = [ChatMessage(**prompt) for prompt in full_prompt]
            response_str=None
            while response_str is None:
                try:
                    response_str = self.llm.chat(messages=messages)
                    response_str = str(response_str)
                except Exception as msg:
                    print(msg)
                    sleep(5)
                    continue
            if cot:
                if not few_shot:
                    response_str=response_str.strip()
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
        for choice in ["A", 'B', 'C', 'D']:
            if choice in response_str:
                ans_list.append(choice)
        for p in pattern:
            if len(ans_list) == 0:
                ans_list = re.findall(p, response_str)
            else:
                break
        return ans_list
