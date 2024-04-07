import os
from typing import Dict

from llama_index.core.base.base_query_engine import BaseQueryEngine
from tqdm import tqdm
import openai
from evaluator import Evaluator
from time import sleep
import re


class QueryEngineEvaluator(Evaluator):
    def __init__(self, query_engine: BaseQueryEngine, prompt_dict: Dict, choices, k, model_name):
        super().__init__(choices, model_name, k)
        self.query_engine = query_engine
        self.prompt_dict = prompt_dict

    def format_example(self, line, include_answer=True, cot=False):
        example=line['question']
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'

        example += '\n答案：'
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
                "role": "system",
                "content": f"你是一个中文人工智能助手，以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。"
            }
        ]
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            tmp=self.format_example(dev_df.iloc[i,:],include_answer=True,cot=cot)
            if i == 0:
                tmp[0]["content"] = f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n"+tmp[0]["content"]
            prompt += tmp
        return prompt

    def eval_subject(self, subject_name, test_df, dev_df=None, few_shot=False,
                     save_result_dir=None, cot=False):
        correct_num = 0
        all_answer = {}
        if save_result_dir:
            result = []
            score=[]
        if few_shot:
            few_shot_prompt = self.generate_few_shot_prompt(subject_name, dev_df,cot=cot)
        else:
            few_shot_prompt=[
                {
                    "role":"system",
                    "content":f"你是一个中文人工智能助手，以下是中国关于{subject_name}考试的单项选择题，请选出其中的正确答案。"
                }
            ]
        answers = list(test_df['answer'])
        for row_index, row in tqdm(test_df.iterrows(),total=len(test_df)):
            question = self.format_example(row, include_answer=False,)
            full_prompt = few_shot_prompt + question
            if not few_shot:
                full_prompt[-1]["content"]=f"以下是中国关于{subject_name}考试的单项选择题，请选出其中的正确答案。\n\n"+full_prompt[-1]["content"]
            response=None
            timeout_counter=0
            while response is None and timeout_counter <= 30:
                try:
                    sleep(1.5)
                    response = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=full_prompt,
                        temperature=0.
                    )
                except Exception as msg:
                    if "timeout=600" in str(msg):
                        timeout_counter+=1
                    print(msg)
                    sleep(5)
                    continue
            if response is None:
                response_str = ""
            else:
                response_str = response['choices'][0]['message']['content']
            # print(response_str)
            if cot:
              if not few_shot:
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
            test_df['model_output']=result
            test_df["correctness"]=score
            test_df.to_csv(os.path.join(save_result_dir, f'{subject_name}_val.csv'),encoding="utf-8",index=False)
        return correct_ratio,all_answer

    def extract_ans(self,response_str):
        pattern=[
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
        ans_list=[]
        if response_str[0] in ["A",'B','C','D']:
            ans_list.append(response_str[0])
        for p in pattern:
            if len(ans_list)==0:
                ans_list=re.findall(p,response_str)
            else:
                break
        return ans_list

    def get_llama_index_query_result(self, query_engine: BaseQueryEngine, prompt_dict: Dict):
        query_engine.update_prompts(prompts_dict=prompt_dict)
        try:
            response_str = query_engine.query('')
            return response_str
        except:
            return ''

