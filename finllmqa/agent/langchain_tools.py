import ast
import collections
import json
import math
import re
import threading
from abc import ABC, abstractmethod
import logging
import pymysql
from datetime import datetime, timedelta
import pandas as pd
import asyncio
from copy import deepcopy
from langchain.tools import BaseTool
from langchain.base_language import BaseLanguageModel
from langchain import PromptTemplate, LLMChain
from langchain_community.llms.chatglm3 import ChatGLM3

from finllmqa.kg.search import AnswerSearcher
from finllmqa.api.embedding import get_embedding
from finllmqa.api.core import SERVER_API_URL, SERVER_LLM_API_PORT, CHAT_API_URL
from finllmqa.vector_db.construct import Milvus

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BaseModel(ABC):
    def __init__(self, llm: BaseLanguageModel, verbose: bool = True, *args, **kwargs):
        self.llm = llm
        self.name = "其它"
        self.description = '''
        无需任何信息，直接回答
        '''
        self._template = """
        问题：{query}
        答案："""
        self.prompt = None
        self.llm_chain = None
        self.reference = dict()
        self.verbose = verbose
        self.progress_func = kwargs.get('progress_func')
        self.progress_key = kwargs.get('progress_key')
        # self.progress(progress_text='开始分析问题')

    def get_prompt(self):
        self.prompt = PromptTemplate.from_template(self._template)

    def get_reference(self, query):
        self.reference = dict()

    def get_llm_chain(self):
        self.get_prompt()
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt, verbose=self.verbose)

    def progress(self, *args, **kwargs):
        """
        问题处理进度
        """
        logging.debug(f'问题处理进度：{kwargs.get("progress_text")}')
        if self.progress_func:
            self.progress_func(self.progress_key, kwargs.get("progress_text"))

    def run(self, query):
        self.get_reference(query)
        self.get_llm_chain()
        self.progress(progress_text='调用LLM')
        resp = self.llm_chain.predict(
            query=query,
            **self.reference
        )
        if self.verbose:
            logging.info(f"Tool's name is {self.name}. Its reference is {self.reference}\n "
                         f"Q: {query} LLM Answer: {resp}.")
        return resp

    # 异步调用组件
    async def async_generate(self, query):
        self.get_reference(query)
        # 将回答的问题开启流式输出
        self.llm.streaming = True
        self.get_llm_chain()
        if self.verbose:
            logging.info(f"Tool's name is {self.name}. Its reference is {self.reference}.")
        self.progress(progress_text='调用LLM')
        resp = await self.llm_chain.apredict(
            query=query,
            **self.reference
        )
        return resp


class IntentAgent(BaseModel):
    def __init__(self, llm: BaseLanguageModel, tools: list, verbose: bool = True, *args, **kwargs):
        super().__init__(llm, verbose, *args, **kwargs)
        self.name = "意图识别"
        self.description = '''
        识别问题意图，选择相应的prompt模板
        '''
        self.tools = tools
        self.progress(progress_text='开始对问题分类')

        self._template = """
        现在有一些意图，类别为{intents}，你的任务是理解用户问题的意图，并判断该问题属于哪一类意图。
        回复的意图类别必须在提供的类别中，并且必须按格式回复：“意图类别：<>”。

        举例：
        问题：今天几号？
        意图类别：查询时间

        问题：现在几点？
        意图类别：查询时间

        问题：昨天妈妈给了100元，我买了三本练习本，一本2元，买了五盒巧克力，还剩下74元，一盒巧克力多少钱？
        意图类别：算术

        问题：现在拿100万进行投资，分众传媒和太平洋哪个更值得投资？
        意图类别：金融投资

        问题：分众传媒这家公司怎么样？
        意图类别：金融投资

        问题：“{query}”获取相关数据
        意图类别："""

    def get_reference(self, query):
        # self.progress(progress_text=f'')
        self.reference = dict(
            intents=[tool.name for tool in self.tools]
        )

    def choose_tools(self, query):
        resp = self.run(query=query)
        self.progress(progress_text=f'分类结果为：{resp}')
        logging.info(f'意图识别结果为：{resp}')
        tool_names = [tool.name for tool in self.tools]
        if resp in tool_names:
            return self.tools[tool_names.index(resp)]
        return self.tools[-1]


class IETool(BaseModel):
    def __init__(self, llm: BaseLanguageModel, verbose: bool = True, *args, **kwargs):
        super().__init__(llm, verbose, *args, **kwargs)
        self.name = "金融信息抽取"
        self.description = '''
        当需要从问题中抽取主体信息时使用
        '''
        self._template = """
        你是一名金融信息抽取员，需要从问题中抽取出['公司', '行业', '时间', '意图']，抽取的内容必须是问题中的字段，不能乱编，
        并且必须以“'公司': [], '行业': [], '时间': [], '意图': []”的形式回复。

        举例1：
            问题：2019年药明康德衍生金融资产和其他非流动金融资产分别是多少元?
            信息抽取：'公司': ['药明康德'], '行业': [], '时间': ['2019年'], '意图': ['衍生金融资产', '其他非流动金融资产']

        举例2：
            问题：上海中谷物流股份有限公司2020年营业收入是多少元?
            信息抽取：'公司': ['上海中谷物流股份有限公司'], '行业': [], '时间': ['2020年'], '意图': ['营业收入']

        举例3：
            问题：激光行业近三年的发展如何？
            信息抽取：'公司': [], '行业': ['激光行业'], '时间': ['最近三年'], '意图': ['发展']

        举例4：
            问题：证券行业最近的发展如何？
            信息抽取：'公司': [], '行业': ['证券行业'], '时间': ['最近一年'], '意图': ['发展']

        举例5：
            问题：分众传媒与新潮传媒谁的利润率更高？
            信息抽取：'公司': ['分众传媒', '新潮传媒'], '行业': [], '时间': [], '意图': ['利润率']

        举例6：
            问题：广发证券 东方财富证券 东方证券的基本面哪一个更好？
            信息抽取：'公司': ['广发证券', '东方财富证券'， '东方证券'], '行业': [], '时间': [], '意图': ['基本面']

        问题：{query}
        信息抽取："""


class TimeResolveTool(BaseModel):
    def __init__(self, llm: BaseLanguageModel, verbose: bool = True):
        super().__init__(llm, verbose)
        self.name = "时间解析"
        self.description = '''
        将时间解析为规范格式
        '''
        self._template = """
        结合今天日期，解析时间中的年、月、日信息，以“['xxxx年x月x日']”的形式回复

        举例：
        问题：今天是2022年05月12日，解析该时间“2019年”。
        时间解析：['2019年']

        问题：今天是2023年02月02日，解析该时间“最近三年”。
        时间解析：['2023年', '2022年', '2021年']

        问题：今天是2024年03月02日，解析该时间“二零二二年五月”。
        时间解析：['2022年05月']

        问题：今天是2023年08月05日，解析该时间“最近三个月”。
        时间解析：['2023年08月', '2023年07月', '2023年06月']

        问题：今天是2024年01月02日，解析该时间“二〇二三年8月七号”。
        时间解析：['2023年08月07日']

        问题：今天是2024年09月02日，解析该时间“2020-7-23”。
        时间解析：['2020年07月23日']

        问题：今天是{date}，解析该时间“{query}”。
        时间解析："""

    def get_reference(self, query):
        self.reference = dict(
            date=datetime.now().strftime("%Y年%m月%d日")
        )


# class AnalysisQuery(BaseModel):

class KGRetrieveTool(BaseModel):
    def __init__(self, llm: BaseLanguageModel,  verbose: bool = True, *args, **kwargs):
        super().__init__(llm, verbose, *args, **kwargs)
        self.name = "金融投资"
        self.description = '''
        当需要查询从知识图谱中查询金融方面的实体时使用
        '''
        self._template = """
        基于以下已知内容来回答最后的问题。
        如果无法直接从中得到答案，请全面地分析已知内容。

        已知内容：
        {content}
        已知内容结束

        问题：{query}
        答案："""
        self.graph_searcher = AnswerSearcher()
        self.kwargs = kwargs
        self.args = args
        self.embeddings_func = get_embedding
        # Milvus(self.embeddings)
        self.stock_matched_threshold = 0.8
        self.intent_matched_threshold = 0.7
        self.refevrence_llm = llm
        self.vec_search_params = {"metric_type": "L2", "params": {"nprobe": 1024}}

    def renew_prompt(self, prompt):
        self._template = prompt
        self.get_llm_chain()

    # 异步调用llm
    async def async_generate(self, query, chain, reference):
        try:
            resp = await chain.apredict(
                query=query,
                **reference
            )
        except:
            try:
                resp = await chain.apredict(
                    query=query,
                    **reference
                )
            except:
                resp = ''
        return resp

    async def genereate_multi_reply_concurrently(self, query, question_analysis):
        # 以知识图谱为数据底层，以大模型为知识补充
        # 分别存储以知识图谱和语言模型作为知识支撑的答案生成调用
        data_pool = {'indicators': [], 'function_call': []}
        llm_pool = {'indicators': [], 'function_call': []}
        data_prompt = """
        请你根据以下给出的数据和问题，找出要回答这个问题可以利用哪些数据，并对已知的数据进行分析
        请你给出具体的分析结果，不用回答问题

        已知数据：
        {data}
        已知数据结束

        问题：{query}
        答案："""
        llm_prompt = """
        对于给定的问题，请你从{indicator}的角度去分析问题,只考虑与{indicator}有关的因素，不用考虑其他因素,分析字数在两百字左右

        问题：{query}
        问题分析："""
        data_chain = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(data_prompt),
                              verbose=self.verbose)
        llm_chain = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(llm_prompt),
                             verbose=self.verbose)
        answer_dict = {}
        table_question_analysis = deepcopy(question_analysis)
        data_schema_dict = GetAttributeTool(self.llm, *self.args, **self.kwargs).run(query)
        logging.info("回答问题的方案和数据库框架:{}".format(
            '\n'.join([f'{key} : {value}' for key, value in data_schema_dict.items()])))
        if not data_schema_dict:
            return ''
        for indicator, two_task_answers in data_schema_dict.items():
            attrs = two_task_answers['task_one_answer']
            db_call = two_task_answers['task_two_answer']
            if 'price_db' in db_call:
                # 行情数据分析
                stock_map = question_analysis['stock_map']
                time_list = self.parse_market_time(question_analysis['时间'])
                mk_data = ''
                start_date = min(time_list)
                end_date = max(time_list)
                mk_query = f"从{'、'.join(attrs)}的角度出发，回答{query}"
                for name, code in stock_map.items():
                    mk_data += self.get_mk_data(name, code, start_date, end_date)
                reference = dict(
                    data=mk_data
                )
                data_pool['indicators'].append(indicator)
                data_pool['function_call'].append(self.async_generate(mk_query, data_chain, reference))
            if 'indicator_db' in db_call:
                # 技术指标分析
                stock_map = question_analysis['stock_map']
                time_list = self.parse_market_time(question_analysis['时间'])
                mk_data = ''
                start_date = min(time_list)
                end_date = max(time_list)
                mk_query = f"从{'、'.join(attrs)}的角度出发，回答{query}"
                for name, code in stock_map.items():
                    mk_data += self.get_mk_indicator(name, code, start_date, end_date)
                reference = dict(
                    data=mk_data
                )
                data_pool['indicators'].append(indicator)
                data_pool['function_call'].append(self.async_generate(mk_query, data_chain, reference))

            if 'finance_db' in db_call:
                effective_kg_question_analysis = deepcopy(question_analysis)
                for attr in attrs:
                    attribute_res = self.vector_search(attr, 'attribute', ['text'])
                    if attribute_res and attribute_res[0].get('score') >= self.attr_threshold:
                        effective_kg_question_analysis['intent'].append(attribute_res[0].get('text'))
                        table_question_analysis['intent'].append(attribute_res[0].get('text'))
                if effective_kg_question_analysis['intent']:
                    # 图谱数据分析
                    kg_data = self.graph_searcher.search_main(effective_kg_question_analysis)
                    reference = dict(
                        data=kg_data
                    )
                    data_pool['indicators'].append(indicator)
                    data_pool['function_call'].append(self.async_generate(query, data_chain, reference))
                elif len(db_call) == 1:
                    reference = dict(
                        indicator=indicator
                    )
                    llm_pool['indicators'].append(indicator)
                    llm_pool['function_call'].append(self.async_generate(query, llm_chain, reference))
            if len(db_call) == 0:
                reference = dict(
                    indicator=indicator
                )
                llm_pool['indicators'].append(indicator)
                llm_pool['function_call'].append(self.async_generate(query, llm_chain, reference))

        logging.info(f"开始第二次查询图表数据: {table_question_analysis}")
        self.finance_table_to_redis(table_question_analysis)
        answers = await asyncio.gather(*(data_pool['function_call'] + llm_pool['function_call']))
        for indicator, ans in zip(data_pool['indicators'] + llm_pool['indicators'], answers):
            if indicator not in answer_dict.keys():
                answer_dict[indicator] = ans + '\n'
            else:
                answer_dict[indicator] += ans + '\n'
        return answer_dict

    def parse_market_time(self, time_list: list) -> list:
        new_time_list = []
        if not time_list:
            now = datetime.now()
            current_year_month = now.strftime('%Y-%m') + '-31'
            last_month_first_day = (now.replace(day=1) - timedelta(days=1)).replace(day=1)
            two_months_ago_first_day = (last_month_first_day.replace(day=1) - timedelta(days=1)).replace(
                day=1).strftime('%Y-%m-%d')
            new_time_list += [current_year_month, two_months_ago_first_day]
        for time in time_list:
            try:
                # 用户给到某年某月某日  精确
                ymd = datetime.strptime(time, "%Y年%m月%d日").strftime("%Y-%m-%d")
                new_time_list.append(ymd)
            except:
                try:
                    # 用户给到某年某月 模糊
                    ym = datetime.strptime(time, "%Y年%m月").strftime("%Y-%m")
                    new_time_list.append(ym + '-31')
                except:
                    try:
                        # 用户给到某年 模糊
                        y = datetime.strptime(time, "%Y年").strftime("%Y")
                        # 获取当前日期时间
                        now = datetime.now()
                        # 获取年份和月份
                        current_year = str(now.year)
                        if y == current_year:
                            current_year_month = now.strftime('%Y-%m') + '-31'
                            last_month_first_day = (now.replace(day=1) - timedelta(days=1)).replace(day=1)
                            two_months_ago_first_day = (
                                        last_month_first_day.replace(day=1) - timedelta(days=1)).replace(
                                day=1).strftime('%Y-%m-%d')
                            new_time_list += [current_year_month, two_months_ago_first_day]
                        else:
                            new_time_list += [f'{y}-12-31', f'{y}-10-01']
                    except:
                        now = datetime.now()
                        current_year_month = now.strftime('%Y-%m') + '-31'
                        last_month_first_day = (now.replace(day=1) - timedelta(days=1)).replace(day=1)
                        two_months_ago_first_day = (last_month_first_day.replace(day=1) - timedelta(days=1)).replace(
                            day=1).strftime('%Y-%m-%d')
                        new_time_list += [current_year_month, two_months_ago_first_day]
        return new_time_list

    def get_reference(self, query):
        question_dict = self.get_question_analysis(query)
        query_res = ''
        if self.verbose:
            logging.info(f"question_analysis: {question_dict}")
        if question_dict:
            query_res = self.graph_searcher.search_main(question_dict)
            # 图表数据写入redis
            # 另起线程去执行
            # logging.info("开始查询图表数据")
            # sub_thread = threading.Thread(target=self.finance_table_to_redis,
            #                               args=(question_dict,))
            # sub_thread.start()

        # self.reference = dict(
        #     content=lore
        # )
        # 返回问题第一次解析后的结果以及在图谱的查询结果
        return question_dict, query_res

    def run(self, query):
        question_dict, query_res = self.get_reference(query)
        self.get_llm_chain()
        # 如果没有识别到有效的主体
        base_prompt = """
        问题：{query}
        答案："""
        if not question_dict:
            self.renew_prompt(base_prompt)
            # 如果问题第一次解析后没有获得有效的图谱意图，则进入第二次解析，增加方案生成和属性获取的组件调用
        elif not query_res:
            question_analysis = deepcopy(question_dict)
            answer_dict = asyncio.run(self.genereate_multi_reply_concurrently(query, question_analysis))
            if answer_dict:
                llm_ans = '\n####\n'.join([f'{key}:{value}' for key, value in answer_dict.items()]) + '\n'
                new_prompt = """
                对下述问题的各角度分析进行汇总得到结论。
                详细说明分析内容如何得到结论。
                结论要全面并且有深度,字数在三百字左右。
                仅关注分析的内容，不用回答其他方面
                分析内容：
                {content}
                分析内容结束

                问题：{query}
                答案："""
                self.renew_prompt(new_prompt)
                self.reference = dict(
                    content=llm_ans,
                )
            else:
                self.renew_prompt(base_prompt)
        else:
            self.reference = dict(
                content=query_res
            )
        if self.verbose:
            logging.info(f"Tool's name is {self.name}. Its reference is {self.reference}.")
        self.progress(progress_text='调用LLM')
        resp = self.llm_chain.predict(
            query=query,
            **self.reference
        )
        return resp

    def vector_search(self, text, collection_name, output_fields):
        embeddings = self.embeddings_func(text=text)
        conn = Milvus(collection_name=collection_name).connect()
        res = conn.search(
            data=[embeddings],
            anns_field='vector',
            param=self.vec_search_params,
            limit=1,
            output_fields=output_fields
        )
        ret = []
        for result in res[0]:
            meta = {x: result.entity.get(x) for x in output_fields}
            meta['score'] = result.score
            ret.append(meta)
        return ret

    def get_question_analysis(self, query):
        ie = IETool(self.llm)
        tr = TimeResolveTool(self.llm)
        ie_res = ast.literal_eval("{*}".replace('*', ie.run(query)))
        question_dict = {
            '主体': {
                '股票': []
            },
            '时间': [],
            '意图': []}
        for extract_stock in ie_res["公司"]:
            matched_stock_list = self.get_kg_matched_subject(subject=extract_stock,
                                                             match_type='stock',
                                                             subject_type='股票')

            question_dict['主体']['股票'] += matched_stock_list
        for extract_time in ie_res["时间"]:
            process_time = ast.literal_eval(tr.run(extract_time))
            question_dict['时间'] += process_time
        for extract_intent in ie_res["意图"]:
            matched_attr_list = self.get_kg_matched_subject(subject=extract_intent,
                                                            match_type='intent',
                                                            subject_type='属性')
            matched_ent_list = self.get_kg_matched_subject(subject=extract_intent,
                                                           match_type='intent',
                                                           subject_type='实体')
            question_dict['意图'] += matched_attr_list + matched_ent_list
        return question_dict

    # bleu 和 编辑距离 用于计算相似度
    @staticmethod
    def bleu(ner, ent):
        """计算抽取实体与现有实体的匹配度 ner候选词, ent查询词"""
        len_pred, len_label = len(ner), len(ent)
        k = min(len_pred, len_label)
        if k == 0:
            return 0
        score = math.exp(min(0, int(1 - len_label / len_pred)))
        # score = 0
        flag = False
        for n in range(1, k + 1):
            num_matches, label_subs = 0, collections.defaultdict(int)
            for i in range(len_label - n + 1):
                label_subs[" ".join(ent[i: i + n])] += 1
            for i in range(len_pred - n + 1):
                if label_subs[" ".join(ner[i: i + n])] > 0:
                    num_matches += 1
                    flag = True
                    label_subs[" ".join(ner[i: i + n])] -= 1  # 不重复
            if not flag and num_matches == 0:  # 一次都没匹配成功
                score = 0
                break
            elif num_matches == 0:  # 进行到最大匹配后不再计算
                break
            score *= math.pow(num_matches / (len_pred -
                                             n + 1), math.pow(0.5, n))
        return score if score > 0 else 0

    @staticmethod
    def editing_distance(word1, word2):
        try:
            m, n = len(word1), len(word2)
        except:
            return float('inf')

        if m == 0 or n == 0:
            return abs(m - n)
        dp = [[float('inf') for _ in range(n + 1)] for _ in range(m + 1)]

        for i in range(m):
            dp[i][0] = i

        for i in range(n):
            dp[0][i] = i

        for i in range(1, m + 1):
            for j in range(1, n + 1):

                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    # 替换
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    # 删除
                    dp[i][j] = min(dp[i][j], min(dp[i - 1][j], dp[i][j - 1]) + 1)
        return dp[-1][-1]

    @staticmethod
    def equal(subject_1: str, subject_2: str) -> bool:
        if subject_2 in subject_1:
            return True
        return False

    def get_kg_matched_subject(self, subject, match_type, subject_type):
        '''

        更新basic_ent[subject]
        '''
        scores_best = float('inf')
        equal_flag = False
        most_similar_subject = ''
        matched_subject_list = []
        match_threshold = self.stock_matched_threshold if match_type == 'stock' else self.intent_matched_threshold
        for kg_subject in self.graph_searcher.knowledge[subject_type]:
            if self.equal(subject, kg_subject):
                matched_subject_list.append(kg_subject)
                equal_flag = True
            scores_cur = self.editing_distance(kg_subject, subject)
            if scores_cur < scores_best:
                most_similar_subject = kg_subject
                scores_best = scores_cur
        if not equal_flag and self.bleu(subject, most_similar_subject) >= match_threshold:
            matched_subject_list.append(most_similar_subject)
        return matched_subject_list

    def get_mk_data(self, name, symbol, start_date, end_date):
        # 创建数据库连接
        conn = pymysql.connect(host='192.168.1.101', port=13306, user='root', password='km101', database='market_data')
        cursor = conn.cursor()

        # 执行sql语句
        sql = 'SELECT trade_date,low,high,close,volume FROM  a_market_data_day_K WHERE symbol = %s and ' \
              'trade_date >= %s and trade_date <= %s ORDER BY trade_date DESC'
        cursor.execute(sql, [int(symbol), start_date, end_date])
        rows = cursor.fetchall()
        conn.close()

        # 将查询结果存入字典
        keys = ['交易日', '最低价', '最高价', '收盘价', '成交量']
        df = pd.DataFrame(list(rows), columns=keys).set_index('交易日', drop=True)
        df[['最低价', '最高价', '收盘价']] = df[['最低价', '最高价', '收盘价']].astype(float).round(2)
        df['成交量'] = df['成交量'].astype(float).apply(lambda x: "{:.0f}".format(x))
        df['涨跌幅'] = (df['收盘价'] / df['收盘价'].shift(-1) - 1).apply(lambda x: "{:.2%}".format(x))
        return f'{name}从{start_date}到{end_date}的行情信息如下: \n {df.to_markdown()}'

    def get_mk_indicator(self, name, symbol, start_date, end_date):
        # 创建数据库连接
        conn = pymysql.connect(host='192.168.1.101', port=13306, user='root', password='km101', database='market_data')
        cursor = conn.cursor()

        # 执行sql语句
        sql = 'SELECT trade_date, MA5, BBANDS_upperband,BBANDS_lowerband, RSI, MACD FROM  talib_indicator_data WHERE symbol = %s and trade_date >= %s and trade_date <= %s ORDER BY trade_date DESC LIMIT 50'
        cursor.execute(sql, [int(symbol), start_date, end_date])
        rows = cursor.fetchall()
        # 结束数据库连接
        conn.close()

        # 将查询结果存入字典
        keys = ['交易日', '5日均线(布林带中轨)', '布林带上轨', '布林带下轨', 'RSI', 'MACD']
        df = pd.DataFrame(list(rows), columns=keys).set_index('交易日', drop=True)
        df[['5日均线(布林带中轨)', '布林带上轨', '布林带下轨', 'RSI', 'MACD']] = df[
            ['5日均线(布林带中轨)', '布林带上轨', '布林带下轨', 'RSI', 'MACD']].astype(float)
        df = df.applymap(lambda x: "{:.2f}".format(x) if isinstance(x, (int, float)) else x)
        return f'{name}从{start_date}到{end_date}的技术指标如下: \n {df.to_markdown()}'


class CreateSchemeTool(BaseModel):
    def __init__(self, llm: BaseLanguageModel, verbose: bool = True):
        super().__init__(llm, verbose)
        self.name = "方案生成"
        self.description = '''
        根据问题生成方案
        '''
        self._template = """
        根据问题分析出回答该问题所需要涉及的不同角度，尽可能回答可以量化的角度,并以列表的形式回复（不需要包含主体的名字），角度不超过四个，并且不同角度之间的相关性要比较低。

        举例：
        问题：葛洲坝是否值得投资？
        方案生成：['*','*','*']

        问题：人工智能行业发展前景如何？
        方案生成：['*','*','*']

        问题：分众传媒和新潮传媒哪个更好？
        方案生成：['*','*','*']

        问题：{query}
        方案生成："""


class GetAttributeTool(BaseModel):
    def __init__(self, llm: BaseLanguageModel, verbose: bool = True, *args, **kwargs):
        super().__init__(llm, verbose, *args, **kwargs)
        self.name = "关注指标"
        self.description = '''
        根据问题和方案生成具体指标
        '''
        self._template = """
        请按照指令完成任务
        任务一：
        根据给出的问题和问题分析因素,回答在当前分析因素下具体需要关注的指标,尽可能回答可以量化的指标,并以列表的形式回复。
        任务二：
        我有三个数据库，存储的数据内容如下：
        finance_db:股票的财务信息
        price_db:股票的行情信息
        indicator_db:股票的技术指标信息
        请你根据任务一中得到的关注指标分析需要从哪些数据库去访问这些指标，并返回数据库的名称，如果数据库中的数据与关注指标关联度不高请返回空列表

        举例：
        问题：葛洲坝是否值得投资？
        分析因素：财务表现
        任务一答案: ['*','*','*',]
        任务二答案: ['finance_db']
        
        问题：葛洲坝是否值得投资？
        分析因素：价格走势
        任务一答案: ['*','*','*',]
        任务二答案: ['price_db']

        问题：葛洲坝是否值得投资？
        分析因素：盈利能力
        任务一答案: ['*','*','*',]
        任务二答案: ['finance_db','price_db']

        问题：葛洲坝是否值得投资？
        分析因素：竞争对手
        任务一答案: ['*','*','*',]
        任务二答案: []
        
        问题：{query}
        分析因素：{indicator}
        任务一答案: ？
        任务二答案: ？"""

    def extract_answers(self, input_string):
        # 使用正则表达式提取任务一和任务二的答案
        pattern_task_one = r"任务一答案: (\[.*?\])"
        pattern_task_two = r"任务二答案: (\[.*?\])"

        # 使用正则表达式匹配
        match_task_one = re.search(pattern_task_one, input_string)
        match_task_two = re.search(pattern_task_two, input_string)

        # 提取匹配的内容
        task_one_answer = eval(match_task_one.group(1)) if match_task_one else None
        task_two_answer = eval(match_task_two.group(1)) if match_task_two else None

        return task_one_answer, task_two_answer

    def run(self, query):
        self.get_llm_chain()
        scheme_tool = CreateSchemeTool(self.llm)
        try:
            scheme = scheme_tool.run(query)
            indicators = ast.literal_eval(scheme)
        except:
            logging.info(f"生成的方案不是列表形式，无法被解析:{scheme}")
            return ''
        response = {}
        for indicator in indicators:
            self.progress(progress_text=f'正在分析 {indicator}')
            self.reference = dict(
                indicator=indicator
            )
            if self.verbose:
                logging.info(f"Tool's name is {self.name}. Its reference is {self.reference}.")
            resp = self.llm_chain.predict(
                query=query,
                **self.reference
            )
            logging.info(f"抽出来的指标:{resp}")
            try:
                task_one_ans, task_two_ans = self.extract_answers(resp)
                response[indicator] = dict(
                    task_one_answer=task_one_ans,
                    task_two_answer=task_two_ans
                )
            except Exception as e:
                logging.info(e)
                logging.info(f"生成的指标不是列表形式，无法被解析:{resp}")
                return ''
        return response


def handle_answer(text: str):
    """
    过滤掉一些回答
    """
    text = re.sub(r"ChatGPT", "", text)
    text = re.sub(r"OpenAI", "", text)
    text = re.sub(r"GPT-3\.5", "", text)
    return text


if __name__ == "__main__":
    llm = ChatGLM3(
        endpoint_url=SERVER_API_URL + SERVER_LLM_API_PORT + CHAT_API_URL,
        max_tokens=8096,
        top_p=0.9
    )
    tools = [KGRetrieveTool(llm), BaseModel(llm)]
    agent = IntentAgent(llm=llm, tools=tools)
    query = "李白写过哪些诗"
    tool = agent.choose_tools(query)
    print(tool.name)
    print(tool.run(query))

    query = "后天是周末吗"
    tool = agent.choose_tools(query)
    print(tool.name)
    print(tool.run(query))

    # query = "分众传媒的实际控制人是谁"
    # tool = agent.choose_tools(query)
    # print(tool.name)
    # print(tool.run(query))
