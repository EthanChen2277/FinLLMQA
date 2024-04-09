import ast
import collections
import math
import re
from abc import ABC
import logging
import pymysql
from datetime import datetime, timedelta
import pandas as pd
import asyncio
from copy import deepcopy

from langchain.base_language import BaseLanguageModel
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain

from finllmqa.kg.search import AnswerSearcher
from finllmqa.api.embedding import get_embedding
from finllmqa.api.core import LLM_API_URL
from finllmqa.vector_db.construct import Milvus

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LangChainTool(ABC):
    def __init__(self, llm: BaseLanguageModel = None, verbose: bool = True, *args, **kwargs):
        if llm is None:
            self.llm = ChatOpenAI(model="chatglm3",
                                  max_tokens=2048,
                                  base_url=LLM_API_URL,
                                  api_key='null')
        else:
            self.llm = llm
        self.llm_chain = None
        self.name = "其它"
        self.description = '''
        无需任何信息，直接回答
        '''
        self._template = """
        问题：{query}
        答案："""
        self.prompt_template = None
        self.prompt_str = None
        self.reference = None
        self.verbose = verbose
        self.progress_func = kwargs.get('progress_func')
        # self.progress(progress_text='开始分析问题')

    def get_prompt_template(self):
        self.prompt_template = PromptTemplate.from_template(self._template)

    def get_str_prompt(self):
        self.get_prompt_template()
        self.prompt_str = self.prompt_template.format(**self.reference)

    def get_reference(self, **prompt_kwargs):
        self.reference = prompt_kwargs

    def get_llm_chain(self):
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt_template, verbose=self.verbose)

    def get_chat_llm_chain(self):
        chat_prompt = ChatPromptTemplate.from_messages([('human', self.prompt_str)])
        self.llm_chain = chat_prompt | self.llm

    def progress(self, progress_text):
        """
        问题处理进度
        """
        logging.debug(f'问题处理进度：{progress_text}')
        if self.progress_func:
            self.progress_func(progress_text)

    def run(self, **prompt_kwargs):
        if self.reference is None:
            self.get_reference(**prompt_kwargs)
        self.get_prompt_template()
        self.get_llm_chain()
        self.progress(progress_text='调用LLM')
        query = prompt_kwargs.get('query', None)
        if 'query' not in self.reference.keys():
            assert query is not None, 'query must be given when not included in self reference'
            self.reference.update({'query': query})
        resp = self.llm_chain.predict(
            **self.reference
        )
        if self.verbose:
            logging.info(f"Tool's name is {self.name}. Its reference is {self.reference}\n "
                         f"Q: {query} LLM Answer: {resp}.")
        return resp

    # def send_query_to_autogen(self):
    #     query = self.prompt
    #     response = get_autogen_stream_answer(query=query)
    #     logging.info(f'query: {query} send_query_to_autogen result: {response}')

    # 异步调用组件
    def get_stream_response(self, **prompt_kwargs):
        # 将回答的问题开启流式输出
        self.llm.streaming = True
        if self.reference is None:
            self.get_reference(**prompt_kwargs)
        query = prompt_kwargs.get('query', None)
        if 'query' not in self.reference.keys():
            assert query is not None, 'query must be given when not included in self reference'
            self.reference.update({'query': query})
        self.get_str_prompt()
        self.get_chat_llm_chain()
        if self.verbose:
            logging.info(f"Tool's name is {self.name}. Its reference is {self.reference}.")
        self.progress(progress_text='调用LLM')
        chunks = self.llm_chain.stream({})
        return chunks


class IETool(LangChainTool):
    def __init__(self, llm: BaseLanguageModel = None, verbose: bool = True, *args, **kwargs):
        super().__init__(llm, verbose, *args, **kwargs)
        self.name = "金融信息抽取"
        self.description = '''
        当需要从问题中抽取主体信息时使用
        '''
        self._template = """
        你是一名金融信息抽取员，需要从问题中抽取出["公司", "行业", "时间", "意图"]，抽取的内容必须是问题中的字段，不能乱编，
        并且必须以'*公司*: [], *行业*: [], *时间*: [], *意图*: []'的形式回复。

        举例1：
            问题：2019年药明康德衍生金融资产和其他非流动金融资产分别是多少元?
            信息抽取：*公司*: [药明康德], *行业*: [], *时间*: [2019年], *意图*: [衍生金融资产, 其他非流动金融资产]

        举例2：
            问题：上海中谷物流股份有限公司2020年营业收入是多少元?
            信息抽取：*公司*: [上海中谷物流股份有限公司], *行业*: [], *时间*: [2020年], *意图*: [营业收入]

        举例3：
            问题：激光行业近三年的发展如何？
            信息抽取：*公司*: [], *行业*: [激光行业], *时间*: [最近三年], *意图*: [发展]

        举例4：
            问题：证券行业最近的发展如何？
            信息抽取：*公司*: [], *行业*: [证券行业], *时间*: [最近一年], *意图*: [发展]

        举例5：
            问题：分众传媒与新潮传媒谁的利润率更高？
            信息抽取：*公司*: [分众传媒, 新潮传媒], *行业*: [], *时间*: [], *意图*: [利润率]

        举例6：
            问题：广发证券 东方财富证券 东方证券的基本面哪一个更好？
            信息抽取：*公司*: [广发证券, 东方财富证券， 东方证券], *行业*: [], *时间*: [], *意图*: [基本面]

        问题：{query}
        信息抽取："""

    @staticmethod
    def process_information_extraction_result(text: str):
        # 使用正则表达式匹配并提取信息
        match = re.search(r"\*公司\*: \[(.*?)\], \*行业\*: \[(.*?)\], \*时间\*: \[(.*?)\], \*意图\*: \[(.*?)\]", text)
        if match:
            company = match.group(1)
            industry = match.group(2)
            time = match.group(3)
            intent = match.group(4)

            ie_dict = {
                '公司': [company],
                '行业': [industry],
                '时间': [time],
                '意图': [intent]
            }
        else:
            ie_dict = {}
        return ie_dict


class TimeResolveTool(LangChainTool):
    def __init__(self, llm: BaseLanguageModel = None, verbose: bool = True):
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
            query=query,
            date=datetime.now().strftime("%Y年%m月%d日")
        )


class KGRetrieverTool(LangChainTool):
    def __init__(self, llm: BaseLanguageModel = None, verbose: bool = True, *args, **kwargs):
        super().__init__(llm, verbose, *args, **kwargs)
        self.name = "知识图谱检索"
        self.description = '''
        用于检索知识图谱数据
        '''
        # self._template = """
        # 基于以下已知内容来回答最后的问题。
        # 如果无法直接从中得到答案，请全面地分析已知内容。
        #
        # 已知内容：
        # {content}
        # 已知内容结束
        #
        # 问题：{query}
        # 答案："""
        self.graph_searcher = AnswerSearcher()
        self.kwargs = kwargs
        self.args = args
        self.embeddings_func = get_embedding
        # Milvus(self.embeddings)
        self.stock_matched_threshold = 0.8
        self.intent_matched_threshold = 0.7
        self.vec_search_params = {"metric_type": "L2", "params": {"nprobe": 1024}}

    def get_kg_query_result(self, ent_dict):
        logging.info(f'start querying kg with ent_dict: {ent_dict}')
        query_res = self.graph_searcher.search_main(ent_dict=ent_dict)
        return query_res

    # 异步调用autogen问答
    async def generate_multi_reply_concurrently(self, query, question_analysis):
        # 以知识图谱为数据底层，以大模型为知识补充
        # 分别存储以知识图谱和语言模型作为知识支撑的答案生成调用
        data_pool = {'angles': [], 'function_call': []}
        llm_pool = {'angles': [], 'function_call': []}
        data_prompt = """
        请你根据以下给出的数据和问题，找出要回答这个问题可以利用哪些数据，并对已知的数据进行分析
        请你给出具体的分析结果，不用回答问题

        已知数据：
        {data}
        已知数据结束

        问题：{query}
        答案："""
        llm_prompt = """
        对于给定的问题，请你从{angle}的角度去分析问题,只考虑与{angle}有关的因素，不用考虑其他因素,分析字数在两百字左右

        问题：{query}
        问题分析："""
        data_chain = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(data_prompt),
                              verbose=self.verbose)
        llm_chain = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(llm_prompt),
                             verbose=self.verbose)
        answer_dict = {}
        table_question_analysis = deepcopy(question_analysis)
        data_schema_dict = GetAttributeTool(self.llm).run(query=query)
        logging.info("回答问题的方案和数据库框架:{}".format(
            '\n'.join([f'{key} : {value}' for key, value in data_schema_dict.items()])))
        if not data_schema_dict:
            return ''
        for angle, two_task_answers in data_schema_dict.items():
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
                data_pool['angles'].append(angle)
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
                data_pool['angles'].append(angle)
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
                    data_pool['angles'].append(angle)
                    data_pool['function_call'].append(self.async_generate(query, data_chain, reference))
                elif len(db_call) == 1:
                    reference = dict(
                        angle=angle
                    )
                    llm_pool['angles'].append(angle)
                    llm_pool['function_call'].append(self.async_generate(query, llm_chain, reference))
            if len(db_call) == 0:
                reference = dict(
                    angle=angle
                )
                llm_pool['angles'].append(angle)
                llm_pool['function_call'].append(self.async_generate(query, llm_chain, reference))

        logging.info(f"开始第二次查询图表数据: {table_question_analysis}")
        self.finance_table_to_redis(table_question_analysis)
        answers = await asyncio.gather(*(data_pool['function_call'] + llm_pool['function_call']))
        for angle, ans in zip(data_pool['angles'] + llm_pool['angles'], answers):
            if angle not in answer_dict.keys():
                answer_dict[angle] = ans + '\n'
            else:
                answer_dict[angle] += ans + '\n'
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
        ie_res = ie.run(query=query)
        ie_res = ie.process_information_extraction_result(text=ie_res)
        logging.info(f'processed ie result: {ie_res}')
        if not ie_res:
            return None
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
            if len(matched_stock_list) == 0:
                return None
            question_dict['主体']['股票'] += matched_stock_list
        for extract_time in ie_res["时间"]:
            process_time = ast.literal_eval(tr.run(query=extract_time))
            question_dict['时间'] += process_time
        for extract_intent in ie_res["意图"]:
            matched_attr_list = self.get_kg_matched_subject(subject=extract_intent,
                                                            match_type='intent',
                                                            subject_type='属性')
            matched_ent_list = self.get_kg_matched_subject(subject=extract_intent,
                                                           match_type='intent',
                                                           subject_type='实体')
            question_dict['意图'] += matched_attr_list + matched_ent_list
        if self.verbose:
            logging.info(f'question extraction dict: {question_dict}')
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
        except Exception:
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


class CreateSchemeTool(LangChainTool):
    def __init__(self, llm: BaseLanguageModel = None, verbose: bool = True):
        super().__init__(llm, verbose)
        self.name = "方案生成"
        self.description = '''
        根据问题生成方案
        '''
        self._template = """
        根据问题分析出回答该问题所需要涉及的不同角度，尽可能回答可以量化的角度,并以列表的形式回复（不需要包含主体的名字），角度不超过四个，并且不同角度之间的相关性要比较低。

        举例：
        问题：葛洲坝是否值得投资？
        方案生成：['财务分析','行业地位分析','风险评估']

        问题：A股最近表现如何？
        方案生成：['市场表现指标','宏观经济因素','行业动态', '投资者情绪']

        问题：分众传媒和新潮传媒哪个更好？
        方案生成：['市场地位和品牌影响力','财务状况和盈利能力','业务模式和创新能力']

        问题：{query}
        方案生成："""


class GetAttributeTool(LangChainTool):
    def __init__(self, llm: BaseLanguageModel = None, verbose: bool = True, *args, **kwargs):
        super().__init__(llm, verbose, *args, **kwargs)
        self.name = "关注指标"
        self.description = '''
        根据问题和方案生成具体指标
        '''
        self._template = """
        根据给出的问题和问题分析因素,回答在当前分析因素下具体需要关注的指标,尽可能回答可以量化的指标,并以列表的形式回复。

        举例：
        问题：葛洲坝是否值得投资？
        分析因素：财务表现
        关注指标: ['资产负债率', '净利润', '营业总收入'] 
        
        问题：葛洲坝是否值得投资？
        分析因素：价格走势
        关注指标: ['五日均线', '二十日均线', '布林带指标']

        问题：葛洲坝是否值得投资？
        分析因素：盈利能力
        关注指标: ['净资产收益率','总资产净利率','毛利率']

        问题：葛洲坝是否值得投资？
        分析因素：竞争对手
        关注指标: []
        
        问题：{query}
        分析因素：{angle}
        关注指标: """

    def run(self, query):
        self.get_prompt_template()
        self.get_llm_chain()
        scheme_tool = CreateSchemeTool(self.llm)
        scheme = scheme_tool.run(query=query)
        try:
            angle_list = ast.literal_eval(scheme)
        except:
            logging.info(f"生成的方案不是列表形式，无法被解析:{scheme}")
            return ''
        response = {}
        for angle in angle_list:
            self.progress(progress_text=f'正在分析 {angle}')
            self.reference = dict(
                angle=angle
            )
            if self.verbose:
                logging.info(f"Tool's name is {self.name}. Its reference is {self.reference}.")
            resp = self.llm_chain.predict(
                query=query,
                **self.reference
            )
            logging.info(f"抽出来的指标:{resp}")
            try:
                indicator_list = ast.literal_eval(resp)
                response[angle] = indicator_list
            except Exception as e:
                logging.info(e)
                logging.info(f"生成的指标不是列表形式，无法被解析:{resp}")
                return ''
        return response


class KnowledgeAnalysisTool(LangChainTool):
    def __init__(self, llm: BaseLanguageModel = None, verbose: bool = True):
        super().__init__(llm, verbose)
        self.name = "数据分析"
        self.description = '''
        根据已知数据分析问题
        '''
        self._template = """
        请你根据以下给出的数据和问题，对已知的数据进行深度分析并回答问题。

        已知数据：
        {data}
        已知数据结束

        问题：{query}
        答案："""
        self.angle = ''


class PretrainInferenceTool(LangChainTool):
    def __init__(self, llm: BaseLanguageModel = None, verbose: bool = True):
        super().__init__(llm, verbose)
        self.name = "预训练推理"
        self.description = '''
        根据预训练模型进行推理
        '''
        self._template = """
        对于给定的问题，请你从{angle}的角度去分析问题,只考虑与{angle}有关的因素，不用考虑其他因素,分析字数在两百字左右

        问题：{query}
        问题分析："""
        self.angle = ''


class SummarizeTool(LangChainTool):
    def __init__(self, llm: BaseLanguageModel = None, verbose: bool = True):
        super().__init__(llm, verbose)
        self.name = "总结答案"
        self.description = '''
        总结问题不同角度的答案
        '''
        self._template = """
        对于给定的问题以及从不同角度进行分析的答案，请你对所有的答案给出总结

        问题：{query}
        不同角度答案：{total_answer}
        总结："""


def handle_answer(text: str):
    """
    过滤掉一些回答
    """
    text = re.sub(r"ChatGPT", "", text)
    text = re.sub(r"OpenAI", "", text)
    text = re.sub(r"GPT-3\.5", "", text)
    return text


# if __name__ == "__main__":
    # llm = ChatGLM3(
    #     endpoint_url=CHAT_API_URL,
    #     max_tokens=8096,
    #     top_p=0.9
    # )
    # tools = [KGRetrieveTool(llm), LangChainTool(llm)]
    # agent = IntentAgent(llm=llm, tools=tools)
    # query = "李白写过哪些诗"
    # tool = agent.choose_tools(query)
    # print(tool.name)
    # print(tool.run(query))
    #
    # query = "后天是周末吗"
    # tool = agent.choose_tools(query)
    # print(tool.name)
    # print(tool.run(query))

    # query = "分众传媒的实际控制人是谁"
    # tool = agent.choose_tools(query)
    # print(tool.name)
    # print(tool.run(query))
