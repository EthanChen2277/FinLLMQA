import ast
import collections
import math
import re
from abc import ABC
import logging
from datetime import datetime

from langchain.base_language import BaseLanguageModel
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
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
        self.chat_messages = None
        self.reference = None
        self.angle = '原问题'
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
        chat_prompt = ChatPromptTemplate.from_messages(self.chat_messages or [('human', self.prompt_str)])
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
        history = prompt_kwargs.get('history', None)
        if history is None and self.chat_messages is None:
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

    def get_kg_schema(self):
        # 查询节点标签
        node_labels_query = """
        CALL db.labels()
        """
        # 查询关系类型（关系标签）
        relationship_types_query = """
        CALL db.relationshipTypes()
        """
        # 执行查询获取节点标签
        node_labels = self.graph_searcher.g.run(node_labels_query).to_series().tolist()
        # 执行查询获取关系类型（关系标签）
        relationship_types = self.graph_searcher.g.run(relationship_types_query).to_series().tolist()
        # 构造图谱结构和可视化描述
        output = "图谱结构和可视化描述：\n"
        # 输出节点标签
        output += "节点标签：\n"
        for label in node_labels:
            output += f"- {label}\n"
        # 输出关系类型（关系标签）
        output += "\n关系类型（关系标签）：\n"
        for rel_type in relationship_types:
            output += f"- {rel_type}\n"
        # 输出图谱结构
        output += "\n图谱结构：\n"
        output += "节点标签 -> 关系类型 -> 节点标签\n"
        # 查询节点和关系的连接
        for label in node_labels:
            for rel_type in relationship_types:
                query = f"""
                MATCH (n:`{label}`)-[r:`{rel_type}`]->(m)
                RETURN DISTINCT labels(n) AS start_labels, type(r) AS relationship_type, labels(m) AS end_labels
                """
                result = self.graph_searcher.g.run(query)
                for record in result:
                    start_labels = ", ".join(record['start_labels'])
                    relationship_type = record['relationship_type']
                    end_labels = ", ".join(record['end_labels'])
                    output += f"- {start_labels} -> {relationship_type} -> {end_labels}\n"
        return output

    def get_kg_query_result(self, ent_dict, _type: str = 'llm'):
        logging.info(f'start querying kg with ent_dict: {ent_dict} in type {_type}')
        query_res = self.graph_searcher.search_main(ent_dict=ent_dict, _type=_type)
        return query_res

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
        question_dict['主体']['股票'] = list(set(question_dict['主体']['股票']))
        for extract_time in ie_res["时间"]:
            if extract_time:
                process_time = ast.literal_eval(tr.run(query=extract_time))
                question_dict['时间'] += process_time
        question_dict['时间'] = list(set(question_dict['时间']))
        for extract_intent in ie_res["意图"]:
            matched_attr_list = self.get_kg_matched_subject(subject=extract_intent,
                                                            match_type='intent',
                                                            subject_type='属性')
            matched_ent_list = self.get_kg_matched_subject(subject=extract_intent,
                                                           match_type='intent',
                                                           subject_type='实体')
            question_dict['意图'] += matched_attr_list + matched_ent_list
        question_dict['意图'] = list(set(question_dict['意图']))
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
    def include(subject_1: str, subject_2: str) -> bool:
        if subject_2 in subject_1 or subject_1 in subject_2:
            return True
        return False

    def get_kg_matched_subject(self, subject, match_type, subject_type):
        """

        更新basic_ent[subject]
        """
        scores_best = float('inf')
        include_flag = False
        most_similar_subject = ''
        matched_subject_list = []
        match_threshold = self.stock_matched_threshold if match_type == 'stock' else self.intent_matched_threshold
        for kg_subject in self.graph_searcher.knowledge[subject_type]:
            scores_cur = self.editing_distance(kg_subject, subject)
            if include_flag:
                if self.include(subject, kg_subject):
                    if scores_cur < scores_best:
                        matched_subject_list = [kg_subject]
                else:
                    continue
            else:
                if self.include(subject, kg_subject):
                    matched_subject_list = [kg_subject]
                    scores_best = scores_cur
                    include_flag = True
                else:
                    if scores_cur < scores_best:
                        most_similar_subject = kg_subject
                        scores_best = scores_cur
        if not include_flag and self.bleu(subject, most_similar_subject) >= match_threshold:
            matched_subject_list.append(most_similar_subject)
        matched_subject_list = list(set(matched_subject_list))
        return matched_subject_list


class CreateSchemeTool(LangChainTool):
    def __init__(self, llm: BaseLanguageModel = None, verbose: bool = True):
        super().__init__(llm, verbose)
        self.name = "方案生成"
        self.description = '''
        根据问题生成方案
        '''
        self._template = """
        你是一名股票投资方面的专家，对于股票投资类问题，你需要分析出回答该问题可能需要涉及哪些基本面和行情数据，并以列表的形式回复，
        不同分析角度之间的相关性要低，分析角度不要超过三个。
        
        举例：
        问题：贵州茅台是否值得投资？
        方案生成：['盈利指标','估值','价格走势']

        问题：分众传媒和新潮传媒哪个更好？
        方案生成：['盈利指标','风险指标','价值指标']

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
        你是一名股票投资方面的专家，根据给出的股票投资类问题和问题分析角度,回答在当前分析角度下需要关注哪些意图指标,
        并以列表的形式回复。

        举例：
        问题：贵州茅台是否值得投资？
        分析角度：盈利指标
        意图指标: ['净利润', '每股净利润', '营业收入']
        
        问题：葛洲坝是否值得投资？
        分析因素：价格走势
        关注指标: ['MA5', 'MA20', 'RSI']

        问题：葛洲坝是否值得投资？
        分析因素：盈利能力
        关注指标: ['净资产收益率','总资产净利率','毛利率']
        
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
                continue
        return response or ''


class KnowledgeAnalysisTool(LangChainTool):
    def __init__(self, llm: BaseLanguageModel = None, verbose: bool = True):
        super().__init__(llm, verbose)
        self.name = "数据分析"
        self.description = '''
        你是一名专业的股票投资顾问, 请根据已知数据分析问题
        '''
        self._template = """
        请你根据以下给出的数据和问题，对已知的数据进行深度分析并回答问题。
        分析字数在500字左右。

        已知数据：
        {data}
        已知数据结束

        问题：{query}
        答案："""
        self.angle = ''
        self.question_intent = {}
        self.display_table = None


class PretrainInferenceTool(LangChainTool):
    def __init__(self, llm: BaseLanguageModel = None, verbose: bool = True):
        super().__init__(llm, verbose)
        self.name = "预训练推理"
        self.description = '''
        你是一名专业的股票投资顾问
        '''
        self._template = """
        你是一名专业的股票投资顾问
        对于给定的股票投资问题，请你从{angle}的角度去分析问题,只考虑与{angle}有关的因素，不用考虑其他因素, 分析字数在500字左右

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
