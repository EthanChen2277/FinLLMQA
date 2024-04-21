import logging
from copy import deepcopy
from langchain_core.language_models import BaseLanguageModel

from finllmqa.agent.llama_index_tools import RAGQueryEngineTool
from finllmqa.agent.langchain_tools import KGRetrieverTool, KnowledgeAnalysisTool, GetAttributeTool, \
    PretrainInferenceTool, LangChainTool


class FinInvestmentQA(LangChainTool):
    def __init__(self, llm: BaseLanguageModel = None, verbose: bool = True, *args, **kwargs):
        super().__init__(llm, verbose, *args, **kwargs)
        self.name = "股票投资"
        self.description = '''
        股票投资类问答Agent
        '''
        self._template = """
        你是一个专业的股票投资顾问，请你回答以下问题

        问题：{query}"""
        self.angle = '原问题'
        # 分别存储以知识图谱和语言模型作为知识支撑的答案生成调用
        self.knowledge_analysis_pool = []
        self.pretrain_inference_pool = []
        self.kg_retriever = KGRetrieverTool()
        self.fail_matched_intent_dc = {}

    def get_reference(self, query):
        question_dict = self.kg_retriever.get_question_analysis(query)
        if question_dict:
            if len(question_dict['意图']) > 0:
                query_res = self.kg_retriever.get_kg_query_result(ent_dict=question_dict)
                ka_tool = KnowledgeAnalysisTool()
                ka_tool.reference = {
                    'data': query_res,
                    'query': query
                    }
                ka_tool.get_str_prompt()
                ka_tool.question_intent = question_dict
                ka_tool.angle = '原问题'
                self.knowledge_analysis_pool.append(ka_tool)
        else:
            self.reference = None
            return False
        angel_intent_dict = GetAttributeTool(self.llm).run(query)
        if not angel_intent_dict:
            self.reference = None
            return False
        for angle, intent_ls in angel_intent_dict.items():
            new_question_dict = deepcopy(question_dict)
            for intent in intent_ls:
                matched_attr_list = self.kg_retriever.get_kg_matched_subject(subject=intent,
                                                                             match_type='intent',
                                                                             subject_type='属性')
                matched_ent_list = self.kg_retriever.get_kg_matched_subject(subject=intent,
                                                                            match_type='intent',
                                                                            subject_type='实体')
                matched_intent_list = matched_attr_list + matched_ent_list
                if len(matched_intent_list) == 0:
                    self.fail_matched_intent_dc[angle] = intent
                new_question_dict['意图'] += matched_intent_list
            if len(new_question_dict['意图']) > 0:
                query_res = self.kg_retriever.get_kg_query_result(new_question_dict)
                if query_res:
                    ka_tool = KnowledgeAnalysisTool()
                    ka_tool.reference = {
                        'data': query_res,
                        'query': query
                    }
                    ka_tool.get_str_prompt()
                    ka_tool.angle = angle
                    ka_tool.question_intent = new_question_dict
                    self.knowledge_analysis_pool.append(ka_tool)
                    continue
            pi_tool = PretrainInferenceTool()
            pi_tool.reference = {
                'angle': angle,
                'query': query
            }
            pi_tool.get_str_prompt()
            pi_tool.angle = angle
            self.pretrain_inference_pool.append(pi_tool)
        return True

    def run(self, query):
        kg_matched_flag = self.get_reference(query=query)
        if not kg_matched_flag:
            self.reference = {
                'query': query
            }
            self.get_str_prompt()
        return kg_matched_flag


class FinKnowledgeQA(LangChainTool):
    def __init__(self, llm: BaseLanguageModel = None, verbose: bool = True):
        super().__init__(llm, verbose)
        self.name = "财经百科"
        self.description = '''
        财经百科类问答Agent
        '''
        self._template = """
        你是一个财经百科专家，请你回答以下问题

        问题：{query}"""
        self.angle = '原问题'
        self.query_engine = RAGQueryEngineTool()

    def query(self, query, retrieve_mode, response_mode):
        self.query_engine.get_retriever(mode=retrieve_mode)
        response = self.query_engine.run(text=query, response_mode=response_mode)
        return response


class IntentAgent(LangChainTool):
    def __init__(self, llm: BaseLanguageModel = None, verbose: bool = True):
        super().__init__(llm, verbose)
        self.name = "问题分类"
        self.description = '''
        识别问题意图，选择相应的问答代理
        '''
        self.progress(progress_text='开始对问题分类')

        self._template = """
        理解用户问题的意图，并判断该问题的类别属于股票投资还是财经问答，按格式回复：'问题类别：<>'。
        股票投资类问题指与股票投资相关的问题，对应股票的基本面和行情数据，
        财经问答类问题指与金融，经济，会计等财经类学科的专业问题，对应财经类书籍数据，
        如果问题不属于这两种类别，请回复'其他'

        问题：请分析海天味业的存货周转率和应收账款周转率，以及其销售费用占比，以了解其运营效率和成本控制情况。？
        问题类别：股票投资
        
        问题：商业银行的资产运用方式主要包括哪些？
        问题类别：财经百科

        问题：我想了解京东方A的液晶面板出货量和价格走势，以及其技术创新和市场份额，以便分析其行业地位和盈利能力。
        问题类别：股票投资
        
        问题：从银行借了100万，年化利率为3%，三个月后归还，需要偿还多少利息？
        问题类别：财经百科
        
        问题：中国移动的每股收益在过去两个季度内有何变化？
        问题类别：股票投资
        
        问题：一般而言，若市场利率上升，货币流通速度会？
        问题类别：财经百科
        
        问题：大模型是什么？
        问题类别：其他

        问题：{query}
        问题类别："""

    def get_reference(self, query):
        # self.progress(progress_text=f'')
        self.reference = dict(
            query=query
        )

    def choose_qa_tools(self, query) -> LangChainTool:
        i = 0
        while True:
            if i == 3:
                logging.info(f'无法识别问题类别')
                return LangChainTool()
            resp = self.run(query=query)
            self.progress(progress_text=f'分类结果为：{resp}')
            logging.info(f'意图识别结果为：{resp}')
            matched_tool_ls = []
            for tool in [FinKnowledgeQA(), FinInvestmentQA(), LangChainTool()]:
                if tool.name in resp:
                    matched_tool_ls.append(tool)
            if len(matched_tool_ls) == 1:
                return matched_tool_ls[0]
            i += 1
            continue
