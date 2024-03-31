import collections
import logging
import math
import time
from py2neo import Graph
import heapq
import re
import tiktoken
from datetime import datetime, timedelta
import random

from finllmqa.kg.search.question_parser import QuestionParser
from finllmqa.api.core import NEO4J_API_URL, STOCK_KG_USER, STOCK_KG_PW


class AnswerSearcher:
    def __init__(self, encoding_name: str = 'cl100k_base', max_length=2000, timeout=15):
        self.g = Graph(NEO4J_API_URL, auth=(STOCK_KG_USER, STOCK_KG_PW), name='neo4j')

        stock_data = self.g.run(f"match (n:`股票`) return n.name, n.代码").data()
        stock_pool, code_pool = list(
            zip(*[(item['n.name'], item['n.代码']) for item in stock_data]))

        start_year = 2010
        current_year = datetime.now().year
        current_month = datetime.now().month
        fin_time_pool = []

        for year in range(start_year, current_year + 1):
            for month in range(3, 13, 3):
                if year == current_year and month >= current_month:
                    break
                end_of_quarter = (datetime(year, month+1, 1) - timedelta(days=1)
                                  ) if month < 12 else (datetime(year, month, 31))
                fin_time_pool.append(end_of_quarter.strftime('%Y-%m-%d'))

        latest_time = heapq.nlargest(3, fin_time_pool)

        ent_data = self.g.run("CALL db.labels() YIELD label").data()
        ent_pool = [item['label'] for item in ent_data]
        ent_pool.remove('个股研报')
        ent2attr = {}
        attr_pool = set()        # 返回节点属性名
        for ent in ent_pool:
            tmp = set(self.g.run(f"match (n:`{ent}`) return properties(n) limit 1").data()[
                      0]['properties(n)'])
            attr_pool |= tmp
            ent2attr[ent] = tmp

        rel_data = self.g.run(
            "MATCH ()-[r]->() RETURN DISTINCT type(r) AS relationship_name").data()
        rel_pool = [item['relationship_name'] for item in rel_data]
        self.rel_triple = collections.defaultdict(list)
        self.rel_triple_hlt = collections.defaultdict(list)
        for rel in rel_pool:
            rel_data = self.g.run(
                f"MATCH ()-[r:`{rel}`]->() RETURN distinct labels(startNode(r))[0] as h, labels(endNode(r))[0] as t").data()
            for data in rel_data:
                self.rel_triple[rel].append([data['h'], data['t']])
                self.rel_triple_hlt[data['h']].append(data['t'])

        stock_1_grade_nodes = [x['reachableLabels'] for x in self.g.run(
            "MATCH (n:`股票`)-[*1]->(m) RETURN DISTINCT labels(m) AS reachableLabels").data()]
        # stock_2_grade_nodes = [x['reachableLabels'] for x in self.g.run("MATCH (n:`股票`)-[*2]->(m)
        # RETURN DISTINCT labels(m) AS reachableLabels").data()]

        # attr_data = self.g.run('CALL db.propertyKeys()').data()
        # attr_pool = [item['propertyKey'] for item in attr_data]

        # 股东信息的时间
        # times = {}
        # for tp, subject in ent_dict['主体'].values():
        #     time = self.g.run(f"match (n:`{tp}`) where n.name='{subject}' return n.时间 as time").data()
        #     times[subject] =

        self.knowledge = {'股票': stock_pool, '股票代码': code_pool, '财务指标_时间': fin_time_pool, '近一年': latest_time,
                          '实体': ent_pool, "单节点属性": ent2attr, '关系': rel_pool, '关系三元组': self.rel_triple,
                          '关系三元组辅助': self.rel_triple_hlt, '属性': attr_pool, '股票一级子节点': stock_1_grade_nodes}

        self.first_step_labels = [x['labels'] for x in self.g.run(
            "match (n:`股票`)-[*0..1]-(m) where n.name = '光洋股份' return distinct labels(m) as labels").data()]
        self.QP = QuestionParser(self.knowledge)
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.max_length = max_length
        self.timeout = timeout  # 设置查询超时时间（以秒为单位）
        self.truncation_dict = {
            'llm': 10,
            'table': 5
        }
        self.is_trunc = False

    def check_length(self, output):
        return True if len(self.encoding.encode(output)) > self.max_length else False

    def search_main(self, ent_dict, type='llm'):

        if len(ent_dict['主体']['股票']) > 1 or len(ent_dict['时间']) > 1 or len(ent_dict['意图']) > 2 or '基本面' in ent_dict['意图']:
            self.is_trunc = True
        else:
            self.is_trunc = False

        times_fin, times_gudong, times_dayline = {}, {}, {}
        stock_list = ent_dict['主体']['股票']
        intent_list = ent_dict['意图']
        time_list = ent_dict['时间']

        if stock_list:
            for subject in stock_list:
                time_fin = self.g.run(
                    f"MATCH (n:`股票`)-[*2]-(m:`常用指标`) where n.name = '{subject}' RETURN properties(m)['报告期'] as time").data()
                # time_gudong = self.g.run(
                #     f"match (n:`股票`)-[r:`基本面`]-(m:`主要股东`) where n.name = '{subject}' return collect(m.报告期) as time").data()
                # time_dayline = self.g.run(f"match (n:`日线行情`) where n.name = '交易日' and n.股票名称='{subject}' return n.交易日 as time").data()

                times_fin[subject] = [res['time'] for res in time_fin]
                # times_gudong[subject] = time_gudong[0]['time']
                # times_dayline[subject] = sorted(time_dayline[0]['time'].split(','))

        cypher_dict_table = ''
        if type == 'table':
            basic_table_ent = {'主体': {'股票': ent_dict['主体']['股票']}, '时间': [
                '2023年'], '意图': ['基本面']}
            cypher_dict_table = self.QP.question2cypher(
                basic_table_ent, (times_fin, times_gudong, times_dayline))
            # 拿最近十年的数据
            ent_dict['时间'] = [f'{2023 - x +1}年' for x in range(10, 0, -1)]

        cypher_dict = self.QP.question2cypher(
            ent_dict, (times_fin, times_gudong, times_dayline))

        if cypher_dict['times'] > 2 or type == 'table' or self.is_trunc:
            self.is_trunc = True
        else:
            self.is_trunc = False

        # def write_prop(labl: str, prop: dict, no_rep: set):
        #     ret = ''
        #     if labl not in no_rep and prop: #
        #         ret += f"{labl}包括以下属性：{','.join([key for key in prop.keys()])}\n"
        #         no_rep.add(labl)
        #     return ret

        # def write_prop_desc(labl: str, prop: dict, no_rep_desc):
        #     ret = ''
        #     if labl and prop:
        #         ret_tmp = ';'.join(list(map(str, [val for val in prop.values()])))
        #         if ret_tmp not in no_rep_desc:
        #             no_rep_desc.add(ret_tmp)
        #             ret = f"存在{labl}，属性值为:{ret_tmp}\n"
        #     return ret

        def write_prop_tabular(labl: str, prop: dict, no_rep: set, is_trunc: bool = False):
            ret = ''
            truncation = self.truncation_dict[type]
            if labl and prop:
                truc = (len(prioritize_keys[labl]) if labl in prioritize_keys.keys(
                ) else truncation) if is_trunc else len(prop)
                # 利润表优先属性中有两个属性（营业成本和营业支出）是互斥的
                truc = 5 if labl == '利润表' else truc
                ret_tmp = f'存在{labl}如下表所示\n|属性|值|\n|----|----|\n' + '\n'.join(
                    [f"|{k}|{v}|" for k, v in list(prop.items())[:truc] if v != 'nan'])+'\n|----|----|\n\n'
                if ret_tmp not in no_rep:
                    # if is_trunc: # 先取全部 后截取 截取后加入集合
                    # no_rep.add(ret_tmp)
                    ret = ret_tmp
            return ret
        # def write_prop_tabular_fin(labl: str, prop: dict, no_rep: set, is_trunc: bool=False):
        #     ret = ''
        #     if labl and prop:
        #         truc = self.truncation if is_trunc else len(prop)
        #         ret_tmp = f'存在财务指标-{labl}如下表所示\n|属性|值|\n|----|----|\n' + '\n'.join([f"|{k}|{v}|"
        #         for k, v in list(prop.items())[:truc]])+'\n|----|----|\n\n'
        #         if ret_tmp not in no_rep:
        #             # if is_trunc: # 先取全部 后截取 截取后加入集合
        #             # no_rep.add(ret_tmp)
        #             ret = ret_tmp
        #     return ret

        # def run_query(query, queue, stop_event):
        #     result = self.g.run(query)
        #     try:
        #         result = self.g.run(query)
        #     except Exception as e:
        #         stop_event.set()  # 设置停止事件
        #     finally:
        #         pass

        #     queue.put(result)

        # def get_res(cypher):
        #     result_queue = Queue()
        #     # 创建停止事件对象
        #     stop_event = threading.Event()
        #     thread = threading.Thread(target=run_query, args=(cypher, result_queue, stop_event))
        #     thread.setDaemon(True) # 将线程设置为守护线程，当主线程终止时，守护线程也会被强制终止。

        #     thread.start() # 启动线程
        #     thread.join(self.timeout) # 等待线程执行完毕或超时
        #     if thread.is_alive(): # 判断线程是否仍在运行
        #         print("查询超时")
        #         stop_event.set()
        #         # thread.join()
        #         res = None
        #     else:
        #         res = result_queue.get()
        #     return res

        def prioritize_key(prop_node: dict, key: list):
            '''优先显示key集合的属性'''
            new_dic = {}
            for k in key:
                if k in prop_node:
                    v = prop_node.pop(k)
                    new_dic[k] = v
            new_dic.update(**prop_node)
            return new_dic

        def filtered_key(prop_node: dict, key: list):
            '''除去key集合的属性'''
            new_dic = prop_node
            for k in key:
                if k in new_dic:
                    new_dic.pop(k)
            return new_dic

        prioritize_keys = {
            '资产负债表': ['资产总计', '负债合计', '所有者权益(或股东权益)合计', '流动资产合计', '流动负债合计'],
            '利润表': ['营业收入', '营业成本', '营业支出', '营业利润', '净利润', '基本每股收益'],
            '现金流量表': ['经营活动现金流入小计', '经营活动现金流出小计', '筹资活动现金流入小计', '筹资活动现金流出小计', '投资活动现金流入小计', '投资活动现金流出小计'],
            '每股指标': ['基本每股收益', '每股经营现金流', '每股净资产_最新股数', '每股营业收入', '每股留存收益'],
            '常用指标': ["净资产收益率(ROE)", "资产负债率", "总资产报酬率(ROA)", "净利润", "经营现金流量净额", "毛利率"],
            '财务风险': ['流动比率', '速动比率', '资产负债率', '权益乘数', '现金比率'],
            '盈利能力': ['净资产收益率(ROE)', '毛利率', '销售净利率', '总资产净利率_平均', '总资本回报率'],
            '营运能力': ['应付账款周转率', '应收账款周转率', '存货周转率', '流动资产周转率', '总资产周转率'],
            '成长能力': ['归属母公司净利润增长率', '营业总收入增长率', '净利润', '营业总收入', '归母净利润', '扣非净利润'],
            '收益质量': ["经营活动净现金/归属母公司的净利润", "经营性现金净流量/营业总收入", "销售成本率", "成本费用率", "所得税/利润总额", "经营活动净现金/销售收入"],
            '主营指标': ['name', '分类类型', '收入比例', '成本比例', '利润比例'],
            '持股信息': ['name', '名次', '持股数', '占总股本持股比例', '增减']
        }
        attr_unit = {
            "资产总计": "亿元",
            "负债合计": "亿元",
            "所有者权益(或股东权益)合计": "亿元",
            "流动资产合计": "亿元",
            "流动负债合计": "亿元",
            "营业收入": "亿元",
            "营业总收入": "亿元",
            "营业成本": "亿元",
            "营业支出": "亿元",
            "营业利润": "亿元",
            "净利润": "亿元",
            "基本每股收益": "元/股",
            "经营活动现金流入小计": "亿元",
            "经营活动现金流出小计": "亿元",
            "筹资活动现金流入小计": "亿元",
            "筹资活动现金流出小计": "亿元",
            "投资活动现金流入小计": "亿元",
            "投资活动现金流出小计": "亿元",
            "每股经营现金流": "元/股",
            "每股净资产_最新股数": "元/股",
            "每股营业收入": "元/股",
            "每股留存收益": "元/股",
            "净资产收益率(ROE)": "%",
            "资产负债率": "%",
            "总资产报酬率(ROA)": "%",
            "经营现金流量净额": "亿元",
            "毛利率": "%",
            "流动比率": "",
            "速动比率": "",
            "权益乘数": "",
            "现金比率": "%",
            "销售净利率": "%",
            "总资产净利率_平均": "%",
            "总资本回报率": "%",
            "应付账款周转率": "",
            "应收账款周转率": "",
            "存货周转率": "",
            "流动资产周转率": "",
            "总资产周转率": "",
            "归属母公司净利润增长率": "%",
            "营业总收入增长率": "%",
            "归母净利润": "亿元",
            "扣非净利润": "亿元",
            "经营活动净现金/归属母公司的净利润": "",
            "经营性现金净流量/营业总收入": "",
            "销售成本率": "%",
            "成本费用率": "%",
            "所得税/利润总额": "%",
            "经营活动净现金/销售收入": ""
        }

        def query_data_from_neo4j(cypher_dict):
            output = ''
            no_rep = set()  # 相同的节点类型提前声明有哪些属性prop 且不重复
            no_rep_desc = set()  # 相同的prop描述 不重复
            node_res = []  # 将查询结果缓存
            rel_res = []
            for desc_prefix, cypher in cypher_dict['node']:
                # try:
                #     #超时将抛出异常
                #     with eventlet.Timeout(self.timeout, True):  # 设置超时时间为2秒
                #         res = self.g.run(cypher).data()
                # except eventlet.timeout.Timeout:
                #     print('查询超时！')

                # 设置随机数种子
                random.seed(42)

                res = self.g.run(cypher).data()
                if res:
                    if desc_prefix not in no_rep:
                        output += '*'+desc_prefix
                        if type == 'llm':
                            no_rep.add(desc_prefix)

                    tmp_output = ""
                    tmp_output_trunc = ""
                    for r in res:
                        labl, prop = list(r.values())
                        if labl in prioritize_keys.keys():
                            prop = prioritize_key(prop, prioritize_keys[labl])
                        else:
                            prop = prioritize_key(
                                prop, ['name', '报告期', '分类类型', '股票名称'])

                        ret1 = write_prop_tabular(labl, prop, no_rep, True)
                        ret2 = write_prop_tabular(
                            labl, prop, no_rep, self.is_trunc)

                        tmp_output_trunc += ret1
                        tmp_output += ret2
                        if type == 'llm':
                            no_rep.add(ret1)
                            no_rep.add(ret2)
                    if self.is_trunc:
                        output += "请注意，以下只截取了部分属性\n"

                    cur_length = output+tmp_output_trunc if self.is_trunc else output+tmp_output
                    output = cur_length

            for desc_prefix, cypher in cypher_dict['path']:
                # print(cypher)
                # res = test(cypher)
                # try:
                #     #超时将抛出异常
                #     with eventlet.Timeout(self.timeout, True):  # 设置超时时间为2秒
                #         res = self.g.run(cypher).data()
                # except eventlet.timeout.Timeout:
                #     print('查询超时！')

                # 设置随机数种子
                random.seed(42)
                # global start
                res = self.g.run(cypher).data()
                if res:
                    if desc_prefix not in no_rep:
                        output += '*'+desc_prefix
                        if type == 'llm':
                            no_rep.add(desc_prefix)

                    tmp_output_desc = ""  # 等到循环表示的查询统计结束 再考虑是否将属性值放到output中.
                    tmp_output = ""
                    tmp_output_trunc = ""
                    for r in res:
                        labl_rel, prop_rel, labl_node, prop_node = list(
                            r.values())
                        if labl_node is None:
                            continue
                        if labl_node in prioritize_keys.keys():
                            prop_node = prioritize_key(
                                prop_node, prioritize_keys[labl_node])
                        else:
                            prop_node = prioritize_key(
                                prop_node, ['name', '报告期', '分类类型', '股票名称'])
                        if labl_node == '财务指标':
                            prop_node = filtered_key(prop_node, ['报告期'])
                        # if [labl_node] not in self.first_step_labels:
                        #     prop_node = filtered_key(prop_node,['name','报告期','股票名称','股票代码'])
                        #     prop_node = shuffle_key(prop_node)
                        # elif labl_node == '财务指标':
                        #     prop_node = filtered_key(prop_node,['报告期'])

                        # rel_res.append((labl_rel, prop_rel, labl_node, prop_node))
                        # if labl_rel not in no_rep:
                        #     for h, t in self.rel_triple[labl_rel]:# 打印关系描述
                        #         output += f"存在{labl_rel}关系：由{h}指向{t}\n"
                        #         # no_rep.add((h, labl_rel, t))
                        #         if self.check_length(output):
                        #             return output
                        #     no_rep.add(labl_rel)

                        # output += write_prop_tabular(labl_rel, prop_rel, no_rep)

                        if labl_node in ['财务指标', '主营构成', '主要股东'] and '基本面' not in intent_list:
                            continue

                        ret1 = write_prop_tabular(
                            labl_node, prop_node, no_rep, True)
                        ret2 = write_prop_tabular(
                            labl_node, prop_node, no_rep, self.is_trunc)
                        tmp_output_trunc += ret1
                        tmp_output += ret2
                        if type == 'llm':
                            no_rep.add(ret1)
                            no_rep.add(ret2)

                    if self.is_trunc:
                        output += "请注意，以下只截取了部分属性\n"

                    cur_length = output+tmp_output_trunc + \
                        tmp_output_desc if self.is_trunc else output+tmp_output+tmp_output_desc
                    # or self.is_trunc
                    if self.check_length(cur_length) and self.is_trunc and type == 'llm':
                        output = cur_length

                        # 正则表达式模式，匹配|----|----|
                        pattern = r'\|----\|----\|\n'
                        # 使用正则表达式进行替换，将|----|----|替换为空字符串''
                        output = re.sub(pattern, '', output)
                        return output
                    else:
                        output = cur_length
            return output

        output = query_data_from_neo4j(cypher_dict)
        output_table = query_data_from_neo4j(cypher_dict_table) if cypher_dict_table else ''

        def json_load(data):
            table = []
            # 标记是否处于需要提取的部分
            in_section = False
            # 拆分表格数据为行
            rows = data.strip().split('\n')
            # 解析并存储键值对
            for row in rows:
                if row.strip() == '|----|----|':  # 如果遇到分隔线
                    if in_section:  # 如果已经在提取的部分，则结束提取
                        in_section = False
                    else:
                        in_section = True  # 否则进入需要提取的部分
                elif in_section:
                    columns = row.split('|')
                    if len(columns)-2 == 2:  # 确保每行有2列
                        key = columns[1].strip()  # 提取属性并去除空格
                        value = columns[2].strip()  # 提取值并去除空格
                        unit = attr_unit[key] if key in attr_unit.keys(
                        ) else ''
                        norm_value = f'{round(float(value)/100000000,3)}' if unit == '亿元' else value
                        table.append({'key': key, 'value': value,
                                     'norn_value': norm_value, 'unit': unit})
            return table

        if type == 'table':
            info_dict = {
                'stock_info': []
            }

            output_split = output.split('*')
            output_table_split = output_table.split('*')
            for stock in ent_dict['主体']['股票']:
                stock_dict = {}
                stock_dict['name'] = stock
                stock_dict['data'] = {
                    'basic': [],
                    'question': []
                }
                # 基本面数据
                basic_info = [x for x in output_table_split if stock in x][0]
                data_split = basic_info.split('存在')
                # 正则表达式模式，匹配时间
                time_pattern = r'(\d{4}-\d{2}-\d{2})'
                time_match = re.search(time_pattern, basic_info)
                if time_match:
                    report_time = time_match.group(1)
                # 匹配类型
                type_pattern = r'(.+?)如下表所示'
                for data in data_split:
                    # 使用正则表达式进行匹配
                    record = {}
                    type_match = re.search(type_pattern, data)
                    if type_match and type_match.group(1) not in ['股票', '财务指标', '主营构成', '主要股东']:
                        record['type_'] = type_match.group(1)
                        record['report_time'] = report_time
                        record['record'] = json_load(data)
                        stock_dict['data']['basic'].append(record)

                # 问题数据
                ques_info_list = [x for x in output_split if stock in x]
                ques_data = []
                for ques_info in ques_info_list:
                    data_split = ques_info.split('存在')
                    # 正则表达式模式，匹配时间
                    time_pattern = r'(\d{4}-\d{2}-\d{2})'
                    time_match = re.search(time_pattern, ques_info)
                    if time_match:
                        report_time = time_match.group(1)
                    # 匹配类型
                    type_pattern = r'(.+?)如下表所示'
                    for data in data_split:
                        record = {}
                        type_match = re.search(type_pattern, data)
                        if type_match and type_match.group(1) not in ['股票', '财务指标', '主营构成', '主要股东']:
                            record['type_'] = type_match.group(1)
                            record['report_time'] = report_time
                            record['record'] = json_load(data)
                            ques_data.append(record)
                indicator_data_dict = {}
                for record in ques_data:
                    type_ = record['type_']
                    if type_ not in indicator_data_dict.keys():
                        indicator_data_dict[type_] = [
                            {
                                'report_time': record['report_time'],
                                'record': record['record'],
                            }
                        ]
                    else:
                        indicator_data_dict[type_].append({
                            'report_time': record['report_time'],
                            'record': record['record'],
                        })
                stock_dict['data']['question'] = [
                    {"type_": type_, "report_list": data} for type_, data in indicator_data_dict.items()]
                info_dict['stock_info'].append(stock_dict)
            return info_dict

        # 正则表达式模式，匹配|----|----|
        pattern = r'\|----\|----\|\n'
        # 使用正则表达式进行替换，将|----|----|替换为空字符串''
        output = re.sub(pattern, '', output)
        return output


if __name__ == '__main__':
    AS = AnswerSearcher(max_length=2000, timeout=5)  # 在2020年3月31日的

    logger = logging.getLogger()
    fh = logging.FileHandler("answer.log", encoding="utf-8", mode="a")
    formatter = logging.Formatter("%(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    # logging.basicConfig(filename='answer2.log', level=logging.INFO, format='%(message)s')

    ent_dict = {'主体': {'股票': ['东方财富']}, '时间': [],
                '意图': ['常用指标']}
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(1, end-start, len(answer))
    print(answer)
    # 1
    ent_dict = {'主体': {'股票': ['广发证券', '光大证券']},
                '时间': ['2022年', '2023年'], '意图': ['财务报表']}
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(1, end-start, len(answer))
    logging.info('测试1:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
    # 2
    ent_dict = {'主体': {'股票': ['东方财富']}, '时间': [''], '意图': ['实际控制人']}
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(2, end-start, len(answer))
    logging.info('测试2:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
    # 3
    ent_dict = {'主体': {'股票': ['广发证券']}, '时间': ['2022年'], '意图': ['财务指标']}
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(3, end-start, len(answer))
    logging.info('测试3:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
    # 4
    ent_dict = {'主体': {'股票': ['广发证券']},
                "时间": ['2022年', '2021年'],  # '2020年3月31日',
                '意图': ['财务指标']}
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(4, end-start, len(answer))
    logging.info('测试4:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
    # 5
    ent_dict = {'主体': {'股票': ['东方财富']},
                "时间": ['2022年', '2021年'],  # '2020年3月31日',
                '意图': ['财务指标']}
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(5, end-start, len(answer))
    logging.info('测试5:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
    # 6
    ent_dict = {'主体': {'股票': ['广发证券']},
                "时间": ['2023年3月'],  # '2020年3月31日',
                '意图': ['基本面']}
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(6, end-start, len(answer))
    logging.info('测试6:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
    # 7
    ent_dict = {'主体': {'股票': [''], '行业': ['医疗服务']},
                "时间": [''],  # '2020年3月31日',
                '意图': ['成分股']}
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(7, end-start, len(answer))
    logging.info('测试7:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
    # 8
    ent_dict = {'主体': {'股票': [''], '行业': ['化学制药']},
                "时间": ['2022年'],  # '2020年3月31日',
                '意图': ['']}
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(8, end-start, len(answer))
    logging.info('测试8:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
    # 9
    ent_dict = {'主体': {'股票': [''], '行业': ['化学制药']},
                "时间": [''],  # '2020年3月31日',
                '意图': ['板块']}
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(9, end-start, len(answer))
    logging.info('测试9:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
    # 10
    ent_dict = {'主体': {'股票': [''], '行业': ['化学制药']},
                "时间": ['2022年'],  # '2020年3月31日',
                '意图': ['评价']}  # 失败
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(10, end-start, len(answer))
    logging.info('测试10:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
    # 11
    ent_dict = {'主体': {'股票': ['中国中免']},
                "时间": ['2022年'],  # '2020年3月31日',
                '意图': ['基本面']}
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(11, end-start, len(answer))
    logging.info('测试11:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
    # 12
    ent_dict = {'主体': {'股票': ['泸州老窖', '洋河股份']},
                "时间": ['2022年', '2021年'],  # '2020年3月31日',
                '意图': ['财务指标']}
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(12, end-start, len(answer))
    logging.info('测试12:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
    # 13
    ent_dict = {'主体': {'股票': ['分众传媒']},
                "时间": [''],  # '2020年3月31日',   fail
                '意图': ['实控人']}
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(13, end-start, len(answer))
    logging.info('测试13:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
    # 14
    ent_dict = {'主体': {'股票': ['贵州茅台']},
                "时间": ['2022年'],  # '2020年3月31日',
                '意图': ['净利润']}  # '财务指标',
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(14, end-start, len(answer))
    logging.info('测试14:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
    # 15
    ent_dict = {'主体': {'股票': ['协和电子']},
                "时间": ['2020年3月31日'],  # '2020年3月31日',
                '意图': ['财务指标']}
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(15, end-start, len(answer))
    logging.info('测试15:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
    # 16
    ent_dict = {'主体': {'股票': ['协和电子']},
                "时间": ['2020年3月31日'],  # '2020年3月31日',
                '意图': ['基本每股收益']}
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(16, end-start, len(answer))
    logging.info('测试16:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
    # 17
    ent_dict = {'主体': {'股票': ['光洋股份'], '行业': ['航空航天', '航空机场', '包装材料']},
                "时间": [''],  # '2020年3月31日',
                '意图': ['基本每股收益', '财务指标', '每股指标', '板块']}
    answer = AS.search_main(ent_dict)
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(17, end-start, len(answer))
    logging.info('测试17:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)

    # 18
    ent_dict = {'主体': {'股票': ['N赛维']},
                "时间": [''],  # '2020年3月31日',
                '意图': ['基本面']}
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(18, end-start, len(answer))
    logging.info('测试18:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
    # 19
    ent_dict = {'主体': {'股票': ['N赛维']},
                "时间": [''],  # '2020年3月31日',
                '意图': ['板块']}
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(19, end-start, len(answer))
    logging.info('测试19:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
    # 20
    ent_dict = {'主体': {'股票': ['长电科技']},
                "时间": [''],  # '2020年3月31日',
                '意图': ['财务指标']}
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(20, end-start, len(answer))
    logging.info('测试20:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
    # 21
    ent_dict = {'主体': {'股票': ['长电科技']},
                "时间": [''],  # '2020年3月31日',
                '意图': ['主营信息']}
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(21, end-start, len(answer))
    logging.info('测试21:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
    # 22
    ent_dict = {'主体': {'股票': ['长电科技']},
                "时间": [''],  # '2020年3月31日',
                '意图': ['行业板块']}
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(22, end-start, len(answer))
    logging.info('测试22:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
    # 23
    ent_dict = {'主体': {'股票': ['协和电子']},
                "时间": ['2020年3月31日'],  # '2020年3月31日',
                '意图': ['营业总收入']}
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(23, end-start, len(answer))
    logging.info('测试23:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
    # 24
    ent_dict = {'主体': {'股票': ['N赛维']},
                "时间": [''],  # '2020年3月31日',   失败 时间没对上
                '意图': ['股东信息']}
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(24, end-start, len(answer))
    logging.info('测试24:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
    # 25
    ent_dict = {'主体': {'股票': ['N赛维']},
                "时间": [''],  # '2020年3月31日',
                '意图': ['股东信息', '情况', '可以', '控股人']}
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(25, end-start, len(answer))
    logging.info('测试25:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
    # 26
    ent_dict = {'主体': {'股票': ['广发证券']},
                "时间": ['2022年', '2023'],  # '2020年3月31日',
                '意图': ['财务指标']}
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(26, end-start, len(answer))
    logging.info('测试26:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
    # 27
    ent_dict = {'主体': {'股票': ['浙江仙通']},
                "时间": ['2022年', '2023'],  # '2020年3月31日',
                '意图': ['实控人']}
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(27, end-start, len(answer))
    logging.info('测试27:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
    # 28
    ent_dict = {'主体': {'股票': ['浙江仙通']},
                "时间": ['2022年', '2023'],  # '2020年3月31日',
                '意图': ['控股人']}
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(28, end-start, len(answer))
    logging.info('测试28:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
    # 29
    ent_dict = {'主体': {'股票': ['广发证券']},
                "时间": ['2022年9月'],  # '2022年', '2023',
                '意图': ['评级']}
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(29, end-start, len(answer))
    logging.info('测试29:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
    # 30
    ent_dict = {'主体': {'股票': ['广发证券']},
                "时间": ['2022年9月'],  # '2022年', '2023',
                '意图': ['基本面']}
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(30, end-start, len(answer))
    logging.info('测试30:'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)

    # 31
    ent_dict = {'主体': {'股票': ['光洋股份'], '行业': ['航空航天', '航空机场', '包装材料']},
                "时间": ['2023'],  # '2020年3月31日',
                '意图': ['基本每股收益',]}
    answer = AS.search_main(ent_dict)
    start = time.time()
    answer = AS.search_main(ent_dict)
    end = time.time()
    print(31, end-start, len(answer))
    logging.info('测试31'+str(ent_dict))
    logging.info(len(AS.encoding.encode(answer)))
    logging.info(answer)
