import collections
from copy import deepcopy
from datetime import datetime
import math
from itertools import combinations
import heapq

# import json

# bleu 和 编辑距离 用于计算相似度


def bleu(ner, ent):
    """计算抽取实体与现有实体的匹配度 ner候选词,ent查询词"""
    len_pred, len_label = len(ner), len(ent)
    k = min(len_pred, len_label)
    if k == 0:
        return 0
    score = math.exp(min(0, 1 - len_label / len_pred))
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
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score if score > 0 else 0


def editing_distance(word1, word2):
    try:
        m, n = len(word1), len(word2)
    except:
        return float('inf')

    if m == 0 or n == 0:
        return abs(m-n)
    dp = [[float('inf') for _ in range(n+1)] for _ in range(m+1)]

    for i in range(m):
        dp[i][0] = i

    for i in range(n):
        dp[0][i] = i

    for i in range(1, m+1):
        for j in range(1, n+1):

            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                # 替换
                dp[i][j] = dp[i-1][j-1] + 1
                # 删除
                dp[i][j] = min(dp[i][j], min(dp[i-1][j], dp[i][j-1]) + 1)
    return dp[-1][-1]


class QuestionParser:
    def __init__(self, knowledge):
        self.most_k_similar = 3
        self.knowledge = knowledge
        self.max_return = 15  # 对路径查询限制路径条数

    def question2cypher(self, ent_dict, times_all):

        sql_dict = collections.defaultdict(list)  # {返回类型: 查询语句}

        basic_ent = collections.defaultdict(list)  # 要查询的主体、时间、意图信息

        def match_helper(match_subject, subject):
            '''更新basic_ent[subject]'''

            # 目标日期
            match_time = datetime.strptime(match_subject[0], "%Y-%m-%d")
            # 现有日期
            sj_time = [datetime.strptime(x, "%Y-%m-%d")
                       for x in self.knowledge[subject]]
            # 计算日期之间的差距，并选择最相近的日期
            closest_date = min(
                sj_time, key=lambda date: abs(date - match_time))
            basic_ent[subject].append(closest_date.strftime("%Y-%m-%d"))

        basic_ent['股票'] = ent_dict['主体'].get('股票', [])

        # 选出最匹配的时间 为空则使用近一年的
        extraction_time = ent_dict.get('时间')
        times_fin, times_gudong, times_dayline = times_all

        def find_time(index_type, time_pool):
            if len(extraction_time) == 0 or extraction_time[0] == '':
                for subject, sj_time in time_pool.items():
                    if sj_time:
                        basic_ent[f'{index_type}_{subject}'] = heapq.nlargest(
                            2, sj_time)
                    else:
                        basic_ent[f'{index_type}_{subject}'] = [
                            '2023', '2024']  # heapq.nlargest(3, sj_time)
            else:
                for subject, sj_time in time_pool.items():
                    self.knowledge[f'{index_type}_{subject}'] = sj_time
                    for e_time in extraction_time:  # 保证用户想要的时间是 图谱中有该主体最接近的时间
                        e_time_trans = []  # 精确查询
                        try:  # 用户给到某年某月某日  精确
                            ymd = datetime.strptime(
                                e_time, "%Y年%m月%d日").strftime("%Y-%m-%d")
                            # e_time_trans.append(ymd)
                            if ymd[5:] in ['03-31', '06-30', '09-30', '12-31'] and ymd in sj_time:
                                basic_ent[f'{index_type}_{subject}'].append(
                                    ymd)  # 精确
                            else:
                                e_time_trans.append(ymd)

                        except:
                            try:  # 用户给到某年某月 模糊
                                date_map = {
                                    3: '03-31',
                                    6: '06-30',
                                    9: '09-30',
                                    12: '12-31'
                                }
                                ym = datetime.strptime(e_time, "%Y年%m月")
                                if ym.month in [3, 6, 9, 12]:
                                    ymd = str(ym.year) + '-' + \
                                        date_map[ym.month]
                                    if ymd in sj_time:
                                        basic_ent[f'{index_type}_{subject}'].append(
                                            ymd)  # 精确
                                    else:
                                        e_time_trans.append(ymd)
                                else:
                                    e_time_trans.append(
                                        f'{ym.strftime("%Y-%m")}-01')
                            except:
                                try:  # 用户给到某年 模糊
                                    nianbao = datetime.strptime(
                                        e_time, "%Y年").strftime("%Y")
                                    if f'{nianbao}-12-31' in sj_time:
                                        basic_ent[f'{index_type}_{subject}'].append(
                                            f'{nianbao}-12-31')
                                    else:
                                        e_time_trans.append(f'{nianbao}-12-31')
                                except:
                                    e_time_trans.append(
                                        datetime.now().strftime("%Y-%m-%d"))
                        if sj_time == []:
                            for tm in e_time_trans:
                                basic_ent[f'{index_type}_{subject}'].append(tm)
                        if e_time_trans:
                            match_helper(
                                e_time_trans, f'{index_type}_{subject}')

        find_time('财务指标_发布时间', times_fin)
        find_time('十大股东_发布时间', times_gudong)

        # 去除模糊匹配得到重复的日期
        for subject in times_fin.keys():
            basic_ent[f'财务指标_发布时间_{subject}'] = list(
                collections.OrderedDict.fromkeys(basic_ent[f'财务指标_发布时间_{subject}']))
        for subject in times_gudong.keys():
            basic_ent[f'十大股东_发布时间_{subject}'] = list(
                collections.OrderedDict.fromkeys(basic_ent[f'十大股东_发布时间_{subject}']))

        # 用户意图  找出最相关的几个意图作为属性条件
        intent_match = set()
        extraction_intent = ent_dict.get('意图', [])
        for e_intent in extraction_intent:
            if '行业' in e_intent:
                intent_match.add('行业板块')
                continue
            if '股东' in e_intent:
                intent_match.add('十大股东')
                continue
            if e_intent in ['主营业务', '主营构成', '主营指标']:
                intent_match.add('主营构成')
                continue
            if e_intent in self.knowledge['实体']:
                intent_match.add(e_intent)
                continue
            for key in ['属性', '关系', '实体']:
                # #每股指标容易与属性匹配导致查出股东信息，需要忽略每股指标的属性特性
                # if e_intent == '每股指标' and key =='属性':
                #     continue
                # #财务指标与财务费用属性匹配度高，忽略财务指标属性特性
                # if e_intent == '财务指标' and key =='属性':
                #     continue
                if key == '关系':  # 关系匹配需要准确 否则查询出来很多 故给定0.5的阈值
                    tmp_scores_bleu = {kg_attr: bleu(
                        kg_attr, e_intent) for kg_attr in self.knowledge[key]}
                    topk = heapq.nlargest(
                        1, tmp_scores_bleu.items(), key=lambda x: x[1])
                else:
                    tmp_scores_ed = {kg_attr: editing_distance(
                        kg_attr, e_intent) for kg_attr in self.knowledge[key]}
                    topk = heapq.nsmallest(
                        self.most_k_similar, tmp_scores_ed.items(), key=lambda x: x[1])
                if (key != '关系' and topk[0][1] == 0) or (key == '关系' and topk[0][1] == 1):
                    intent_match.add(topk[0][0])
                    break
                else:
                    intent_match.update({kg_attr for kg_attr, score in topk if kg_attr not in ['name'] and
                                         len(e_intent) > score > 0.5})  # not in ['name', '股票']

        # 单节点查询：意图为单一结点的属性 直接返回该节点信息 并剔除查询该节点类型的意图
        for key, vals in self.knowledge['单节点属性'].items():
            rm_val = vals & intent_match  # 交集
            # 控制只返回一个匹配到的节点
            if rm_val:
                intent_match -= rm_val  # 剔除该意图
                intent_match.update({key})

        # 按关系选择路径
        single_path = []
        rels = self.knowledge['关系三元组'].keys()
        rel_rm = rels & intent_match
        if rel_rm:
            for rel in rel_rm:
                single_path.append(rel)     # 并集保存 查询节点类型
            intent_match -= rel_rm  # 剔除该意图

        # 记录有效查询语句的数量
        sql_dict['times'] = 0

        # 根据将主体作为首节点 时间和用户意图作为条件进行路径查询
        for subject_type in ['股票']:
            for subject in set(basic_ent.get(subject_type, [])):
                # time_set = set()
                subject_intent_match = deepcopy(intent_match)
                cur_time = datetime.now().strftime('%Y-%m-%d')
                time_list = basic_ent.get(
                    f'财务指标_发布时间_{subject}') if subject_type == '股票' else [cur_time]
                for time in time_list:
                    # if time in time_set:
                    #     continue
                    # time_set.add(time)
                    if subject_intent_match:
                        intent_match_iter = deepcopy(subject_intent_match)
                        sql_dict['times'] += 1
                        # 查询主体与意图实体的2跳路径
                        for intent in intent_match_iter:
                            if intent == '股票':
                                sql_dict['path'].append([f'{subject}{time}的信息如下\n', f"match (n:`股票`) where n.name = '{subject}' \
                                                            return '1', '2', labels(n)[0], properties(n) limit 3"])
                                subject_intent_match -= {intent}
                            # 意图在一级，但数据在二级，匹配到二级
                            elif intent in ['财务指标', '主营构成', '十大股东', '财务报表']:
                                sql_dict['path'].append([f'{subject}{time}的{intent}信息如下\n', f"match path=(n:`股票`)-[r:基本面]-(m:`{intent}`)-[*1]-() where n.name='{subject}' and \
                                                    all(node in nodes(path)[1..3] where (node.报告期 is null or node.报告期 contains '{time}')) \
                                                    WITH DISTINCT path LIMIT {self.max_return} \
                                                    return distinct type(relationships(path)[1]), properties(relationships(path)[1]), labels(nodes(path)[2])[0], properties(nodes(path)[2]) limit 3"])
                            # 意图和数据都在一级，则匹配到一级
                            elif intent in ['实际控制人', '行业板块']:
                                sql_dict['path'].append([f'{subject}{time}的{intent}信息如下\n', f"match path=(n:`{subject_type}`)-[*1]-(m:`{intent}`) where n.name='{subject}' and \
                                                            all(node in nodes(path) where (node.报告期 is null or node.报告期 =~'{time}.*')) \
                                                            WITH DISTINCT path LIMIT {self.max_return} \
                                                            return distinct type(relationships(path)[0]), properties(relationships(path)[0]), labels(nodes(path)[1])[0], properties(nodes(path)[1]) limit 3"])
                            # 意图和数据都在二级，需要增加匹配路径长度
                            else:
                                sql_dict['path'].append([f'{subject}{time}的{intent}信息如下\n', f"match path=(n:`股票`)-[*1..2]-(m:`{intent}`) where n.name='{subject}' and \
                                                    all(node in nodes(path)[1..3] where (node.报告期 is null or node.报告期 contains '{time}')) \
                                                    WITH DISTINCT path LIMIT {self.max_return} \
                                                    return distinct type(relationships(path)[1]), properties(relationships(path)[1]), labels(nodes(path)[2])[0], properties(nodes(path)[2]) limit 3"])

        for stock in basic_ent.get('股票', []):
            for rel in single_path:  # 按关系选择路径 限制时间
                for time in basic_ent.get(f'财务指标_发布时间_{stock}'):
                    sql_dict['times'] += 1
                    if rel == '按报告期':
                        sql_dict['path'].append([f'{stock}{time}的{rel}信息如下\n', f"match path=(n:`股票`)-[r:基本面]-(m:`{intent}`)-[*0..1]-() where n.name='{stock}' and \
                                                    all(node in nodes(path)[1..3] where (node.报告期 is null or node.报告期 contains '{time}')) \
                                                    WITH DISTINCT path LIMIT {self.max_return} \
                                                    return distinct type(relationships(path)[1]), properties(relationships(path)[1]), labels(nodes(path)[2])[0], properties(nodes(path)[2]) limit 3"])
                    elif rel == '基本面':
                        sql_dict['path'].append([f'{stock}{time}的{rel}信息如下\n', f"match path=(n:`股票`)-[r:`{rel}`]->(m)-[*1]-() WHERE (m:`财务指标`) and n.name='{stock}' and \
                                                        all(node IN nodes(path) where node.报告期 IS NULL OR node.报告期 contains '{time}' OR node.name contains '{time}') \
                                                    WITH DISTINCT path \
                                                    return distinct type(relationships(path)[1]), properties(relationships(path)[1]), labels(nodes(path)[2])[0], properties(nodes(path)[2]) limit 3"])  # limit {self.max_return}
                    else:
                        sql_dict['path'].insert(0, [f'{stock}{time}的{rel}信息如下\n', f"match path=(n)-[r:`{rel}`]->(m) WHERE n.name='{stock}' and \
                                                        (m.报告期 IS NULL OR m.报告期 =~'{time}.*') \
                                                    WITH DISTINCT path LIMIT {self.max_return} \
                                                    unwind nodes(path) as node unwind relationships(path) as rel \
                                                    return distinct type(rel), properties(rel), labels(node)[0], properties(node) limit 3"])  # limit {self.max_return}

                        sql_dict['path'].append([f'{stock}{time}的{rel}信息如下\n', f"match path=(n)-[r:`{rel}`]->(m)-[*0..2]->() WHERE n.name='{stock}' and \
                                                         (m.报告期 IS NULL OR m.报告期 =~'{time}.*') \
                                                    WITH DISTINCT path LIMIT {self.max_return} \
                                                    unwind nodes(path) as node unwind relationships(path) as rel \
                                                    return distinct type(rel), properties(rel), labels(node)[0], properties(node) limit 3"])

        if not sql_dict:  # 没有意图直接返回从主体出发的相关信息
            for subject_type in ['股票']:
                for subject in set(basic_ent.get(subject_type, [])):
                    cur_time = datetime.now().strftime('%Y-%m-%d')
                    time_list = basic_ent.get(
                        f'财务指标_发布时间_{subject}') if subject_type == '股票' else [cur_time]
                    for time in time_list:
                        sql_dict['times'] += 1
                        if subject_type == '股票':
                            sql_dict['path'].insert(0, [f'{subject}{time}的信息如下\n', f"match path=(n)-[*0..1]-(m) WHERE n.name='{subject}' and \
                                                        all(node in nodes(path) where (node.报告期 IS NULL OR node.报告期 =~'{time}.*')) \
                                                        WITH DISTINCT path LIMIT {self.max_return} \
                                                        unwind nodes(path) as node unwind relationships(path) as rel \
                                                        return distinct type(rel), properties(rel), labels(node)[0], properties(node) limit 3"])  # limit {self.max_return}
        return sql_dict
