import pandas as pd
import random
import json
import os

# 确保输出目录存在
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'raw')
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# 定义SQL查询模板
sql_templates = [
    # 原有模板
    "公司{department}部门的男女比例是多少",
    "{department}部门的平均工资是多少",
    "公司{year}年的营业额是多少",
    "公司{department}部门有多少人",
    "公司{year}年{quarter}季度的利润是多少",
    "{department}部门{year}年的人员流动率是多少",
    "公司目前有多少名{position}职位的员工",
    "{department}部门的{position}平均薪资是多少",
    "公司{year}年招聘了多少名新员工",
    "公司{department}部门的主管是谁",
    "公司{year}年的员工离职率是多少",
    "公司各部门的人数分布情况是怎样的",
    "{department}部门的项目完成率是多少",
    "公司{year}年的培训预算使用了多少",
    "公司{department}部门的绩效最高的员工是谁",
    "公司{year}年的财务报表显示了什么",
    "公司员工的平均年龄是多少",
    "{department}部门的项目数量是多少",
    "公司{year}年的税收支出是多少",
    "公司各职级的薪资范围是什么",
    
    # 新增模板 - 更多变体和表达
    "公司的远程工作政策是怎样的",
    "如何申请公司提供的教育补贴",
    "公司的技术栈包括哪些内容",
    "公司的数据安全政策有哪些规定",
    "如何使用公司的打印机和复印机",
    "公司的差旅报销流程是什么",
    "如何获取公司的品牌资源和素材",
    "公司的环保政策有哪些内容",
    "如何参加公司组织的培训课程",
    "公司的绩效评估周期是怎样的",
    "如何申请公司的内部转岗",
    "公司的加班补偿政策是什么",
    "如何使用公司的健身房设施",
    "公司的员工福利包括哪些内容",
    "如何获取公司的产品折扣",
    "公司的信息安全培训内容是什么",
    "如何参与公司的社会责任项目",
    "公司的知识产权保护政策是什么",
    "如何使用公司的共享单车福利",
    "公司的团队建设活动有哪些",
    "如何提交产品改进建议",
    "公司的办公用品申请流程是什么",
    "如何使用公司的电子签名系统",
    "公司的员工心理健康支持项目有哪些",
    "如何参与公司的创新项目",
    
    # 更多口语化表达 - 新增
    "请问公司的病假政策是怎么规定的",
    "能告诉我公司的发展历史吗",
    "我想了解一下公司的企业文化",
    "帮我查一下公司的技术发展路线",
    "公司有哪些福利待遇",
    "如何申请在家办公",
    "公司的年会一般什么时候举行",
    "我想了解公司的晋升通道",
    "公司的考勤制度是怎样的",
    "如何使用公司的内部沟通工具",
    "公司的加班餐补政策是什么",
    "我想知道公司的培训体系",
    "公司的绩效奖金如何计算",
    "如何申请调岗",
    "公司的试用期政策是什么",
    "如何使用公司的云盘系统",
    "公司的专利申请流程是什么",
    "如何参加公司的技术分享会",
    "公司的年度体检政策是什么",
    "如何申请带薪休假",
    
    # 更具体的问题 - 新增
    "公司的VPN如何连接",
    "公司的WiFi密码是多少",
    "如何重置公司内部系统密码",
    "公司的打卡时间是几点到几点",
    "如何申请公司的学习资源",
    "公司的停车场在哪里",
    "如何使用公司的会议预订系统",
    "公司的餐厅营业时间是什么时候",
    "如何申请公司的法定节假日调休",
    "公司的紧急联系人是谁",
    "如何获取公司的品牌标识使用指南",
    "公司的安全出口在哪里",
    "如何使用公司的快递收发服务",
    "公司的IT支持电话是多少",
    "如何申请公司的内部转账",
    "公司的保密协议内容是什么",
    "如何使用公司的打印机扫描功能",
    "公司的员工手册在哪里可以找到",
    "如何申请公司的资源访问权限",
    "公司的防火演习流程是什么",
    "查询{department}部门{year}年的人员构成",
    "{department}部门{year}年{quarter}季度的业绩如何",
    "统计公司{year}年各部门的预算使用情况",
    "分析{department}部门{position}的工作效率",
    "计算公司{year}年的人均产值",
    "查看{department}部门{year}年的客户满意度",
    "统计{year}年公司的收入构成",
    "分析{department}部门{year}年的成本结构",
    "查询{position}在各部门的分布情况",
    "计算{department}部门{year}年的投资回报率",
    "统计公司{year}年的市场份额变化",
    "分析{department}部门{year}年的绩效考核结果",
    "查询公司{year}年的产品销量排名",
    "计算{department}部门{position}的平均工作年限",
    "统计公司{year}年的客户增长率",
    "分析{department}部门{year}年的项目成功率",
    "查询公司{position}的晋升周期",
    "计算{department}部门{year}年的人均培训时长",
    "统计公司{year}年的研发投入占比",
    "分析{department}部门的年龄结构",
    "查询公司{year}年的国内外市场销售比例",
    "计算{department}部门{year}年的人均加班时长",
    "统计公司各部门的人才流失率",
    "分析{year}年公司的季度业绩波动",
    "查询{department}部门的管理层结构",
    
    # 更多口语化表达 - 新增
    "帮我看看{department}部门{year}年的员工数量变化",
    "我想了解一下{department}部门的人员结构",
    "能告诉我{year}年公司的盈利情况吗",
    "麻烦查一下{department}部门{position}的绩效如何",
    "请问{year}年{quarter}季度的销售额是多少",
    "我需要知道{department}部门的预算执行情况",
    "给我统计一下公司各部门的人均薪资",
    "帮忙分析下{year}年公司的业绩趋势",
    "查一下{department}部门{year}年新招聘了多少人",
    "我想看看公司{position}的分布情况",
    "统计一下{department}部门的男女比例",
    "帮我对比一下各部门{year}年的业绩",
    "查询一下{department}部门的主要项目进展",
    "我想了解公司{year}年的人员流动情况",
    "给我看看{department}部门的组织架构",
    
    # 更复杂的查询 - 新增
    "比较{department}部门和{department}部门的人均产值",
    "分析{year}年和{year}年公司营收的变化趋势",
    "统计{department}部门{year}年各季度的业绩波动",
    "对比{position}和{position}在各部门的薪资差异",
    "分析{department}部门{year}年至{year}年的人员变化",
    "查询公司{year}年各季度的利润率变化",
    "统计{department}部门不同职级的人员占比",
    "分析公司{year}年各月度的收入支出情况",
    "对比{department}部门和{department}部门的项目成功率",
    "查询公司{year}年不同产品线的销售占比"
]

# 定义知识库查询模板
rag_templates = [
    # 原有模板
    "公司的人工客服电话号码是多少",
    "公司的休假政策是什么",
    "如何申请公司的报销流程",
    "公司的工作时间是怎么规定的",
    "公司的地址在哪里",
    "公司的创始人是谁",
    "公司的使命和愿景是什么",
    "如何重置公司邮箱密码",
    "公司的年终奖发放标准是什么",
    "公司的晋升机制是怎样的",
    "如何使用公司的内部知识库",
    "公司的服务器维护时间是什么时候",
    "公司的产品线包括哪些产品",
    "如何向IT部门报告技术问题",
    "公司的办公室布局图在哪里可以找到",
    "公司的健康保险政策是什么",
    "如何预订公司的会议室",
    "公司的主要竞争对手有哪些",
    "公司的股票代码是什么",
    "如何加入公司的员工俱乐部",
    
    # 新增模板 - 更多变体和表达
    "公司的远程工作政策是怎样的",
    "如何申请公司提供的教育补贴",
    "公司的技术栈包括哪些内容",
    "公司的数据安全政策有哪些规定",
    "如何使用公司的打印机和复印机",
    "公司的差旅报销流程是什么",
    "如何获取公司的品牌资源和素材",
    "公司的环保政策有哪些内容",
    "如何参加公司组织的培训课程",
    "公司的绩效评估周期是怎样的",
    "如何申请公司的内部转岗",
    "公司的加班补偿政策是什么",
    "如何使用公司的健身房设施",
    "公司的员工福利包括哪些内容",
    "如何获取公司的产品折扣",
    "公司的信息安全培训内容是什么",
    "如何参与公司的社会责任项目",
    "公司的知识产权保护政策是什么",
    "如何使用公司的共享单车福利",
    "公司的团队建设活动有哪些",
    "如何提交产品改进建议",
    "公司的办公用品申请流程是什么",
    "如何使用公司的电子签名系统",
    "公司的员工心理健康支持项目有哪些",
    "如何参与公司的创新项目",
    
    # 更多口语化表达 - 新增
    "请问公司的病假政策是怎么规定的",
    "能告诉我公司的发展历史吗",
    "我想了解一下公司的企业文化",
    "帮我查一下公司的技术发展路线",
    "公司有哪些福利待遇",
    "如何申请在家办公",
    "公司的年会一般什么时候举行",
    "我想了解公司的晋升通道",
    "公司的考勤制度是怎样的",
    "如何使用公司的内部沟通工具",
    "公司的加班餐补政策是什么",
    "我想知道公司的培训体系",
    "公司的绩效奖金如何计算",
    "如何申请调岗",
    "公司的试用期政策是什么",
    "如何使用公司的云盘系统",
    "公司的专利申请流程是什么",
    "如何参加公司的技术分享会",
    "公司的年度体检政策是什么",
    "如何申请带薪休假",
    
    # 更具体的问题 - 新增
    "公司的VPN如何连接",
    "公司的WiFi密码是多少",
    "如何重置公司内部系统密码",
    "公司的打卡时间是几点到几点",
    "如何申请公司的学习资源",
    "公司的停车场在哪里",
    "如何使用公司的会议预订系统",
    "公司的餐厅营业时间是什么时候",
    "如何申请公司的法定节假日调休",
    "公司的紧急联系人是谁",
    "如何获取公司的品牌标识使用指南",
    "公司的安全出口在哪里",
    "如何使用公司的快递收发服务",
    "公司的IT支持电话是多少",
    "如何申请公司的内部转账",
    "公司的保密协议内容是什么",
    "如何使用公司的打印机扫描功能",
    "公司的员工手册在哪里可以找到",
    "如何申请公司的资源访问权限",
    "公司的防火演习流程是什么"
]

# 定义变量替换的选项
departments = [
    "人力资源", "财务", "市场营销", "研发", "销售", "客户服务", "法务", "行政",
    "产品", "设计", "数据分析", "质量控制", "供应链", "采购", "公关", "战略规划",
    "国际业务", "培训发展", "信息技术", "安全", "运营", "物流", "客户成功", "内审"
]
years = ["2018", "2019", "2020", "2021", "2022", "2023", "2024"]
quarters = ["第一", "第二", "第三", "第四"]
positions = [
    "工程师", "经理", "总监", "助理", "专员", "顾问", 
    "分析师", "架构师", "主管", "技术员", "实习生", "副总裁",
    "组长", "讲师", "研究员", "开发者", "测试员", "设计师"
]

# 生成SQL查询样本
def generate_sql_samples(n=500):
    samples = []
    for _ in range(n):
        template = random.choice(sql_templates)
        query = template.format(
            department=random.choice(departments) if "{department}" in template else "",
            year=random.choice(years) if "{year}" in template else "",
            quarter=random.choice(quarters) if "{quarter}" in template else "",
            position=random.choice(positions) if "{position}" in template else ""
        )
        samples.append({"query": query, "intent": "sql"})
    return samples

# 生成知识库查询样本
def generate_rag_samples(n=500):
    samples = []
    for _ in range(n):
        query = random.choice(rag_templates)
        samples.append({"query": query, "intent": "rag"})
    return samples

# 生成自定义样本
def generate_custom_samples():
    custom_samples = [
        # SQL查询样本 - 更加口语化和多样化的表达
        {"query": "我们公司男性员工和女性员工的比例是多少", "intent": "sql"},
        {"query": "研发部门的人均薪资是多少", "intent": "sql"},
        {"query": "去年第四季度的销售额是多少", "intent": "sql"},
        {"query": "哪个部门的人数最多", "intent": "sql"},
        {"query": "今年的营业收入比去年增长了多少", "intent": "sql"},
        {"query": "帮我看一下我们公司的男女比例", "intent": "sql"},
        {"query": "能告诉我销售部门的平均工资吗", "intent": "sql"},
        {"query": "想知道公司去年的总收入", "intent": "sql"},
        {"query": "帮我查一下研发部门有多少人", "intent": "sql"},
        {"query": "我想了解一下公司各部门的人员分布", "intent": "sql"},
        {"query": "能帮我统计一下今年的招聘人数吗", "intent": "sql"},
        {"query": "请问公司的员工平均年龄是多少", "intent": "sql"},
        {"query": "帮我分析一下各部门的项目完成率", "intent": "sql"},
        {"query": "我想查询一下技术部门的人员流动情况", "intent": "sql"},
        {"query": "能帮我看看哪个部门的绩效最好吗", "intent": "sql"},
        
        # 知识库查询样本 - 更加口语化和多样化的表达
        {"query": "公司的年假政策是怎样的", "intent": "rag"},
        {"query": "如何申请内部转岗", "intent": "rag"},
        {"query": "公司的发展历史是什么", "intent": "rag"},
        {"query": "公司的核心价值观是什么", "intent": "rag"},
        {"query": "如何使用公司的VPN服务", "intent": "rag"},
        {"query": "请问公司有什么福利政策", "intent": "rag"},
        {"query": "我想了解一下公司的培训体系", "intent": "rag"},
        {"query": "公司的办公地点在哪里", "intent": "rag"},
        {"query": "如何申请公司的健身卡", "intent": "rag"},
        {"query": "公司的技术栈主要包括哪些", "intent": "rag"},
        {"query": "请问公司的考勤制度是怎样的", "intent": "rag"},
        {"query": "如何使用公司的打印机", "intent": "rag"},
        {"query": "公司的组织架构是怎样的", "intent": "rag"},
        {"query": "如何申请公司的电脑设备", "intent": "rag"},
        {"query": "公司的文化活动有哪些", "intent": "rag"}
    ]
    return custom_samples

# 生成数据集并保存
def generate_dataset(train_size=4000, test_size=1000, val_size=1000, seed=42):
    random.seed(seed)
    
    # 生成样本 - 增加样本数量
    sql_samples = generate_sql_samples(train_size // 2 + test_size // 2 + val_size // 2)
    rag_samples = generate_rag_samples(train_size // 2 + test_size // 2 + val_size // 2)
    custom_samples = generate_custom_samples()
    
    # 合并所有样本
    all_samples = sql_samples + rag_samples + custom_samples
    random.shuffle(all_samples)
    
    # 分割训练集、验证集和测试集
    train_data = all_samples[:train_size]
    val_data = all_samples[train_size:train_size+val_size]
    test_data = all_samples[train_size+val_size:train_size+val_size+test_size]
    
    # 转换为DataFrame
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)
    
    # 保存为CSV文件
    train_df.to_csv(os.path.join(RAW_DATA_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(RAW_DATA_DIR, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(RAW_DATA_DIR, 'test.csv'), index=False)
    
    # 保存为JSON文件
    with open(os.path.join(RAW_DATA_DIR, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(RAW_DATA_DIR, 'val.json'), 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(RAW_DATA_DIR, 'test.json'), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"生成的训练集大小: {len(train_data)}")
    print(f"生成的验证集大小: {len(val_data)}")
    print(f"生成的测试集大小: {len(test_data)}")
    print(f"SQL查询样本数量: {sum(1 for sample in all_samples if sample['intent'] == 'sql')}")
    print(f"知识库查询样本数量: {sum(1 for sample in all_samples if sample['intent'] == 'rag')}")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    train_df, val_df, test_df = generate_dataset()
    
    # 显示一些样本
    print("\n训练集样本:")
    print(train_df.sample(5))
    
    print("\n验证集样本:")
    print(val_df.sample(5))
    
    print("\n测试集样本:")
    print(test_df.sample(5))