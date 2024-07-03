'''
Pretraining Tasks -- 5 Prompt Families (1, 2, 3, 4, 5)
Zeroshot Tasks -- 1 Prompt Family (Z)
'''

# 所有的prompts

all_tasks = {}

# Task2 U2Q任务的ID base数据，3条prompts，现已弃用
# Task3 U2Q任务的text base数据，1～3是原始的prompt，4～6使用chatGPT修改过，7～9使用了国籍信息和类目偏好信息，10～12规范了格式, 13只使用国籍和兴趣信息进行预测，14只使用历史交互数据
# Task4 QAC任务的数据，1～3使用用户的历史行为信息辅助预测，4～6使用用户的类目偏好信息和国籍信息辅助预测，7～9规范了格式，10只使用历史交互信息进行预测
# Task5 Q2Q任务的数据，1~4是原始的prompt，5～8使用了国籍信息和类目偏好信息，9～11规范了格式，12只使用国籍和兴趣进行预测
# Task6 I2Q任务的数据，3条prompt，目前没法添加额外信息，4～6规范了格式
# Task7 U2QC任务的数据，目前没有在使用
# Task8 Q2C任务的数据，1，2是旧prompt，3，4规范了格式，5是从candidate中选择
# Task9 U2C任务的数据，1，2是直接预测，3，4是预测mask结果
# Task10 traditional任务的数据，1，2是判断题，3，4是选择题
# =====================================================
# Task Subgroup 2 -- Sequential -- 3 Prompts
# =====================================================

task_subgroup_2 = {}

template = {}

'''
Input template:
Given the following purchase history of user {{user_id}}:
{{history item list of {{item_id}}}}
predict next possible item to be purchased by the user?
 
 
Target template:
{{item [item_id]}}
 
 
Metrics:
HR, NDCG, MRR
'''

template['source'] = "Given the following purchase history of user_{} : \n {} \n predict next possible item to be purchased by the user ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-1"

task_subgroup_2["2-1"] = template


template = {}
'''
Input template:
I find the purchase history list of user {{user_id}}:
{{history item list of {{item_id}}}}
I wonder which is the next item to recommend to the user. Can you help me decide?
 
 
Target template:
{{item [item_id]}}
 
 
Metrics:
HR, NDCG, MRR
'''
template['source'] = "I find the purchase history list of user_{} : \n {} \n I wonder what is the next item to recommend to the user . Can you help me decide ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-2"

task_subgroup_2["2-2"] = template


template = {}
'''
Input template:
Here is the purchase history list of user {{user_id}}:
{{history item list of {{item_id}}}}
try to recommend next item to the user
 
Target template:
{{item [item_id]}}
 
 
Metrics:
HR, NDCG, MRR
'''
template['source'] = "Here is the purchase history list of user_{} : \n {} \n try to recommend next item to the user"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-3"

task_subgroup_2["2-3"] = template


all_tasks['sequential'] =  task_subgroup_2


# =====================================================
# Task Subgroup 3 -- Sequential Text generation（Text） -- 6 Prompts
# =====================================================

task_subgroup_3 = {}

template = {}

'''
Input template:
Given the following purchase history of user {{user_id}}:
{{history item list of {{item}}}}
generate next possible item to be purchased by the user?


Target template:
{{item [item]}}


Metrics:
未确定
'''
template['source'] = "Given the following purchase history of user_{} : \n {} \n generate next possible item to be purchased by the user ?"
template['target'] = "{}"
template['task'] = "text"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item']
template['id'] = "3-1"

task_subgroup_3["3-1"] = template

template = {}
'''
Input template:
I find the purchase history list of user {{user_id}}:
{{history item list of {{item}}}}
I wonder what is the next item to recommend to the user. Can you help me generate it?


Target template:
{{item [item]}}


Metrics:
未确定
'''
template[
    'source'] = "I find the purchase history list of user_{} : \n {} \n I wonder what is the next item to recommend to the user . Can you help me generate it ?"
template['target'] = "{}"
template['task'] = "text"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item']
template['id'] = "3-2"

task_subgroup_3["3-2"] = template

template = {}
'''
Input template:
Here is the purchase history list of user {{user_id}}:
{{history item list of {{item}}}}
try to generate the next item to recommend to the user

Target template:
{{item [item]}}


Metrics:
未确定
'''
template['source'] = "Here is the purchase history list of user_{} : \n {} \n try to generate the next item to recommend to the user"
template['target'] = "{}"
template['task'] = "text"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item']
template['id'] = "3-3"

task_subgroup_3["3-3"] = template

template = {}
'''
Input template:
Here is the purchase history list of user_{} in reverse chronological order:
{{history item list of {{item}}}}
Please generate a recommendation for the next item to recommend to the user based on their recent purchases.

Target template:
{{item [item]}}


Metrics:
未确定
'''
template['source'] = "Here is the purchase history list of user_{} in reverse chronological order: \n {} \n Please generate a recommendation for the next item to recommend to the user based on their recent purchases."
template['target'] = "{}"
template['task'] = "text"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item']
template['id'] = "3-4"

task_subgroup_3["3-4"] = template

template = {}
'''
Input template:
I have the purchase history list of user_{} in chronological order::
{{history item list of {{item}}}}
Can you help me generate a recommendation based on the user's recent purchase history?

Target template:
{{item [item]}}


Metrics:
未确定
'''
template['source'] = "I have the purchase history list of user_{} in chronological order: \n {} \n Can you help me generate a recommendation based on the user's recent purchase history?"
template['target'] = "{}"
template['task'] = "text"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item']
template['id'] = "3-5"

task_subgroup_3["3-5"] = template

template = {}
'''
Input template:
Based on the following purchase history of user_{} in chronological order:
{{history item list of {{item}}}}
Please generate a recommendation for the next item to be purchased by the user.

Target template:
{{item [item]}}


Metrics:
未确定
'''
template['source'] = "Based on the following purchase history of user_{} in chronological order: \n {} \n Please generate a recommendation for the next item to be purchased by the user."
template['target'] = "{}"
template['task'] = "text"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item']
template['id'] = "3-6"

task_subgroup_3["3-6"] = template

template = {}
'''
Input template:
Based on the purchase history of user_{} in chronological order:
{{history item list of {{item}}}}, 
user's nationality: {county}, and category preferences: {interest_category}.
Please generate a recommendation for the next item to be purchased by the user.

Target template:
{{item [item]}}


Metrics:
未确定
'''
template['source'] = "Based on the purchase history of user_{} in chronological order: \n {}, \n user's nationality: {}, and category preferences: {}. \nPlease generate a recommendation for the next item to be purchased by the user."
template['target'] = "{}"
template['task'] = "text"
template['source_argc'] = 4
template['source_argv'] = ['user_id', 'purchase_history', 'county', 'interest_category']
template['target_argc'] = 1
template['target_argv'] = ['item']
template['id'] = "3-7"

task_subgroup_3["3-7"] = template

template = {}
'''
Input template:
I have the purchase history list of user_{} in chronological order::
{{history item list of {{item}}}}, 
user's nationality: {county}, and category preferences: {interest_category}.
Can you help me generate a recommendation based on the user's recent purchase history?

Target template:
{{item [item]}}


Metrics:
未确定
'''
template['source'] = "I have the purchase history list of user_{} in chronological order: \n {}, \n user's nationality: {}, and category preferences: {}. \n Can you help me generate a recommendation based on the user's recent purchase history?"
template['target'] = "{}"
template['task'] = "text"
template['source_argc'] = 4
template['source_argv'] = ['user_id', 'purchase_history', 'county', 'interest_category']
template['target_argc'] = 1
template['target_argv'] = ['item']
template['id'] = "3-8"

task_subgroup_3["3-8"] = template

template = {}
'''
Input template:
Here is the purchase history list of user_{} in reverse chronological order:
{{history item list of {{item}}}}, 
user's nationality: {county}, and category preferences: {interest_category}.
Please generate a recommendation for the next item to recommend to the user based on their recent purchases.

Target template:
{{item [item]}}


Metrics:
未确定
'''
template['source'] = "Here is the purchase history list of user_{} in reverse chronological order: \n {}, \n user's nationality: {}, and category preferences: {}.\n Please generate a recommendation for the next item to recommend to the user based on their recent purchases."
template['target'] = "{}"
template['task'] = "text"
template['source_argc'] = 4
template['source_argv'] = ['user_id', 'purchase_history', 'county', 'interest_category']
template['target_argc'] = 1
template['target_argv'] = ['item']
template['id'] = "3-9"

task_subgroup_3["3-9"] = template

template = {}
'''
Input template:
Based on the facts, please generate a recommendation for the next item to be purchased by the user.
User ID: {ID}
Nationality: {county} 
Query category preferences: {interest_category} 
Purchase history: {{history item list of {{item}}}} 


Target template:
{{item [item]}}


Metrics:
未确定
'''
template['source'] = "Based on the facts, please generate a recommendation for the next item to be purchased by the user. \nUser ID: {} \nNationality: {} \nQuery category preferences: {} \nPurchase history: {}."
template['target'] = "{}"
template['task'] = "text"
template['source_argc'] = 4
template['source_argv'] = ['user_id', 'county', 'interest_category', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item']
template['time_emb'] = True
template['id'] = "3-10"

task_subgroup_3["3-10"] = template

template = {}
'''
Input template:
Can you help me generate a recommendation based on the user's information?
User ID: {ID}
Nationality: {county} 
Query category preferences: {interest_category} 
Purchase history: {{history item list of {{item}}}} 


Target template:
{{item [item]}}


Metrics:
未确定
'''
template['source'] = "Can you help me generate a recommendation based on the user's information? \nUser ID: {} \nNationality: {} \nQuery category preferences: {} \nPurchase history: {}."
template['target'] = "{}"
template['task'] = "text"
template['source_argc'] = 4
template['source_argv'] = ['user_id', 'county', 'interest_category', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item']
template['time_emb'] = True
template['id'] = "3-11"

task_subgroup_3["3-11"] = template

template = {}
'''
Input template:
Please generate a recommendation for the next item to recommend to the user based on his information.
User ID: {ID}
Nationality: {county} 
Query category preferences: {interest_category} 
Purchase history: {{history item list of {{item}}}} 

Target template:
{{item [item]}}


Metrics:
未确定
'''
template['source'] = "Please generate a recommendation for the next item to recommend to the user based on his information. \nUser ID: {} \nNationality: {} \nQuery category preferences: {} \nPurchase history: {}."
template['target'] = "{}"
template['task'] = "text"
template['source_argc'] = 4
template['source_argv'] = ['user_id', 'county', 'interest_category', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item']
template['time_emb'] = True
template['id'] = "3-12"

task_subgroup_3["3-12"] = template

template = {}
'''
Input template:
Based on the facts, please generate a recommendation for the next item to be purchased by the user.
User ID: {ID}
Nationality: {county} 
Query category preferences: {interest_category} 


Target template:
{{item [item]}}


Metrics:
未确定
'''
template['source'] = "Based on the facts, please generate a recommendation for the next item to be purchased by the user. \nUser ID: {} \nNationality: {} \nQuery category preferences: {} \n"
template['target'] = "{}"
template['task'] = "text"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'county', 'interest_category']
template['target_argc'] = 1
template['target_argv'] = ['item']
template['id'] = "3-13"

task_subgroup_3["3-13"] = template

template = {}
'''
Input template:
Can you help me generate a recommendation based on the user's information?
User ID: {ID}
Purchase history: {{history item list of {{item}}}} 


Target template:
{{item [item]}}


Metrics:
未确定
'''
template['source'] = "Can you help me generate a recommendation based on the user's information? \nUser ID: {} \nPurchase history: {}."
template['target'] = "{}"
template['task'] = "text"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item']
template['time_emb'] = True
template['id'] = "3-14"

task_subgroup_3["3-14"] = template

all_tasks['text'] = task_subgroup_3

# =====================================================
# Task Subgroup 4 -- Query Auto-Complete(QAC) -- 3 Prompts
# =====================================================

task_subgroup_4 = {}

template = {}

'''
Input template:
Given the purchase history of user {{user_id}}：
{{list of item_id}}
and current partial input {{prefix}}, complete the input query to reflect user's purchase interest.

Target template:
{{item [item_id]}}


Metrics:
未确定
'''

template[
    'source'] = "Given the following purchase history of user_{} : \n {} \n and current partial input {}, complete the input query to reflect user's purchase interest."
template['target'] = "{}"
template['task'] = "QAC"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'purchase_history', 'prefix']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "4-1"

task_subgroup_4["4-1"] = template

template = {}
'''
Input template:
I find the purchase history list of user {{user_id}}:
{{history item list of {{item_id}}}}
and current partial input {{prefix}}. 
I wonder which is the next item the user wants. Can you complete the input query?


Target template:
{{item [item_id]}}


Metrics:
未确定
'''
template[
    'source'] = "I find the purchase history list of user_{} : \n {} \n and current partial input {}. I wonder which is the next item the user wants. Can you complete the input query?"
template['target'] = "{}"
template['task'] = "QAC"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'purchase_history', 'prefix']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "4-2"

task_subgroup_4["4-2"] = template

template = {}
'''
Input template:
Here is the purchase history list of user {{user_id}}:
{{history item list of {{item_id}}}}
and current partial input {{prefix}}. 
try to complete the input query for the user.

Target template:
{{item [item_id]}}


Metrics:
'''
template['source'] = "Here is the purchase history list of user_{} : \n {} \n and current partial input {}. Try to complete the input query for the user."
template['target'] = "{}"
template['task'] = "QAC"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'purchase_history', 'prefix']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "4-3"

task_subgroup_4["4-3"] = template

template = {}

'''
Input template:
Given the purchase interest category of user {{user_id}}：
{{list of item_id}}, 
user's nationality: {county} and current partial input {{prefix}}, complete the input query to reflect user's purchase interest.

Target template:
{{item [item_id]}}


Metrics:
未确定
'''

template[
    'source'] = "Given the purchase interest category of user_{} : \n {}, \n user's nationality: {} and current partial input {}, \n complete the input query to reflect user's purchase interest."
template['target'] = "{}"
template['task'] = "QAC"
template['source_argc'] = 4
template['source_argv'] = ['user_id', 'interest_category', 'county', 'prefix']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "4-4"

task_subgroup_4["4-4"] = template

template = {}
'''
Input template:
I find the purchase interest category list of user {{user_id}}:
{{history item list of {{item_id}}}}, 
user's nationality: {county} and current partial input {{prefix}}. 
I wonder which is the next item the user wants. Can you complete the input query?


Target template:
{{item [item_id]}}


Metrics:
未确定
'''
template[
    'source'] = "I find the purchase interest category list of user_{} : \n {}, \n user's nationality: {} and current partial input {}. \n I wonder which is the next item the user wants. Can you complete the input query?"
template['target'] = "{}"
template['task'] = "QAC"
template['source_argc'] = 4
template['source_argv'] = ['user_id', 'interest_category', 'county', 'prefix']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "4-5"

task_subgroup_4["4-5"] = template

template = {}
'''
Input template:
Here is the purchase interest category list of user {{user_id}}:
{{history item list of {{item_id}}}}, 
user's nationality: {county} and current partial input {{prefix}}. 
try to complete the input query for the user.

Target template:
{{item [item_id]}}


Metrics:
'''
template['source'] = "Here is the purchase interest category list of user_{} : \n {}, \n user's nationality: {} and current partial input {}. \n Try to complete the input query for the user."
template['target'] = "{}"
template['task'] = "QAC"
template['source_argc'] = 4
template['source_argv'] = ['user_id', 'interest_category', 'county', 'prefix']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "4-6"

task_subgroup_4["4-6"] = template

template = {}

'''
Input template:
Complete the input query to reflect user's purchase interest.
Current query: {prefix}
User ID: {ID}
Nationality: {county} 
Query Category preferences: {interest_category} 

Target template:
{{item [item_id]}}


Metrics:
未确定
'''

template[
    'source'] = "Complete the input query to reflect user's purchase interest. \nCurrent query: {} \nUser ID: {} \nNationality: {} \nQuery category preferences: {} \n"
template['target'] = "{}"
template['task'] = "QAC"
template['source_argc'] = 4
template['source_argv'] = ['prefix', 'user_id', 'county', 'interest_category']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "4-7"

task_subgroup_4["4-7"] = template

template = {}
'''
Input template:
I wonder which is the next item the user wants. Can you use the user information to complete the input query?
Current query: {prefix}
User ID: {ID}
Nationality: {county} 
Query category preferences: {interest_category} 

Target template:
{{item [item_id]}}


Metrics:
未确定
'''
template[
    'source'] = "I wonder which is the next item the user wants. Can you use the user information to complete the input query? \nCurrent query: {} \nUser ID: {} \nNationality: {} \nQuery category preferences: {}"
template['target'] = "{}"
template['task'] = "QAC"
template['source_argc'] = 4
template['source_argv'] = ['prefix', 'user_id', 'county', 'interest_category']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "4-8"

task_subgroup_4["4-8"] = template

template = {}
'''
Input template:
Try to complete the input query for the user based on the user information.
Current query: {prefix}
User ID: {ID}
Nationality: {county} 
Query category preferences: {interest_category} 

Target template:
{{item [item_id]}}


Metrics:
'''
template['source'] = "Try to complete the input query for the user based on the user information. \nCurrent query: {} \nUser ID: {} \nNationality: {} \nQuery category preferences: {} "
template['target'] = "{}"
template['task'] = "QAC"
template['source_argc'] = 4
template['source_argv'] = ['prefix', 'user_id', 'county', 'interest_category']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "4-9"

task_subgroup_4["4-9"] = template

template = {}
'''
Input template:
Complete the input query to reflect user's purchase interest.
Current query: {prefix}
User ID: {ID}

Target template:
{{item [item_id]}}


Metrics:
未确定
'''

template[
    'source'] = "Complete the input query to reflect user's purchase interest. \nCurrent query: {} \nUser ID: {} \n"
template['target'] = "{}"
template['task'] = "QAC"
template['source_argc'] = 2
template['source_argv'] = ['prefix', 'user_id']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "4-10"

task_subgroup_4["4-10"] = template

all_tasks['QAC'] = task_subgroup_4


# =====================================================
# Task Subgroup 5 -- Related Query Recommendation(Q2Q) -- 3 Prompts
# =====================================================

task_subgroup_5 = {}

template = {}

'''
Input template:
Given the purchase history of user {{user_id}}：
{{list of item_id}}
and current query {{query_trigger}}, please recommend the most related query that the user may click on.

Target template:
{{item [item_id]}}


Metrics:
未确定
'''

template[
    'source'] = "Given the following purchase history of user_{} : \n {} \n and current query {}, please recommend the most related query that the user may click on."
template['target'] = "{}"
template['task'] = "Q2Q"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'purchase_history', 'query_trigger']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "5-1"

task_subgroup_5["5-1"] = template

template = {}
'''
Input template:
Based on the purchase history list of user {{user_id}}:
{{history item list of {{item_id}}}}
and current query {{query_trigger}}, 
suggest the most related query that the user may find interesting.


Target template:
{{item [item_id]}}


Metrics:
未确定
'''
template[
    'source'] = "Based on the purchase history list of user_{} : \n {} \n and current query {}, suggest the most related query that the user may find interesting."
template['target'] = "{}"
template['task'] = "Q2Q"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'purchase_history', 'query_trigger']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "5-2"

task_subgroup_5["5-2"] = template

template = {}
'''
Input template:
Using the purchase history list of user {{user_id}}:
{{history item list of {{item_id}}}}
and current query {{query_trigger}}, 
generate the most related query that the user may be interested in clicking on.

Target template:
{{item [item_id]}}


Metrics:
'''
template['source'] = "Using the purchase history list of user_{} : \n {} \n and current partial input {}. Generate the most related query that the user may be interested in clicking on."
template['target'] = "{}"
template['task'] = "Q2Q"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'purchase_history', 'query_trigger']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "5-3"

task_subgroup_5["5-3"] = template

template = {}
'''
Input template:
Given the purchase history list of user {{user_id}}:
{{history item list of {{item_id}}}}
and current query {{query_trigger}}, 
provide recommendations for the most related query that the user may want to explore further.

Target template:
{{item [item_id]}}


Metrics:
'''
template['source'] = "Given the purchase history list of user_{} : \n {} \n and current partial input {}, provide recommendations for the most related query that the user may want to explore further."
template['target'] = "{}"
template['task'] = "Q2Q"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'purchase_history', 'query_trigger']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "5-4"

task_subgroup_5["5-4"] = template

template = {}

'''
Input template:
Given the purchase history of user {{user_id}}：
{{list of item_id}}, 
user's nationality: {county}, category preferences: {interest_category} 
and current query {{query_trigger}}, 
please recommend the most related query that the user may click on.

Target template:
{{item [item_id]}}


Metrics:
未确定
'''

template[
    'source'] = "Given the following purchase history of user_{} : \n {} \n user's nationality: {}, category preferences: {} and current query {}, \n please recommend the most related query that the user may click on."
template['target'] = "{}"
template['task'] = "Q2Q"
template['source_argc'] = 5
template['source_argv'] = ['user_id', 'purchase_history', 'county', 'interest_category', 'query_trigger']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "5-5"

task_subgroup_5["5-5"] = template

template = {}
'''
Input template:
Based on the purchase history list of user {{user_id}}:
{{history item list of {{item_id}}}}, 
user's nationality: {county}, category preferences: {interest_category} 
and current query {{query_trigger}}, 
suggest the most related query that the user may find interesting.


Target template:
{{item [item_id]}}


Metrics:
未确定
'''
template[
    'source'] = "Based on the purchase history list of user_{} : \n {}, \n user's nationality: {}, category preferences: {} and current query {}, \n suggest the most related query that the user may find interesting."
template['target'] = "{}"
template['task'] = "Q2Q"
template['source_argc'] = 5
template['source_argv'] = ['user_id', 'purchase_history', 'county', 'interest_category', 'query_trigger']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "5-6"

task_subgroup_5["5-6"] = template

template = {}
'''
Input template:
Using the purchase history list of user {{user_id}}:
{{history item list of {{item_id}}}}, 
user's nationality: {county}, category preferences: {interest_category} 
and current query {{query_trigger}}, 
generate the most related query that the user may be interested in clicking on.

Target template:
{{item [item_id]}}


Metrics:
'''
template['source'] = "Using the purchase history list of user_{} : \n {}, \n user's nationality: {}, category preferences: {} and current partial input {}. \n Generate the most related query that the user may be interested in clicking on."
template['target'] = "{}"
template['task'] = "Q2Q"
template['source_argc'] = 5
template['source_argv'] = ['user_id', 'purchase_history', 'county', 'interest_category', 'query_trigger']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "5-7"

task_subgroup_5["5-7"] = template

template = {}
'''
Input template:
Given the purchase history list of user {{user_id}}:
{{history item list of {{item_id}}}}, 
user's nationality: {county}, category preferences: {interest_category} 
and current query {{query_trigger}}, 
please provide recommendations for the most related query that the user may want to explore further.

Target template:
{{item [item_id]}}


Metrics:
'''
template['source'] = "Given the purchase history list of user_{} : \n {}, \n user's nationality: {}, category preferences: {} and current partial input {}, \n please provide recommendations for the most related query that the user may want to explore further."
template['target'] = "{}"
template['task'] = "Q2Q"
template['source_argc'] = 5
template['source_argv'] = ['user_id', 'purchase_history', 'county', 'interest_category', 'query_trigger']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "5-8"

task_subgroup_5["5-8"] = template

template = {}
'''
Input template:
Suggest the most related query that the user may find interesting.
Current input query {{query_trigger}} 
User ID: {ID}
Nationality: {county} 
Query category preferences: {interest_category} 
Purchase history: {{history item list of {{item}}}} 


Target template:
{{item [item_id]}}


Metrics:
未确定
'''
template[
    'source'] = "Suggest the most related query that the user may find interesting. \nCurrent input query {} \nUser ID: {} \nNationality: {} \nQuery category preferences: {} \nPurchase history: {} "
template['target'] = "{}"
template['task'] = "Q2Q"
template['source_argc'] = 5
template['source_argv'] = ['query_trigger', 'user_id', 'county', 'interest_category', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['time_emb'] = True
template['id'] = "5-9"

task_subgroup_5["5-9"] = template

template = {}
'''
Input template:
Generate the most related query that the user may be interested in clicking on.
Current query {{query_trigger}} 
User ID: {ID}
Nationality: {county} 
Query category preferences: {interest_category} 
Purchase history: {{history item list of {{item}}}} 

Target template:
{{item [item_id]}}


Metrics:
'''
template['source'] = "Suggest the most related query that the user may find interesting. \nCurrent query {} \nUser ID: {} \nNationality: {} \nQuery category preferences: {} \nPurchase history: {} "
template['target'] = "{}"
template['task'] = "Q2Q"
template['source_argc'] = 5
template['source_argv'] = ['query_trigger', 'user_id', 'county', 'interest_category', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['time_emb'] = True
template['id'] = "5-10"

task_subgroup_5["5-10"] = template

template = {}
'''
Input template:
Please provide recommendations for the most related query that the user may want to explore further.
Current query {{query_trigger}} 
User ID: {ID}
Nationality: {county} 
Query category preferences: {interest_category} 
Purchase history: {{history item list of {{item}}}} 

Target template:
{{item [item_id]}}


Metrics:
'''
template['source'] = "Please provide recommendations for the most related query that the user may want to explore further. \nCurrent query {} \nUser ID: {} \nNationality: {} \nQuery category preferences: {} \nPurchase history: {} "
template['target'] = "{}"
template['task'] = "Q2Q"
template['source_argc'] = 5
template['source_argv'] = ['query_trigger', 'user_id', 'county', 'interest_category', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['time_emb'] = True
template['id'] = "5-11"

task_subgroup_5["5-11"] = template

template = {}
'''
Input template:
Suggest the most related query that the user may find interesting.
Current input query {{query_trigger}} 
User ID: {ID}
Nationality: {county} 
Query category preferences: {interest_category} 

Target template:
{{item [item_id]}}


Metrics:
未确定
'''
template[
    'source'] = "Suggest the most related query that the user may find interesting. \nCurrent input query {} \nUser ID: {} \nNationality: {} \nQuery category preferences: {} "
template['target'] = "{}"
template['task'] = "Q2Q"
template['source_argc'] = 4
template['source_argv'] = ['query_trigger', 'user_id', 'county', 'interest_category']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['time_emb'] = True
template['id'] = "5-12"

task_subgroup_5["5-12"] = template

template = {}
'''
Input template:
Please provide recommendations for the most related query that the user may want to explore further.
Current query {{query_trigger}} 
Purchase history: {{history item list of {{item}}}} 

Target template:
{{item [item_id]}}


Metrics:
'''
template['source'] = "Please provide recommendations for the most related query that the user may want to explore further. \nCurrent query {} \nUser ID: {} \nNationality: {} \nQuery category preferences: {} \nPurchase history: {} "
template['target'] = "{}"
template['task'] = "Q2Q"
template['source_argc'] = 5
template['source_argv'] = ['query_trigger', 'user_id', 'county', 'interest_category', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['time_emb'] = True
template['id'] = "5-13"

task_subgroup_5["5-13"] = template


all_tasks['Q2Q'] = task_subgroup_5

# =====================================================
# Task Subgroup 6 -- Item to Query Recommendation(I2Q) -- 3 Prompts
# =====================================================

task_subgroup_6 = {}

template = {}

'''
Input template:
Based on the product information provided, generate a query that users might be interested in.
The product ID is {ID}, the product title is {title}, and the product keywords are {keywords}.
Consider the user's search intent and try to incorporate the product keywords into the query to improve the match.


Target template:
{{item [item_id]}}


Metrics:
未确定
'''

template[
    'source'] = "Based on the product information provided, generate a query that users might be interested in. \n The product ID is {}, the product title is {}, and the product keywords are {}. \n Consider the user's search intent and try to incorporate the product keywords into the query to improve the match."
template['target'] = "{}"
template['task'] = "I2Q"
template['source_argc'] = 3
template['source_argv'] = ['ID', 'title', 'keywords']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "6-1"

task_subgroup_6["6-1"] = template

template = {}

'''
Input template:
You have been given the product ID {ID}, title {title}, and keywords {keywords}.
Your task is to generate a query that users might be interested in.
Keep in mind the user's search intent and try to use the product keywords to improve the match.


Target template:
{{item [item_id]}}


Metrics:
未确定
'''

template[
    'source'] = "You have been given the product ID {}, title {}, and keywords {}. \n Your task is to generate a query that users might be interested in. \n Keep in mind the user's search intent and try to use the product keywords to improve the match."
template['target'] = "{}"
template['task'] = "I2Q"
template['source_argc'] = 3
template['source_argv'] = ['ID', 'title', 'keywords']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "6-2"

task_subgroup_6["6-2"] = template

template = {}

'''
Input template:
Generate a query that users might be interested in based on the given product information.
The product ID is {ID}, the product title is {title}, and the product keywords are {keywords}.
Try to incorporate the product keywords into the query to improve the match and consider the user's search intent.


Target template:
{{item [item_id]}}


Metrics:
未确定
'''

template[
    'source'] = "Generate a query that users might be interested in based on the given product information. \n The product ID is {}, the product title is {}, and the product keywords are {}. \n Try to incorporate the product keywords into the query to improve the match and consider the user's search intent."
template['target'] = "{}"
template['task'] = "I2Q"
template['source_argc'] = 3
template['source_argv'] = ['ID', 'title', 'keywords']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "6-3"

task_subgroup_6["6-3"] = template

template = {}

'''
Input template:
Based on the product information provided, generate a query that users might be interested in.
Product ID: {ID} 
Product title: {title} 
Product keywords: {keywords} 

Target template:
{{item [item_id]}}


Metrics:
未确定
'''

template[
    'source'] = "Based on the product information provided, generate a query that users might be interested in. \nProduct ID: {} \nProduct title {} \nProduct keywords: {}. "
template['target'] = "{}"
template['task'] = "I2Q"
template['source_argc'] = 3
template['source_argv'] = ['ID', 'title', 'keywords']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "6-4"

task_subgroup_6["6-4"] = template

template = {}

'''
Input template:
Given the product information, generate a query that users might be interested in.
Product ID: {ID} 
Product title: {title} 
Product keywords: {keywords} 

Target template:
{{item [item_id]}}


Metrics:
未确定
'''

template[
    'source'] = "Given the product information, generate a query that users might be interested in. \nProduct ID: {} \nProduct title {} \nProduct keywords: {}. "
template['target'] = "{}"
template['task'] = "I2Q"
template['source_argc'] = 3
template['source_argv'] = ['ID', 'title', 'keywords']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "6-5"

task_subgroup_6["6-5"] = template

template = {}

'''
Input template:
Generate a query that users might be interested in based on the given product information.
Product ID: {ID} 
Product title: {title} 
Product keywords: {keywords} 

Target template:
{{item [item_id]}}


Metrics:
未确定
'''

template[
    'source'] = "Generate a query that users might be interested in based on the given product information. \nProduct ID: {} \nProduct title {} \nProduct keywords: {}. "
template['target'] = "{}"
template['task'] = "I2Q"
template['source_argc'] = 3
template['source_argv'] = ['ID', 'title', 'keywords']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "6-6"

task_subgroup_6["6-6"] = template

all_tasks['I2Q'] = task_subgroup_6

# =====================================================
# Task Subgroup 7 -- U2Q with context(U2QC) -- 3 Prompts
# =====================================================

task_subgroup_7= {}

template = {}
'''
Input template:
Here is the purchase history list of user_{} in reverse chronological order:
{{history item list of {{item}}}}
Note that the user has input {prefix}, then he clicked the related query {QAC_query}. 
And the user has searched {query_trigger}, then he clicked the related query {Q2Q_query}. 
Please generate a recommendation for the next item to recommend to the user based on their recent purchases.

Target template:
{{item [item]}}


Metrics:
未确定
'''
template['source'] = "Here is the purchase history list of user_{} in reverse chronological order: \n {} \n Note that the user has input {}, then he clicked the related query {}. \n And the user has searched {}, then he clicked the related query {}. \n Please generate a recommendation for the next item to recommend to the user based on their recent purchases."
template['target'] = "{}"
template['task'] = "U2QC"
template['source_argc'] = 6
template['source_argv'] = ['user_id', 'purchase_history', 'prefix', 'QAC_query', 'query_trigger', 'Q2Q_query']
template['target_argc'] = 1
template['target_argv'] = ['item']
template['id'] = "7-1"

task_subgroup_7["7-1"] = template

template = {}
'''
Input template:
I have the purchase history list of user_{} in chronological order::
{{history item list of {{item}}}}
Note that the user has input {prefix}, then he clicked the related query {QAC_query}. 
Can you help me generate a recommendation based on the user's recent purchase history and context information?

Target template:
{{item [item]}}


Metrics:
未确定
'''
template['source'] = "I have the purchase history list of user_{} in chronological order: \n {} \n Note that the user has input {}, then he clicked the related query {}. \n Can you help me generate a recommendation based on the user's recent purchase history?"
template['target'] = "{}"
template['task'] = "U2QC"
template['source_argc'] = 4
template['source_argv'] = ['user_id', 'purchase_history', 'prefix', 'QAC_query']
template['target_argc'] = 1
template['target_argv'] = ['item']
template['id'] = "7-2"

task_subgroup_7["7-2"] = template

template = {}
'''
Input template:
Based on the following purchase history of user_{} in chronological order:
{{history item list of {{item}}}}.
Note that the user has searched {query_trigger}, then he clicked the related query {Q2Q_query}. 
Please use the above context information to generate a recommendation for the next item to be purchased by the user.

Target template:
{{item [item]}}


Metrics:
未确定
'''
template['source'] = "Based on the following purchase history of user_{} in chronological order: \n {} \n Note that the user has searched {}, then he clicked the related query {}. \n Please use the above context information  generate a recommendation for the next item to be purchased by the user."
template['target'] = "{}"
template['task'] = "U2QC"
template['source_argc'] = 4
template['source_argv'] = ['user_id', 'purchase_history', 'query_trigger', 'Q2Q_query']
template['target_argc'] = 1
template['target_argv'] = ['item']
template['id'] = "7-3"

task_subgroup_7["7-3"] = template


all_tasks['U2QC'] = task_subgroup_7

# =====================================================
# Task Subgroup 8 -- Q2C -- 2 Prompts
# =====================================================

task_subgroup_8 = {}

template = {}
'''
Input template:
Please predict the category information of the given query: {query}.
Note that the query may correspond to a single or multiple categories.



Target template:
{{item [item]}}


Metrics:
未确定
'''
template['source'] = "Please predict the category information of the given query: {}."
template['target'] = "{}"
template['task'] = "Q2C"
template['source_argc'] = 1
template['source_argv'] = ['query']
template['target_argc'] = 1
template['target_argv'] = ['category']
template['id'] = "8-1"

task_subgroup_8["8-1"] = template

template = {}
'''
Input template:
Please predict the category label of the given query: {query}.
Note that the query may correspond to a single or multiple categories.


Target template:
{{item [item]}}


Metrics:
未确定
'''
template['source'] = "Please predict the category label of the given query: {}"
template['target'] = "{}"
template['task'] = "Q2C"
template['source_argc'] = 1
template['source_argv'] = ['query']
template['target_argc'] = 1
template['target_argv'] = ['category']
template['id'] = "8-2"

task_subgroup_8["8-2"] = template

template = {}
'''
Input template:
Please predict the query category of the given query. 
Current query: {query_trigger}

Target template:
{{item [item]}}


Metrics:
未确定
'''
template['source'] = "Please predict the query category of the given query. \nCurrent query: {}"
template['target'] = "{}"
template['task'] = "Q2C"
template['source_argc'] = 1
template['source_argv'] = ['query']
template['target_argc'] = 1
template['target_argv'] = ['category']
template['id'] = "8-3"

task_subgroup_8["8-3"] = template

template = {}
'''
Input template:
Generate the query category of the given query. 
Current query: {query_trigger}

Target template:
{{item [item]}}


Metrics:
未确定
'''
template['source'] = "Generate the query category of the given query. \nCurrent query: {}"
template['target'] = "{}"
template['task'] = "Q2C"
template['source_argc'] = 1
template['source_argv'] = ['query']
template['target_argc'] = 1
template['target_argv'] = ['category']
template['id'] = "8-4"

task_subgroup_8["8-4"] = template

template = {}
'''
Input template:
Choose the best query category from the candidates for the query
Current query: {query_trigger}
Candidates: {{candidate {{item_id}}}}

Target template:
{{groundtruth {{item ids}}}}


Metrics:
未确定
'''
template['source'] = "Choose the best query category from the candidates for the query. \nCurrent query: {} \nCandidates: {}"
template['target'] = "{}"
template['task'] = "Q2C"
template['source_argc'] = 2
template['source_argv'] = ['query', 'candidate']
template['target_argc'] = 1
template['target_argv'] = ['category']
template['id'] = "8-5"

task_subgroup_8["8-5"] = template

all_tasks['Q2C'] = task_subgroup_8


# =====================================================
# Task Subgroup 9 -- U2C -- 2 Prompts
# =====================================================

task_subgroup_9 = {}

template = {}
'''
Input template:
Please predict the query category preferences of the given user. Note that the user may be interested in single or multiple categories.
User ID: {ID}

Target template:
{{item [item]}}


Metrics:
未确定
'''
template['source'] = "Please predict the query category preferences of the given user. Note that the user may be interested in single or multiple categories. \nUser ID: {} "
template['target'] = "{}"
template['task'] = "U2C"
template['source_argc'] = 1
template['source_argv'] = ['user_id']
template['target_argc'] = 1
template['target_argv'] = ['category']
template['id'] = "9-1"

task_subgroup_9["9-1"] = template

template = {}
'''
Input template:
Generate the query category preferences of the given user. Note that the user may be interested in single or multiple categories.
User ID: {ID}

Target template:
{{item [item]}}


Metrics:
未确定
'''
template['source'] = "Generate the query category preferences of the given user. Note that the user may be interested in single or multiple categories. \nUser ID: {}"
template['target'] = "{}"
template['task'] = "U2C"
template['source_argc'] = 1
template['source_argv'] = ['user_id']
template['target_argc'] = 1
template['target_argv'] = ['category']
template['id'] = "9-2"

task_subgroup_9["9-2"] = template

template = {}
'''
Input template:
Please predict the query category of [M] from user's query category preferences list. 
User ID: {ID}
Query category preferences: {interest_category} 

Target template:
{{item [item]}}


Metrics:
未确定
'''
template['source'] = "Please predict the query category of [M] from user's query category preferences list. \nUser ID: {} \nQuery category preferences: {} "
template['target'] = "{}"
template['task'] = "U2C"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'interest_category']
template['target_argc'] = 1
template['target_argv'] = ['category']
template['id'] = "9-3"

task_subgroup_9["9-3"] = template

template = {}
'''
Input template:
Generate the query category of [M] from user's query category preferences list. 
User ID: {ID}
Query category preferences: {interest_category} 

Target template:
{{item [item]}}


Metrics:
未确定
'''
template['source'] = "Generate the query category of [M] from user's query category preferences list. \nUser ID: {} \nQuery category preferences: {} "
template['target'] = "{}"
template['task'] = "U2C"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'interest_category']
template['target_argc'] = 1
template['target_argv'] = ['category']
template['id'] = "9-4"

task_subgroup_9["9-4"] = template

all_tasks['U2C'] = task_subgroup_9

# =====================================================
# Task Subgroup 10 -- traditional -- 2 Prompts
# =====================================================

task_subgroup_10 = {}

template = {}
'''
Input template:
Based on the facts, will the user be interested in the given query?
Given query: {}
User ID: {ID}
Nationality: {county} 
Query category preferences: {interest_category} 

Target template:
{{item [item]}}


Metrics:
未确定
'''
template['source'] = "Based on the facts, will the user be interested in the given query? \nGiven query: {} \nUser ID: {} \nNationality: {} \nQuery category preferences: {} "
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 4
template['source_argv'] = ['query trigger', 'user_id', 'county', 'interest_category']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "10-1"

task_subgroup_10["10-1"] = template

template = {}
'''
Input template:
Based on the facts, do you think it is good to recommend the query to the user?
Given query: {}
User ID: {ID}
Nationality: {county} 
Query category preferences: {interest_category} 

Target template:
{{item [item]}}

Metrics:
未确定
'''
template['source'] = "Based on the facts, do you think it is good to recommend the query to the user? \nGiven query: {} \nUser ID: {} \nNationality: {} \nQuery category preferences: {} "
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 4
template['source_argv'] = ['query trigger', 'user_id', 'county', 'interest_category']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "10-2"

task_subgroup_10["10-2"] = template

template = {}
'''
Input template:
Which of the following query will be recommend for the user?
Query list: {query_list}
User ID: {ID}
Nationality: {county} 
Query category preferences: {interest_category} 

Target template:
{{item [item_id]}}


Metrics:
'''
template['source'] = "Which of the following query will be recommend for the user? \nQuery list: {} \nUser ID: {} \nNationality: {} \nQuery category preferences: {} "
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 4
template['source_argv'] = ['query_list', 'user_id', 'county', 'interest_category']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "10-3"

task_subgroup_10["10-3"] = template

template = {}
'''
Input template:
Choose the best query from the candidates to recommend for the user.
Query list: {query_list}
User ID: {ID}
Nationality: {county} 
Query category preferences: {interest_category} 

Target template:
{{item [item_id]}}


Metrics:
未确定
'''
template[
    'source'] = "Choose the best query from the candidates to recommend for the user. \nQuery list: {} \nUser ID: {} \nNationality: {} \nQuery category preferences: {} "
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 4
template['source_argv'] = ['query_list', 'user_id', 'county', 'interest_category']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "10-4"

task_subgroup_10["10-4"] = template

all_tasks['traditional'] = task_subgroup_10
