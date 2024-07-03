# =====================================================
# Task Subgroup 3 -- Sequential Text generation（Text） -- 6 Prompts
# =====================================================


def geticbutemplate():
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

    template = {}
    '''
    Input template:
    Based on the facts, can you generate a recommendation for the user?
    User ID: {ID}
    Nationality: {county} 
    Query category preferences: {interest_category} 


    Target template:
    {{item [item]}}


    Metrics:
    未确定
    '''
    template[
        'source'] = "Based on the facts, can you generate a recommendation for the user? \nUser ID: {} \nNationality: {} \nQuery category preferences: {} \n"
    template['target'] = "{}"
    template['task'] = "text"
    template['source_argc'] = 3
    template['source_argv'] = ['user_id', 'county', 'interest_category']
    template['target_argc'] = 1
    template['target_argv'] = ['item']
    template['id'] = "3-15"

    task_subgroup_3["3-15"] = template

    template = {}
    '''
    Input template:
    Given the purchase history, please generate a recommendation for the user.
    User ID: {ID}
    Purchase history: {{history item list of {{item}}}} 


    Target template:
    {{item [item]}}


    Metrics:
    未确定
    '''
    template[
        'source'] = "Given the purchase history, please generate a recommendation for the user. \nUser ID: {} \nPurchase history: {}."
    template['target'] = "{}"
    template['task'] = "text"
    template['source_argc'] = 2
    template['source_argv'] = ['user_id', 'purchase_history']
    template['target_argc'] = 1
    template['target_argv'] = ['item']
    template['time_emb'] = True
    template['id'] = "3-16"

    task_subgroup_3["3-16"] = template

    return task_subgroup_3


task_subgroup = geticbutemplate()
