# =====================================================
# Task Subgroup 3 -- Sequential Text generation（Text） -- 6 Prompts
# =====================================================


def geticbutemplate():
    task_subgroup_3 = {}

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
    template['task'] = "PRO"
    template['source_argc'] = 4
    template['source_argv'] = ['user_id', 'county', 'interest_category', 'purchase_history']
    template['target_argc'] = 1
    template['target_argv'] = ['item']
    template['time_emb'] = False
    template['id'] = "13-1"

    task_subgroup_3["13-1"] = template

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
    template['task'] = "PRO"
    template['source_argc'] = 4
    template['source_argv'] = ['user_id', 'county', 'interest_category', 'purchase_history']
    template['target_argc'] = 1
    template['target_argv'] = ['item']
    template['time_emb'] = False
    template['id'] = "13-2"

    task_subgroup_3["13-2"] = template

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
    template['task'] = "PRO"
    template['source_argc'] = 4
    template['source_argv'] = ['user_id', 'county', 'interest_category', 'purchase_history']
    template['target_argc'] = 1
    template['target_argv'] = ['item']
    template['time_emb'] = False
    template['id'] = "13-3"

    task_subgroup_3["13-3"] = template

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
    template['task'] = "PRO"
    template['source_argc'] = 3
    template['source_argv'] = ['user_id', 'county', 'interest_category']
    template['target_argc'] = 1
    template['target_argv'] = ['item']
    template['id'] = "13-4"

    task_subgroup_3["13-4"] = template

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
    template['task'] = "PRO"
    template['source_argc'] = 2
    template['source_argv'] = ['user_id', 'purchase_history']
    template['target_argc'] = 1
    template['target_argv'] = ['item']
    template['time_emb'] = False
    template['id'] = "13-5"

    task_subgroup_3["13-5"] = template

    template = {}
    '''
    Input template:
    Check the spelling of the given query. Note that 1 stands for good and 0 stands for bad.
    Given query: {query}

    Target template:
    {{item [item]}}


    Metrics:
    未确定
    '''
    template[
        'source'] = "Check the spelling of the given query. Note that 1 stands for good and 0 stands for bad. \nGiven query: {}"
    template['target'] = "{}"
    template['task'] = "PRO"
    template['source_argc'] = 1
    template['source_argv'] = ['query_trigger']
    template['target_argc'] = 1
    template['target_argv'] = ['item']
    template['time_emb'] = False
    template['id'] = "13-6"

    task_subgroup_3["13-6"] = template

    return task_subgroup_3


task_subgroup = geticbutemplate()
