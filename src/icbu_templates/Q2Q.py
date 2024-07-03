# =====================================================
# Task Subgroup 5 -- Related Query Recommendation(Q2Q) -- 3 Prompts
# =====================================================

def geticbutemplate():
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
    Based on the facts, generate recommendations for the most related query that the user may click on.
    Current input query {{query_trigger}} 
    User ID: {ID}
    Nationality: {county} 
    Query category preferences: {interest_category} 
    
    Target template:
    {{item [item_id]}}
    
    
    Metrics:
    '''
    template['source'] = "Please provide recommendations for the most related query that the user may want to explore further. \nCurrent query {} \nUser ID: {} \nNationality: {} \nQuery category preferences: {} "
    template['target'] = "{}"
    template['task'] = "Q2Q"
    template['source_argc'] = 5
    template['source_argv'] = ['query_trigger', 'user_id', 'county', 'interest_category']
    template['target_argc'] = 1
    template['target_argv'] = ['item_id']
    template['time_emb'] = True
    template['id'] = "5-13"

    task_subgroup_5["5-13"] = template

    return task_subgroup_5


task_subgroup = geticbutemplate()