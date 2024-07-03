
# =====================================================
# Task Subgroup 7 -- U2Q with context(U2QC) -- 3 Prompts
# =====================================================

def geticbutemplate():
    task_subgroup_7 = {}

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

    template = {}
    '''
    Input template:
    Based on the following information to produce a recommendation for the user.
    The user has been recommended query {query_trigger} but did not clicked.
    User ID: {ID}
    Nationality: {county} 
    Query category preferences: {interest_category} 

    Target template:
    {{item [item]}}


    Metrics:
    未确定
    '''
    template[
        'source'] = "Based on the following information to produce a recommendation for the user. \nThe user has been recommended query {} but did not clicked. \n User ID: {} \n Nationality: {} \nQuery category preferences: {}"
    template['target'] = "{}"
    template['task'] = "U2QC"
    template['source_argc'] = 4
    template['source_argv'] = ['query_trigger', 'user_id', 'county', 'interest_category']
    template['target_argc'] = 1
    template['target_argv'] = ['item']
    template['id'] = "7-4"

    task_subgroup_7["7-4"] = template

    template = {}
    '''
    Input template:
    Given the user's information, generate a recommendation for the user.
    The user has been recommended query {query_trigger} but did not clicked.
    User ID: {ID}
    Purchase history: {{history item list of {{item}}}} 

    Target template:
    {{item [item]}}


    Metrics:
    未确定
    '''
    template[
        'source'] = "Given the user's information, generate a recommendation for the user. \nThe user has been recommended query {} but did not clicked. \n User ID: {} \nPurchase history: {}"
    template['target'] = "{}"
    template['task'] = "U2QC"
    template['source_argc'] = 3
    template['source_argv'] = ['query_trigger', 'user_id', 'purchase_history']
    template['target_argc'] = 1
    template['target_argv'] = ['item']
    template['id'] = "7-5"

    task_subgroup_7["7-5"] = template

    return task_subgroup_7


task_subgroup = geticbutemplate()