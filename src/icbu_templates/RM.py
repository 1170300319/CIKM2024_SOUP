# =====================================================
# Task Subgroup 12 -- Reward Model(RM) -- 3 Prompts
# =====================================================

def geticbutemplate():
    task_subgroup_12 = {}

    template = {}
    '''
    Input template:
    Please predict whether the user will be interested in the given query based on his information.
    Given query: {query} 
    Nationality: {county} 
    Query category preferences: {interest_category} 
    Purchase history: {{history item list of {{item}}}} 

    Target template:
    {{item [item]}}


    Metrics:
    未确定
    '''
    template[
        'source'] = "Please predict whether the user will be interested in the given query based on his information. \nGiven query: {} \nNationality: {} \nQuery category preferences: {} \nPurchase history: {}."
    template['target'] = "{}"
    template['task'] = "RM"
    template['source_argc'] = 4
    template['source_argv'] = ['query_trigger', 'county', 'interest_category', 'purchase_history']
    template['target_argc'] = 1
    template['target_argv'] = ['item']
    template['time_emb'] = True
    template['id'] = "12-1"

    task_subgroup_12["12-1"] = template

    template = {}
    '''
    Input template:
    Please predict whether the user will be interested in the given query based on his information.
    Given query: {query} 
    Nationality: {county} 
    Query category preferences: {interest_category} 
    Purchase history: {{history item list of {{item}}}} 

    Target template:
    {{item [item]}}


    Metrics:
    未确定
    '''
    template[
        'source'] = "Given the user's information, predict whether the user will be interested in the query. Note that 1 stands for Yes and 0 for no. \nGiven query: {} \nNationality: {} \nQuery category preferences: {} \nPurchase history: {}."
    template['target'] = "{}"
    template['task'] = "RM"
    template['source_argc'] = 4
    template['source_argv'] = ['query_trigger', 'county', 'interest_category', 'purchase_history']
    template['target_argc'] = 1
    template['target_argv'] = ['item']
    template['time_emb'] = True
    template['id'] = "12-2"

    task_subgroup_12["12-2"] = template

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
    template['task'] = "RM"
    template['source_argc'] = 1
    template['source_argv'] = ['query_trigger']
    template['target_argc'] = 1
    template['target_argv'] = ['item']
    template['time_emb'] = False
    template['id'] = "12-3"

    task_subgroup_12["12-3"] = template

    return task_subgroup_12


task_subgroup = geticbutemplate()