# =====================================================
# Task Subgroup 10 -- traditional -- 2 Prompts
# =====================================================

def geticbutemplate():
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

    return task_subgroup_10


task_subgroup = geticbutemplate()

