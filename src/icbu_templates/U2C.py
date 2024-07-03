# =====================================================
# Task Subgroup 9 -- U2C -- 2 Prompts
# =====================================================


def geticbutemplate():
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

    return task_subgroup_9


task_subgroup = geticbutemplate()
