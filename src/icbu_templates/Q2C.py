
# =====================================================
# Task Subgroup 8 -- Q2C -- 2 Prompts
# =====================================================

def geticbutemplate():
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

    template = {}
    '''
    Input template:
    Predict whether the given query belongs to the coresponding query category.
    Current query: {query_trigger}
    Query category: {{candidate {{item_id}}}}

    Target template:
    {{groundtruth {{item ids}}}}


    Metrics:
    未确定
    '''
    template[
        'source'] = "Predict whether the given query belongs to the coresponding query category. \nCurrent query: {} \nQuery category: {}"
    template['target'] = "{}"
    template['task'] = "Q2C"
    template['source_argc'] = 2
    template['source_argv'] = ['query', 'Query category']
    template['target_argc'] = 1
    template['target_argv'] = ['category']
    template['id'] = "8-6"

    task_subgroup_8["8-6"] = template

    return task_subgroup_8


task_subgroup = geticbutemplate()