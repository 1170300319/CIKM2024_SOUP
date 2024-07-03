# =====================================================
# Task Subgroup 11 -- Query Item Relevant(Q2T) -- 3 Prompts
# =====================================================

def geticbutemplate():
    task_subgroup_11 = {}

    template = {}

    '''
    Input template:
    Based on the product information provided, produce a relevant score for the query.
    Note that 1 stands for fair and bad, 2 stands for well and good, 3 stands for excellent.
    Current query: {{query_trigger}}
    Product ID: {ID}
    Product title: {title} 
    
    Target template:
    {{item [item_id]}}
    
    
    Metrics:
    未确定
    '''

    template[
        'source'] = "Based on the product information provided, produce a relevant score for the query. \nNote that 1 stands for fair and bad, 2 stands for well and good, 3 stands for excellent. \nCurrent query: {} \nProduct ID: {} \nProduct title: {} "
    template['target'] = "{}"
    template['task'] = "Q2T"
    template['source_argc'] = 3
    template['source_argv'] = ['query trigger', 'ID', 'title', ]
    template['target_argc'] = 1
    template['target_argv'] = ['item_id']
    template['id'] = "11-1"

    task_subgroup_11["11-1"] = template

    template = {}

    '''
    Input template:
    Based on the product information provided, produce a relevant score for the query.
    Note that 1 stands for fair and bad, 2 stands for well and good, 3 stands for excellent.
    Current query: {{query_trigger}}
    Product ID: {ID}
    Product title: {title} 
    Product Category: {title} 

    Target template:
    {{item [item_id]}}


    Metrics:
    未确定
    '''

    template[
        'source'] = "Based on the product information provided, produce a relevant score for the query. \nNote that 1 stands for fair and bad, 2 stands for well and good, 3 stands for excellent. \nCurrent query: {} \nProduct ID: {} \nProduct title: {} \nProduct Category: {} "
    template['target'] = "{}"
    template['task'] = "Q2T"
    template['source_argc'] = 4
    template['source_argv'] = ['query trigger', 'ID', 'title', 'category']
    template['target_argc'] = 1
    template['target_argv'] = ['item_id']
    template['id'] = "11-2"

    task_subgroup_11["11-2"] = template

    template = {}

    '''
    Input template:
    Based on the product information provided, produce a relevant score for the query.
    Then predict the category of the product.
    Note that 1 stands for fair and bad, 2 stands for well and good, 3 stands for excellent.
    Current query: {{query_trigger}}
    Product ID: {ID}
    Product title: {title} 

    Target template:
    {{item [item_id]}}


    Metrics:
    未确定
    '''

    template[
        'source'] = "Based on the product information provided, produce a relevant score for the query. \nThen predict the category of the product. \nNote that 1 stands for fair and bad, 2 stands for well and good, 3 stands for excellent. \nCurrent query: {} \nProduct ID: {} \nProduct title: {}"
    template['target'] = "{}, {}"
    template['task'] = "Q2T"
    template['source_argc'] = 3
    template['source_argv'] = ['query trigger', 'ID', 'title']
    template['target_argc'] = 2
    template['target_argv'] = ['scores', 'category']
    template['id'] = "11-3"

    task_subgroup_11["11-3"] = template

    template = {}

    '''
    Input template:
    Based on the product ID, title and related query provided, predict the category of the product.
    Current query: {{query_trigger}}
    Product ID: {ID}
    Product title: {title} 

    Target template:
    {{item [item_id]}}


    Metrics:
    未确定
    '''

    template[
        'source'] = "Based on the product information provided, predict the category of the product. \nCurrent query: {} \nProduct ID: {} \nProduct title: {}"
    template['target'] = "{}"
    template['task'] = "Q2T"
    template['source_argc'] = 3
    template['source_argv'] = ['query trigger', 'ID', 'title']
    template['target_argc'] = 1
    template['target_argv'] = ['category']
    template['id'] = "11-4"

    task_subgroup_11["11-4"] = template

    return task_subgroup_11


task_subgroup = geticbutemplate()