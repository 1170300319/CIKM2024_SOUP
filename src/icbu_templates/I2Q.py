# =====================================================
# Task Subgroup 6 -- Item to Query Recommendation(I2Q) -- 3 Prompts
# =====================================================

def geticbutemplate():
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

    return task_subgroup_6


task_subgroup = geticbutemplate()