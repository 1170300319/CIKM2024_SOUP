# =====================================================
# Task Subgroup 4 -- Query Auto-Complete(QAC) -- 3 Prompts
# =====================================================


def geticbutemplate():

    task_subgroup_4 = {}

    template = {}

    '''
    Input template:
    Given the purchase history of user {{user_id}}：
    {{list of item_id}}
    and current partial input {{prefix}}, complete the input query to reflect user's purchase interest.
    
    Target template:
    {{item [item_id]}}
    
    
    Metrics:
    未确定
    '''

    template[
        'source'] = "Given the following purchase history of user_{} : \n {} \n and current partial input {}, complete the input query to reflect user's purchase interest."
    template['target'] = "{}"
    template['task'] = "QAC"
    template['source_argc'] = 3
    template['source_argv'] = ['user_id', 'purchase_history', 'prefix']
    template['target_argc'] = 1
    template['target_argv'] = ['item_id']
    template['id'] = "4-1"

    task_subgroup_4["4-1"] = template

    template = {}
    '''
    Input template:
    I find the purchase history list of user {{user_id}}:
    {{history item list of {{item_id}}}}
    and current partial input {{prefix}}. 
    I wonder which is the next item the user wants. Can you complete the input query?
    
    
    Target template:
    {{item [item_id]}}
    
    
    Metrics:
    未确定
    '''
    template[
        'source'] = "I find the purchase history list of user_{} : \n {} \n and current partial input {}. I wonder which is the next item the user wants. Can you complete the input query?"
    template['target'] = "{}"
    template['task'] = "QAC"
    template['source_argc'] = 3
    template['source_argv'] = ['user_id', 'purchase_history', 'prefix']
    template['target_argc'] = 1
    template['target_argv'] = ['item_id']
    template['id'] = "4-2"

    task_subgroup_4["4-2"] = template

    template = {}
    '''
    Input template:
    Here is the purchase history list of user {{user_id}}:
    {{history item list of {{item_id}}}}
    and current partial input {{prefix}}. 
    try to complete the input query for the user.
    
    Target template:
    {{item [item_id]}}
    
    
    Metrics:
    '''
    template['source'] = "Here is the purchase history list of user_{} : \n {} \n and current partial input {}. Try to complete the input query for the user."
    template['target'] = "{}"
    template['task'] = "QAC"
    template['source_argc'] = 3
    template['source_argv'] = ['user_id', 'purchase_history', 'prefix']
    template['target_argc'] = 1
    template['target_argv'] = ['item_id']
    template['id'] = "4-3"

    task_subgroup_4["4-3"] = template

    template = {}

    '''
    Input template:
    Given the purchase interest category of user {{user_id}}：
    {{list of item_id}}, 
    user's nationality: {county} and current partial input {{prefix}}, complete the input query to reflect user's purchase interest.
    
    Target template:
    {{item [item_id]}}
    
    
    Metrics:
    未确定
    '''

    template[
        'source'] = "Given the purchase interest category of user_{} : \n {}, \n user's nationality: {} and current partial input {}, \n complete the input query to reflect user's purchase interest."
    template['target'] = "{}"
    template['task'] = "QAC"
    template['source_argc'] = 4
    template['source_argv'] = ['user_id', 'interest_category', 'county', 'prefix']
    template['target_argc'] = 1
    template['target_argv'] = ['item_id']
    template['id'] = "4-4"

    task_subgroup_4["4-4"] = template

    template = {}
    '''
    Input template:
    I find the purchase interest category list of user {{user_id}}:
    {{history item list of {{item_id}}}}, 
    user's nationality: {county} and current partial input {{prefix}}. 
    I wonder which is the next item the user wants. Can you complete the input query?
    
    
    Target template:
    {{item [item_id]}}
    
    
    Metrics:
    未确定
    '''
    template[
        'source'] = "I find the purchase interest category list of user_{} : \n {}, \n user's nationality: {} and current partial input {}. \n I wonder which is the next item the user wants. Can you complete the input query?"
    template['target'] = "{}"
    template['task'] = "QAC"
    template['source_argc'] = 4
    template['source_argv'] = ['user_id', 'interest_category', 'county', 'prefix']
    template['target_argc'] = 1
    template['target_argv'] = ['item_id']
    template['id'] = "4-5"

    task_subgroup_4["4-5"] = template

    template = {}
    '''
    Input template:
    Here is the purchase interest category list of user {{user_id}}:
    {{history item list of {{item_id}}}}, 
    user's nationality: {county} and current partial input {{prefix}}. 
    try to complete the input query for the user.
    
    Target template:
    {{item [item_id]}}
    
    
    Metrics:
    '''
    template['source'] = "Here is the purchase interest category list of user_{} : \n {}, \n user's nationality: {} and current partial input {}. \n Try to complete the input query for the user."
    template['target'] = "{}"
    template['task'] = "QAC"
    template['source_argc'] = 4
    template['source_argv'] = ['user_id', 'interest_category', 'county', 'prefix']
    template['target_argc'] = 1
    template['target_argv'] = ['item_id']
    template['id'] = "4-6"

    task_subgroup_4["4-6"] = template

    template = {}

    '''
    Input template:
    Complete the input query to reflect user's purchase interest.
    Current query: {prefix}
    User ID: {ID}
    Nationality: {county} 
    Query Category preferences: {interest_category} 
    
    Target template:
    {{item [item_id]}}
    
    
    Metrics:
    未确定
    '''

    template[
        'source'] = "Complete the input query to reflect user's purchase interest. \nCurrent query: {} \nUser ID: {} \nNationality: {} \nQuery category preferences: {} \n"
    template['target'] = "{}"
    template['task'] = "QAC"
    template['source_argc'] = 4
    template['source_argv'] = ['prefix', 'user_id', 'county', 'interest_category']
    template['target_argc'] = 1
    template['target_argv'] = ['item_id']
    template['id'] = "4-7"

    task_subgroup_4["4-7"] = template

    template = {}
    '''
    Input template:
    I wonder which is the next item the user wants. Can you use the user information to complete the input query?
    Current query: {prefix}
    User ID: {ID}
    Nationality: {county} 
    Query category preferences: {interest_category} 
    
    Target template:
    {{item [item_id]}}
    
    
    Metrics:
    未确定
    '''
    template[
        'source'] = "I wonder which is the next item the user wants. Can you use the user information to complete the input query? \nCurrent query: {} \nUser ID: {} \nNationality: {} \nQuery category preferences: {}"
    template['target'] = "{}"
    template['task'] = "QAC"
    template['source_argc'] = 4
    template['source_argv'] = ['prefix', 'user_id', 'county', 'interest_category']
    template['target_argc'] = 1
    template['target_argv'] = ['item_id']
    template['id'] = "4-8"

    task_subgroup_4["4-8"] = template

    template = {}
    '''
    Input template:
    Try to complete the input query for the user based on the user information.
    Current query: {prefix}
    User ID: {ID}
    Nationality: {county} 
    Query category preferences: {interest_category} 
    
    Target template:
    {{item [item_id]}}
    
    
    Metrics:
    '''
    template['source'] = "Try to complete the input query for the user based on the user information. \nCurrent query: {} \nUser ID: {} \nNationality: {} \nQuery category preferences: {} "
    template['target'] = "{}"
    template['task'] = "QAC"
    template['source_argc'] = 4
    template['source_argv'] = ['prefix', 'user_id', 'county', 'interest_category']
    template['target_argc'] = 1
    template['target_argv'] = ['item_id']
    template['id'] = "4-9"

    task_subgroup_4["4-9"] = template

    template = {}
    '''
    Input template:
    Complete the input query to reflect user's purchase interest.
    Current query: {prefix}
    User ID: {ID}
    
    Target template:
    {{item [item_id]}}
    
    
    Metrics:
    未确定
    '''

    template[
        'source'] = "Complete the input query to reflect user's purchase interest. \nCurrent query: {} \nUser ID: {} \n"
    template['target'] = "{}"
    template['task'] = "QAC"
    template['source_argc'] = 2
    template['source_argv'] = ['prefix', 'user_id']
    template['target_argc'] = 1
    template['target_argv'] = ['item_id']
    template['id'] = "4-10"

    task_subgroup_4["4-10"] = template

    template = {}
    '''
    Input template:
    Complete the input query to reflect user's purchase interest.
    Current query: {prefix}
    User ID: {ID}
    Nationality: {county} 
    Query category preferences: {interest_category} 

    Target template:
    {{item [item_id]}}


    Metrics:
    未确定
    '''

    template[
        'source'] = "Given the fact, try to complete the input query to reflect user's purchase interest. \nCurrent query: {} \nUser ID: {} \nNationality: {} \nQuery category preferences: {}"
    template['target'] = "{}"
    template['task'] = "QAC"
    template['source_argc'] = 4
    template['source_argv'] = ['prefix', 'user_id', 'county', 'interest_category']
    template['target_argc'] = 1
    template['target_argv'] = ['item_id']
    template['id'] = "4-11"

    task_subgroup_4["4-11"] = template

    return task_subgroup_4


task_subgroup = geticbutemplate()

