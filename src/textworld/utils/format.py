

def format_settings(setting:dict)->str:
    """
    Takes a dictionary describing the experiment setting and returns an experiment id
    """
    experiment_id = ''
    for key,value in setting.items():
        experiment_id += f'{key}:{value}'
        experiment_id += '_'
    return experiment_id

