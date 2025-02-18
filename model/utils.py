from functools import reduce
import os

def grid_search_build(params):
    keys = params.keys()
    key_pointer = [None] * len(keys)
    max = list(map(lambda k: len(params[k]), keys))
    matrix = []
    
    total_combinations = reduce(lambda a,b : a*b, max)
    
    for index,k in enumerate(keys):
        if index == 0:
            key_pointer[index] = max[index]
        elif index == 1:
            key_pointer[index] = max[0]
        else:
            key_pointer[index] = max[index-1] * key_pointer[index-1]
                
    
    for i in range(0,total_combinations):
        combination = {}
        for index, key in enumerate(keys):
            if index == 0:
                combination[key] = params[key][(i % max[index])]
            elif index == 1:
                combination[key] = params[key][(i // (key_pointer[index])) % max[index]]
            else:
                combination[key] = params[key][(i // key_pointer[index]) % max[index]]
                
        matrix.append(combination)
    return matrix

def get_env(env_var:str, default: any):
    if env_var in os.environ:
        return os.environ[env_var]
    elif default is not None:
        return default
    else:
        raise Exception(f"Variável de ambiente [{env_var}] não está definida e nenhum valor default foi informado.")