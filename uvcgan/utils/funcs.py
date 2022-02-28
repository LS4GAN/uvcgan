import copy

def recursive_update_dict(base_dict, new_dict):
    if new_dict is None:
        return

    for k,v in new_dict.items():
        if (
                isinstance(v, dict)
            and k in base_dict
            and isinstance(base_dict[k], dict)
        ):
            recursive_update_dict(base_dict[k], v)
        else:
            base_dict[k] = copy.deepcopy(v)

def join_dicts(*dicts_list):
    base_dict = {}

    for d in dicts_list:
        recursive_update_dict(base_dict, d)

    return base_dict

