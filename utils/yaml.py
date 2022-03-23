import codecs
import yaml
import os

def _parse_from_yaml(path: str):
    '''Parse a yaml file and build config'''
    with codecs.open(path, 'r', 'utf-8') as file:
        dic = yaml.load(file, Loader=yaml.FullLoader)

    if '_base_' in dic:
        cfg_dir = os.path.dirname(path)
        base_path = dic.pop('_base_')
        base_path = os.path.join(cfg_dir, base_path)
        base_dic = _parse_from_yaml(base_path)
        dic = _update_dic(dic, base_dic)
    return dic

def _update_dic(self, dic, base_dic):
    """
    Update config from dic based base_dic
    """
    base_dic = base_dic.copy()
    dic = dic.copy()

    if dic.get('_inherited_', True) == False:
        dic.pop('_inherited_')
        return dic

    for key, val in dic.items():
        if isinstance(val, dict) and key in base_dic:
            base_dic[key] = self._update_dic(val, base_dic[key])
        else:
            base_dic[key] = val
    dic = base_dic
    return dic