from hyperparameter_search import main
from sklearn.model_selection import ParameterGrid
import os
import json
import warnings
import time
warnings.filterwarnings("ignore")

data_path = '/Users/sli/Projects/data'
search_path = data_path + '/hyperparameter_search/mental_health_forum_simple_clf'

# Prepare data and label
cat_num = 7
max_rate = round(1/cat_num, 4)
min_rate = round(1/(100*cat_num), 4)

params = {
        'tokenizer': ['spell', 'casual'],
        'tfidf': [True, False],
        'binary': [True, False],
        'balanced': [False],
        'rate': [(1, 1.0), (min_rate, max_rate), (min_rate/2, max_rate/2)],
        'classifier': ['MultinomialNB', 'LinearSVM', 'LDA'],
        'multiclass': ['OnevsRest', 'OnevsOne']
    }


param_grid = ParameterGrid(params)
timestamp = int(time.time())

for i, param in enumerate(param_grid):
        param['para_index'] = i
        with open(os.path.join(*[search_path, 'param.json']), 'w') as f:
            json.dump(param, f)
        os.system(f"python3 hyperparameter_search.py {timestamp}")
