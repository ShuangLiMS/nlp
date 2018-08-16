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
        'stemming': ['lemma', 'stem'],
        'tokenizer': ['spell', 'casual'],
        'ngram_range': [(0, 1)],
        'tfidf': [True, False],
        'binary': [False, True],
        'balanced': ['Bootstrap', 'Handsample', 'False'],
        'rate': [(1, 1.0), (min_rate, max_rate), (min_rate*2, max_rate*2)],
        'classifier': ['LinearSVM', 'MultinomialNB'],
        'multiclass': ['OnevsRest'],
        'classnum': [6]
    }

param_grid = ParameterGrid(params)
timestamp = int(time.time())
timestamp = 1534378066

for _, param in enumerate(param_grid):
        with open(os.path.join(*[search_path, 'param.json']), 'w') as f:
            json.dump(param, f)
        #os.system(f"python3 hyperparameter_search.py {timestamp}")
        main(param, timestamp)