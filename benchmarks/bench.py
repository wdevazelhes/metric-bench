import os
import time
from collections import defaultdict
import numpy as np
import pandas as pd
import yaml
import sys
from sklearn.datasets import load_iris, load_wine, fetch_mldata, \
    load_digits, fetch_olivetti_faces, load_breast_cancer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import NeighborhoodComponentsAnalysis as nca, \
    KNeighborsClassifier as knn, LargeMarginNearestNeighbor as lmnn
from sklearn.decomposition import PCA as pca
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler as scaler
sys.path.insert(0, '..')
from utils.dataset_fetcher import fetch_balance, fetch_isolet, \
    fetch_letters

DATASETS = {'iris': load_iris,
            'ionosphere': lambda: fetch_mldata('ionosphere'),
            'wine': load_wine,
            'digits': load_digits,
            'balance': fetch_balance,
            'faces': fetch_olivetti_faces,
            'breast_cancer': load_breast_cancer,
            'usps': lambda: fetch_mldata('usps'),
            'isolet': fetch_isolet,
            'letters': fetch_letters,
            }

RANDOM_SEED = 0


class GS:

    def __init__(self, n_folds=3, random_state=None):
        self.n_folds = n_folds
        self.random_state = random_state

        # variables and paths for storing results
        self.path = os.path.join('results', 'bench{}'.format(
                                    time.strftime("%Y%m%d-%H%M%S")))
        os.makedirs(self.path)
        self.grid_params = defaultdict(dict)
        self.result = defaultdict(dict)

    def set_dataset(self, dataset_name, dataset):
        self.dataset_name = dataset_name
        self.X, self.y = dataset.data, dataset.target
        self.n_features = self.X.shape[1]
        self.kfolds = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                                      random_state=self.random_state)
        self.min_n_samples = min([f[0].shape[0] for f in
                                  self.kfolds.split(self.X, self.y)])

    def compute(self, algo, grid):
        algo_name = ' + '.join(e[0] for e in algo.steps)
        t = time.time()
        print("Computing cross-validation score for {} ...".format(algo_name))
        clf = GridSearchCV(estimator=algo, param_grid=grid,
                           cv=self.kfolds, n_jobs=-1, return_train_score=False)
        clf.fit(self.X, self.y)
        cv_results = pd.DataFrame(clf.cv_results_).loc[clf.best_index_]
        self.result[dataset_name][algo_name] = cv_results[['mean_fit_time',
                'std_fit_time', 'mean_test_score', 'std_test_score']]
        self.grid_params[self.dataset_name][algo_name] = clf.best_params_
        print("Finished. Took {} s.".format(time.time() - t))

    def write_results(self):
        dict_of_df = {k: pd.DataFrame(v) for k, v in self.result.items()}
        result_df = pd.concat(dict_of_df, axis=1)
        result_df.to_csv(os.path.join(self.path, 'result.csv'))
        with open(os.path.join(self.path, 'grid_search_params'), 'w') as f:
            yaml.dump(dict(self.grid_params), f, default_flow_style=False)


def range_nfo(min_n_samples, n_features, n_points):
    return np.unique(np.linspace(2, min(min_n_samples, n_features), n_points,
                                 dtype=int)).tolist()


if __name__ == '__main__':

    simple_knn = Pipeline([('scaler', scaler()), ('knn', knn())])
    lmnn_knn = Pipeline([('scaler', scaler()), ('lmnn', lmnn()),
                         ('knn', knn())])
    nca_knn = Pipeline([('scaler', scaler()), ('nca', nca()), ('knn', knn())])
    pca_knn = Pipeline([('scaler', scaler()), ('pca', pca()), ('knn', knn())])

    cfg_file = sys.argv[1]
    with open(cfg_file, 'r') as f:
        config = yaml.load(f)
    datasets = config['datasets']

    gs = GS(n_folds=3, random_state=RANDOM_SEED)

    for dataset_name in datasets:
        print("Benchmarking dataset {}...".format(dataset_name))
        dataset_func = DATASETS[dataset_name]
        if dataset_name in ['iris', 'ionosphere', 'wine', 'digits', 'faces',
                            'breast_cancer', 'balance', 'isolet']:
            gs.set_dataset(dataset_name, dataset_func())
            gs.compute(simple_knn, [{'knn__n_neighbors': range(1, 6)}])
            gs.compute(lmnn_knn, [{'lmnn__n_features_out': [n_fo],
                                   'lmnn__n_neighbors': [n_n],
                                   'lmnn__random_state': [RANDOM_SEED],
                                   'knn__n_neighbors': [n_n]}
                                    for n_n in range(1, 6)
                                    for n_fo in [2]])
            gs.compute(nca_knn, [{'nca__n_features_out': [2],
                                  'nca__random_state': [RANDOM_SEED],
                                  'knn__n_neighbors': range(1, 6)}])
            gs.compute(pca_knn, [{'pca__n_components': [2],
                                  'pca__random_state': [RANDOM_SEED],
                                  'knn__n_neighbors': range(1, 6)}])

    gs.write_results()

    gs = GS(n_folds=3, random_state=RANDOM_SEED)

    for dataset_name in datasets:
        print("Benchmarking dataset {}...".format(dataset_name))
        dataset_func = DATASETS[dataset_name]
        if dataset_name in ['iris', 'ionosphere', 'wine', 'digits', 'faces',
                            'breast_cancer', 'balance', 'isolet']:
            gs.set_dataset(dataset_name, dataset_func())
            if dataset_name == 'isolet':
                n_features_out = [200]
            else:
                n_features_out = [min(gs.min_n_samples, gs.n_features)]
            gs.compute(simple_knn, [{'knn__n_neighbors': range(1, 6)}])
            gs.compute(lmnn_knn, [{'lmnn__n_features_out': [n_fo],
                                   'lmnn__n_neighbors': [n_n],
                                   'lmnn__random_state': [RANDOM_SEED],
                                   'knn__n_neighbors': [n_n]}
                                  for n_n in range(1, 6)
                                  for n_fo in n_features_out])
            gs.compute(nca_knn, [{'nca__n_features_out': n_features_out,
                                  'nca__random_state': [RANDOM_SEED],
                                  'knn__n_neighbors': range(1, 6)}])
            gs.compute(pca_knn, [{'pca__n_components': n_features_out,
                                  'pca__random_state': [RANDOM_SEED],
                                  'knn__n_neighbors': range(1, 6)}])
    gs.write_results()