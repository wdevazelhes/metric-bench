import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


COLORS = {'scaler + nca + knn': 'g',
          'scaler + lmnn + knn':'c',
          'scaler + pca + knn':'y',
          'scaler + knn':'grey'}

def plot_bench(path='result.csv',
               datasets=('balance', 'digits', 'faces', 'ionosphere',
                         'isolet', 'wine'),
               algos=('scaler + pca + knn', 'scaler + knn',
                      'scaler + nca + knn', 'scaler + lmnn + knn',
                      ),
               min_score=0.2):

    f, ax = plt.subplots(2, sharex=True)
    result_df = pd.read_csv(path, header=[0, 1], index_col=[0])
    result_df = result_df.loc[:, (datasets, algos)]
    result_mean = result_df.loc['mean_test_score'].unstack()[list(algos)]
    colors = [COLORS[k] for k in result_mean.columns]
    result_mean.plot \
        .bar(yerr=result_df.loc['std_test_score'].unstack()[list(algos)],
             ax=ax[0],
             color=colors) \
        .legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[0].set_ylabel('test accuracy (%)')
    ax[0].set_ylim((min_score, 1.0))
    result_df.loc['mean_fit_time'].unstack()[list(algos)].plot \
        .bar(yerr=result_df.loc['std_fit_time'].unstack()[list(algos)],
             ax=ax[1],
             color=colors) \
        .legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1].set_yscale('log')
    ax[1].set_ylabel('fit time (s)')
    f.tight_layout()
    f.savefig('bench_fig', bbox_inches='tight')



