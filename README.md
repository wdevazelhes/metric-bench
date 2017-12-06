# To regenerate the benchmark

1. First, reproduce the environment: 

```bash
conda env create -f environment.yml
```
And install my scikit-learn's fork branch containing both LMNN and NCA: 
https://github.com/wdevazelhes/scikit-learn/tree/ncalmnn
(Clone https://github.com/wdevazelhes/scikit-learn, install it in editable mode and checkout to ncalmnn branch)

2. Then, reproduce the results:

```bash
cd benchmarks
python bench.py bench_gs.cfg
```

3. Once the results are generated (it may take a while), plot the results with the help of `plot_bench.py`:
For instance, open an ipython terminal in the `benchmark` folder, and type: 
```python
cd bench20171204-XXX2
plot_bench(algos=('scaler + knn', 'scaler + lmnn + knn', 'scaler + nca + knn'), 
           min_score=0.8)
cd ../bench20171201-XXX1
plot_bench(algos=('scaler + pca + knn', 'scaler + lmnn + knn', 'scaler + nca + knn'), 
           min_score=0.2)
```
