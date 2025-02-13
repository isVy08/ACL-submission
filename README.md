# ACCESS

This repo includes the benchmark and codes for reproducing the experiments in the [paper](https://arxiv.org/pdf/2502.08148) **ACCESS: A Benchmark for Abstract Causal Event Discovery and Reasoning** to appear in NAACL 2025.

## Dataset


To examine our ACCESS benchmark, please refer to the following scripts
* `data_generator.py`
* `metadata.py`
* `metagraph.py`.
* `test_graph.py`.

The post-processed GLUCOSE data and other utilities are provided in this [Google folder](https://drive.google.com/drive/folders/1jUPNJycRQ2wyhs5lx4wRWRPyeaHRzWNE?usp=sharing). 


The folder `benchmark/` provides only an extraction of the ACCESS data to demonstrate how it should be used.

**For access to the full benchmark, please contact tran.vo@monash.edu.**


## Experiments

### 1. Clustering for Abstract Event Extraction

To reproduce the automatic clustering results, you need to create the data directory `mkdir data`, then download to `data/` the related objects from the above folder. 
Then run the following command lines: 

```
python run_clustering.py pivot
python tune_clustering.py pivot
```

### 2. Statistical Structure Learning

You need to first install `gcastle` and `causal-learn` libraries to run the statistical causal discovery algorithms. Then, specify the graph size (e.g. $25$ nodes), method (e.g. ``NOTEARS``) and run

```
python run_cd_alg.py 25 NOTEARS
```

 
