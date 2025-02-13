# ACCESS

This repo includes the demo benchmark and codes for reproducing the experiments in the paper [**ACCESS: A Benchmark for Abstract Causal Event Discovery and Reasoning**](https://arxiv.org/pdf/2502.08148) to appear in NAACL 2025.

## Dataset

**ACCESS** is a benchmark for discovery and reasoning over abstract causal events in everyday life. The folder `benchmark/` provides an extraction of the ACCESS data for demonstrative purposes only.

To examine our benchmark, please refer to the following scripts
* `data_generator.py`
* `metadata.py`
* `metagraph.py`.
* `test_graph.py`.

The post-processed GLUCOSE data and other utilities are provided in this [Google folder](https://drive.google.com/drive/folders/1jUPNJycRQ2wyhs5lx4wRWRPyeaHRzWNE?usp=sharing). 
 
**For inquires to access the full benchmark, please send a request to tran.vo@monash.edu.**


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

 
