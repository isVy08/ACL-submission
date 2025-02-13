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

To do automatic clustering, you first need to create the data directory `mkdir data`, then download to `data/` the related objects from the above folder. 
Then run the following command lines: 

```
python run_clustering.py pivot
python tune_clustering.py pivot
```

Other algorithms for ablation are also provided, e.g., `louvain`, `leiden`, `optics`.

### 2. Statistical Structure Learning

You need to first install `gcastle` and `causal-learn` libraries to run the statistical causal discovery algorithms. Then, specify the graph size (e.g. $25$ nodes), method (e.g. ``NOTEARS``) and run

```
python run_cd_alg.py 25 NOTEARS
```

### 3. GLUCOSE QA with LLMs

ACCESS can be used to evaluate LLMs on the following tasks:  

a) mapping ACCESS abstractions to mentions in GLUCOSE: refer to `map_abstraction.py`

b) causal discovery: use ACCESS to `generate_causal_pairs.py`.

c) causal QA reasoning: refer to `generate_causal_qa.py` for how to construct the GLUCOSE QA dataset. 
Please refer to `benchmark/` for an example. Note that the data released in our benchmark has been subjected to human validation for true causality. 
 
