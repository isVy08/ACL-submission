# ACCESS

This repo includes the benchmark and codes for reproducing the experiments in the paper **ACCESS: A Benchmark for Abstract Causal Event Discovery and Reasoning** to be published in NAACL 2025.

## Dataset
Our causal graph info, including the clusters and causal relations, are released in ``benchmark/``. 

The extracted GLUCOSE data and other utilities are provided in this [Google folder](https://drive.google.com/drive/folders/1jUPNJycRQ2wyhs5lx4wRWRPyeaHRzWNE?usp=sharing). 

## Experiments

### 1. Clustering for Abstract Event Extraction

To reproduce the automatic clustering results, you need to create the data directory `mkdir data`, then download to `data/` the related objects from the above folder. 
Then run the following command lines: 

```
python run_clustering.py pivot
python tune_clustering.py pivot
```
