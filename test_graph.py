from metagraph import ACCESS
graph = ACCESS(root='benchmark',path_to_graph='final_graph.csv', path_to_cluster='final_updated.cluster')


graph.get_subgraph_info()
# graph.remove_cycles()


pair_confounder_dict = graph.find_confounders()
print("# spurious correlations:", len(pair_confounder_dict))
cfd = 0
for pair, confounder in pair_confounder_dict.items():
    i,j = pair
    if (i,j) in graph.cause_effect_list or (j,i) in graph.cause_effect_list:
        for c in confounder: cfd += 1  

print("# confounders:", cfd)

mediators = graph.find_directed_paths(cutoff=3)
print("# mediators:", len([path for path in mediators if len(path)==3]) )

