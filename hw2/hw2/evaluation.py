# evaluation.py  functions to evaluate clusterings
import numpy as np
from .chemistry import compute_similarity

## QUALITY MEASURE

def compute_similarity_graph(cluster):
    '''Computes between-site distances for the input cluster, returns
    distances as a graph'''
    graph = {i:[] for i in range(len(cluster))}
    for i in range(len(cluster)):
        c1 = cluster[i]
        for j,c2 in enumerate(cluster[i+1:]):
            similarity_12 = compute_similarity(c1,c2)
            all_distances.append(similarity_12)
            graph[i].append((c2,similarity_12))
            graph[i+j+1].append((c1,similarity_12))
    return graph

def compute_cluster_similarity(clusters):
    '''for a list of lists of ActiveSites (a clustering), return a list of
    of the paired (mean,stdev) values for the pairwise similarities of the
    sites within a single cluster.'''
    cluster_similarities = [] #(mean,stdev) of similarity, for each cluster
    for cluster in clusters:
        similarities = []
        for i in range(len(cluster)):
            site_a = cluster[i]
            for site_b in cluster[i+1:]:
                similarities.append(compute_similarity(site_a,site_b))
        cluster_similarities.append((np.mean(similarities),np.std(similarities)))
    #print("Similarities:",cluster_similarities)
    return cluster_similarities

def quality_score(cluster_similarities,inter_cluster_similarity):
    '''very naive ratio of the average of the averages of within-cluster
    similarities compared to the between-cluster average similarity. Returns
    tuple of (mean, stdev) of the ratios; >1 means more similar within than
    between, <1 means clusters are more similar to each other than internally'''
    inter_mean, inter_dev = inter_cluster_similarity
    ratios = [ (mean/inter_mean,stdev/inter_dev) for mean,stdev in cluster_similarities ]
    print("Quality Score:",np.mean(ratios,axis=0))
    return np.mean(ratios,axis=0)

## CLUSTERING COMPARISON

def compare_clusterings(cluster_set_a,cluster_set_b):
    '''for two sets of clusters with the same k (same number of clusters),
    counts the number of sites in common '''
    cluster_mappings = [] # (clusterIndexA, clusterIndexB, n_shared sites)
    lengths = [ [len(c) for c in cluster_set_a],[len(c) for c in cluster_set_b] ]
    for i,a in enumerate(cluster_set_a):
        print("cluster A",i,len(a))
        set_a = set(a)
        for j,b in enumerate(cluster_set_b):
            print("cluster B",j,len(b))
            set_b =  set(b)
            common_sites = set_a.intersection(set_b)
            print(len(set_a.intersection(set_b)))
            cluster_mappings.append((i,j,len(common_sites)))
    res = sorted(cluster_mappings,key=lambda t:t[2],reverse=True)
    print("\nComparison of Clusterings\n")
    print("Lengths",lengths)
    print("Overlap in Clusters\n",res)
    return (lengths,res)
