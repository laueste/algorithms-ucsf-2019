import matplotlib
matplotlib.use('TkAgg') #to fix a weird bug that causes an NSException anytime plt.____ is called
import matplotlib.pyplot as plt
import pandas as pd
from .utils import Atom, Residue, ActiveSite, Cluster
from .chemistry import compute_aa_similarity, compute_dim_similarity

def compute_similarity(site_a, site_b):
    """
    Compute the similarity between two given ActiveSite instances.

    Input: two ActiveSite instances
    Output: the similarity between them (a floating point number)
    """
    ## Naively average the similarity values of the dimensions
    ## and the chemical makeups of the sites to achieve a global
    ## similarity value.
    # NOTE: for this data set, dim similarity varies from 0 to 0.89
    # NOTE: for this data set, chem similarity varies from 0.14 to 1.0
    # (see histograms of similarity value distributions)

    chem_similarity = compute_aa_similarity(site_a,site_b)
    dim_similarity = compute_dim_similarity(site_a,site_b)
    return (chem_similarity + dim_similarity) * 0.5

def plot_similarity(s):
    '''makes a histogram of the input list of similarity values
    to reveal the general data distribution'''
    fig,ax = plt.subplots()
    ax.hist(s,bins=50)
    ax.plot()
    ax.set_title("Distribution of Similarity Values")
    plt.savefig("similarity_histogram.png",bbox_inches="tight")
    return

def cluster_by_partitioning(active_sites):
    """
    Cluster a given set of ActiveSite instances using a partitioning method.

    Input: a list of ActiveSite instances
    Output: a clustering of ActiveSite instances
            (this is really a list of clusters, each of which is list of
            ActiveSite instances)
    """
    # Fill in your code here!
    graph = {s.name:[] for s in active_sites}
    for i in range(len(active_sites)):
        for j in range(len(active_sites[i+1:])):
            site_a = active_sites[i]
            site_b = active_sites[j]
            similarity_ab = compute_similarity(site_a,site_b)
            graph[site_a.name].append((site_b.name,similarity_ab))
            graph[site_b.name].append((site_a.name,similarity_ab))

    ## CURE clustering method, or K-means
    return [[1],[2],[3]]

def graph_sites(active_sites):
    print(active_sites)
    graph = {s.name:[] for s in active_sites}
    for i in range(len(active_sites)):
        site_a = active_sites[i]
        for site_b in active_sites[i+1:]:
            similarity_ab = compute_similarity(site_a,site_b)
            graph[site_a.name].append((site_b.name,similarity_ab))
            graph[site_b.name].append((site_a.name,similarity_ab))
    for k,v in graph.items():
        print(k,v)
        print(min(v, key=lambda e: e[1]))
    return graph


def compute_cluster_distance(cluster_x,cluster_y):

    ## WHAT IS THE BEST WAY TO REPRESENT CLUSTERS HERE?
    # should they be 1D lists always, and record the
    # dendrogram seperately? should clusters have
    # two components, a structure and a simple flattened list of sites?
    # ><

def compute_centroid:


def cluster_hierarchically(active_sites):
    """
    Cluster the given set of ActiveSite instances using a hierarchical algorithm
    In this case, we use agglomerative clustering (to save computing costs in
    the first step), and ________ linkages to join nearest clusters.

    Input: a list of ActiveSite instances
    Output: a list of clusterings
            (each clustering is a list of lists of ActiveSite objects)
    """
    dendrogram = [ Cluster([site]) for site in active_sites ]
    graph = {c.name:[] for c in dendrogram}
    for c in dendrogram:




    # try closest, furthest, mean and see what the clusters look like?

    return []
