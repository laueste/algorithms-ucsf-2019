import numpy as np
import pandas as pd
from .utils import Atom, Residue, ActiveSite, Cluster
from .chemistry import compute_aa_similarity, compute_dim_similarity, compute_coordinates
from .centroids import make_centroid_site

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

## PARTITIONING CLUSTERING

##          DISTANCE MEASURE
def compute_site_distance(site_a,site_b):
    '''Calculate similarity between ActiveSite objects and convert to a distance
    via the function distance = 1 / (1-similarity)'''
    similarity = compute_similarity(site_a,site_b)
    distance = 1 / (1 - similarity)   #alternate distance is 1 - similarity
    # use nonlinear scaling because according to the similarity values histogram
    # most of the similarity values are close together (~normal dist at ~0.5 )
    # The linear alternative would be to use distance = 1 - similarity
    return distance

##          ASSIGNMENT
def assign_to_clusters(centroids,sites):
    '''
    Given an input list of centroids (as ActiveSite objects)
    and an input list of sites to sort, returns a dictionary
    of the form  { centroid1: [site_a, site_b...], centroid2: [site_c, ...]}
    '''
    clusters = { c:[] for c in centroids }
    for s in sites:
        min_pair = min([ (c,compute_site_distance(c,s)) for c in centroids ],key=lambda p:p[1])
        closest_centroid = min_pair[0]
        clusters[closest_centroid].append(s)
    return clusters

##          CENTROID RECALCULATION
def recompute_centroids(cluster_dict):
    '''Given an input cluster dictionary of the form { centroid1:[site,site...],
    centroid2:[site,site...],...}, recompute the centroids and return a tuple
    of the list of centroids and the list of their corresponding clusters'''
    centroids = []
    clusters = []
    for centroid,cluster in cluster_dict.items():
        if len(cluster) == 0:
            centroids.append(centroid) #if orphan centroid, leave it alone
        else:
            # below strategy is hacky, but i made the Cluster class for the
            # hierarchical first, ran out of time to refactor well...
            centroids.append(make_centroid_site(Cluster(cluster)))
        clusters.append(cluster)
    return (centroids,clusters)

def test_convergence(old_centroids,new_centroids,threshold_distance,prev_distance):
    '''Test the pairwise distances between the old and new centroids, and if
    the distance is less than the input threshold, or if the exact distance is
    the same as it was in the previous iteration, end EM iterations
    '''
    zipped = zip(sorted(old_centroids,key=lambda c:c.name),sorted(new_centroids,key=lambda c:c.name))
    pairwise_distances = [compute_site_distance(old,new) for old,new in zipped]
    new_dist = min(pairwise_distances)
    print(new_dist,threshold_distance,prev_distance)
    if new_dist < threshold_distance or new_dist == prev_distance:
        return (True,new_dist)
    else:
        return (False,new_dist)


##          CLUSTERING MAIN ALGROITHM
def cluster_by_partitioning(active_sites,k=25,t=1.0):
    """
    Cluster a given set of ActiveSite instances using a the k-means partitioning
    method.

    Input: a list of ActiveSite instances, k clusters to partition into, and a
           convergence threshold distance t
    Output: a clustering of ActiveSite instances
            (this is really a list of clusters, each of which is list of
            ActiveSite instances)
    """
    # pick k starting centroids as specific active sites
    centroids = active_sites[:k-1] + [active_sites[-1]] #"randomly" choose first k-1 and last 1
    cluster_dict = assign_to_clusters(centroids,active_sites)
    new_centroids,cluster_lists = recompute_centroids(cluster_dict)
    convergence = test_convergence(centroids,new_centroids,
        threshold_distance=t,prev_distance=0.0)
    iterations = 0
    while convergence[0] == False and iterations < 100:
        iterations += 1
        if iterations % 100 == 0:
            print("iteration",iterations)
        centroids = new_centroids
        cluster_dict = assign_to_clusters(centroids,active_sites)
        new_centroids,cluster_lists = recompute_centroids(cluster_dict)
        convergence = test_convergence(centroids,new_centroids,
            threshold_distance=t,prev_distance=convergence[1])
        # ideally would implement a way to check for lack of change in the
    print("finished with iterations", iterations)
    return cluster_lists



## HIERARCHICAL CLUSTERING

##              DISTANCE MEASURE
def compute_cluster_distance(cluster_x,cluster_y):
    '''uses cluster centroid 'ActiveSites' to calculate the similarity
    between clusters and then convert that to cluster distances with the function
    distance = 1 / (1-similarity)'''
    #print("computing cluster distance:",cluster_x,cluster_y)
    centroid_x = make_centroid_site(cluster_x)
    centroid_y = make_centroid_site(cluster_y)
    return compute_site_distance(centroid_x,centroid_y)

##              GRAPH UTILITIES
#(aka more complex code to avoid remaking the distance graph entirely each time)
def graph_clusters(cluster_list,method):
    '''creates a dictionary of form cluster_name:[ (cluster_a_name,dist_to_a),
    (cluster_b_name,dist_to_b),...] for all the clusters in the input list.'''
    #print('making initial graph')
    dist_alg = compute_cluster_distance if method == "H" else compute_site_distance
    graph = {c:[] for c in cluster_list}
    for i in range(len(cluster_list)):
        cluster_a = cluster_list[i]
        for cluster_b in cluster_list[i+1:]:
            distance_ab = dist_alg(cluster_a,cluster_b)
            graph[cluster_a].append((cluster_b,distance_ab))
            graph[cluster_b].append((cluster_a,distance_ab))
    return graph

def remove_cluster_from_all_entries(graph,old_cluster):
    '''in-place modification to remove all instances of
    an old cluster as a destination for any graph edges'''
    for cluster,edges in graph.items():
        for edge in edges: #(cluster,dist_to_cluster) is edge structure
            if old_cluster == edge[0]:
                edges.remove(edge)


def update_graph(graph,new_cluster):
    '''updates input graph in-place to remove the old clusters that merged to
    for the input new cluster, and then add the new cluster and compute all
    distances between that new cluster and the rest of the clusters in the graph'''
    #print("called update graph with ",new_cluster)
    # new cluster always is the combination of two and only two old clusters
    for old_cluster in new_cluster.members: # remove old clusters from the graph
        graph.pop(old_cluster)
        remove_cluster_from_all_entries(graph,old_cluster)
    graph[new_cluster] = []
    #compute all distances to new cluster
    for cluster_c in graph.keys():
        if cluster_c != new_cluster:
            distance_cnew = compute_cluster_distance(new_cluster,cluster_c)
            graph[cluster_c].append((new_cluster,distance_cnew))
            graph[new_cluster].append((cluster_c,distance_cnew))
    return graph

def merge_closest_clusters(graph,t=1000):
    '''updates graph in-place to merge the two closest clusters and recalculate
    distances from the new merged cluster to the rest of the graph clusters'''
    #print("called merge closest clusters")
    # graph format is dictionary, keyed by clusters, where the value
    # is a list of tuples of form (other_cluster,distance)
    min_distances = [ (cluster, min(edges,key=lambda e:e[1])) for cluster,edges in graph.items() ]
    # distances are saved in form (clusterA,(clusterB,distance))
    min_pair = min(min_distances,key=lambda p:p[1][1])
    cluster_a = min_pair[0]
    cluster_b = min_pair[1][0]
    distance_ab = min_pair[1][1]
    if distance_ab > t: ## leave this little hook here in case we implement t...
        return "T_LIMIT_TERMINATION"
    new_cluster = Cluster([cluster_a,cluster_b])
    update_graph(graph,new_cluster)
    return new_cluster


##              LIST/CLUSTER UTILITIES
def flatten(list_clustering):
    '''transforms a nested list into a list of lists'''
    flat_list = []
    for thing in list_clustering:
        if type(thing) == ActiveSite:
            flat_list.append(thing)
        else:
            flat_list += flatten(thing)
    return flat_list

def de_cluster(cluster):
    '''transforms a cluster (nested object) into nested lists'''
    list_cluster = []
    for item in cluster.members:
        if type(item) == ActiveSite:
            list_cluster.append(item)
        elif type(item) == Cluster:
            list_cluster.append(de_cluster(item))
        else:
            raise TypeError("Cluster members must be ActiveSite or Cluster. Found %s of type %s in cluster %s" % (item,type(item),cluster) )
    return list_cluster


##              CLUSTERING MAIN ALGORITHM
def cluster_hierarchically(active_sites,k=3,t=10000):
    """
    Cluster the given set of ActiveSite instances using a hierarchical algorithm
    In this case, we use agglomerative clustering (to save computing costs in
    the first step). The linkages used to measure distances between the clusters
    are the distances between centroids computed at the middle of each cluster;
    the distance metric in this case is 1 / (1-similarity), in order to add
    greater ability to distinguish sets of close similarity values.

    Input: a list of ActiveSite instances and an integer k clusters to obtain
    Output: a list of clusterings
            (each clustering is a list of lists of ActiveSite objects)
    """
    clusters = [ Cluster([s]) for s in active_sites ]
    graph = graph_clusters(clusters)
    while len(graph) > k:
        merge_closest_clusters(graph,t=t)
    # keep this dendrogram around so that we have the option to return it someday
    dendrogram = [ de_cluster(c) for c in list(graph.keys()) ] #dendrogram up to k clusters
    clusters = [ flatten(clist) for clist in dendrogram ]
    return [clusters]
