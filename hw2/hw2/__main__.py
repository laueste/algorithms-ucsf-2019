import sys
from .io import read_active_sites, write_clustering, write_mult_clusterings
from .cluster import cluster_by_partitioning, cluster_hierarchically
from .evaluation import compare_clusterings

# USAGE:
# python -m hw2 [-H/-P] data test.txt -k <Nclusters>
#                  algorithm^  ^data dir  ^output file

# Some quick stuff to make sure the program is called correctly
if len(sys.argv) < 6:
    print("Usage: python -m hw2 [-P|-H|-C] <pdb directory> <output file> -k <k_clusters>")
    sys.exit(0)

active_sites = read_active_sites(sys.argv[2])

# Choose clustering algorithm
if sys.argv[1][0:2] == '-P':
    print("Clustering using Partitioning method")
    clustering,quality = cluster_by_partitioning(active_sites[-2:],k=int(sys.argv[5]))
    print(clustering)
    write_clustering(sys.argv[3], clustering, quality)

if sys.argv[1][0:2] == '-H':
    print("Clustering using hierarchical method")
    clustering,quality = cluster_hierarchically(active_sites[-2:],k=int(sys.argv[5]))
    print(clustering)
    write_clustering(sys.argv[3], clustering, quality)

if sys.argv[1][0:2] == '-C':
    print("Comparing Partitioning and Hierarchical methods")
    clustering_h,quality_h = cluster_hierarchically(active_sites,k=int(sys.argv[5]))
    clustering_p,quality_p = cluster_by_partitioning(active_sites,k=int(sys.argv[5]))
    comparison_results = compare_clusterings(clustering_h,clustering_p)
    write_mult_clusterings(sys.argv[3],
        [clustering_h,clustering_p],[quality_h,quality_p],comparison_results)
