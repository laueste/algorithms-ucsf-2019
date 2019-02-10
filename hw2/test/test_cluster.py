from hw2 import cluster
from hw2 import io
import os

def test_similarity():
    filename_a = os.path.join("data", "97612.pdb") #276
    filename_b = os.path.join("data", "47023.pdb") #4629

    activesite_a = io.read_active_site(filename_a)
    activesite_b = io.read_active_site(filename_b)

    assert cluster.compute_similarity(activesite_a, activesite_b) == 0.5421899871113518

def test_partition_clustering():
    # tractable subset
    #pdb_ids = [276, 4629, 10701]
    pdb_ids = [46495, 23812, 41729] #pick first 3

    active_sites = []
    for id in pdb_ids:
        filepath = os.path.join("data", "%i.pdb"%id)
        active_sites.append(io.read_active_site(filepath))

    result = [[active_sites[0],active_sites[1]],[active_sites[2]]]

    assert cluster.cluster_by_partitioning(active_sites,k=2)[0] == result

def test_hierarchical_clustering():
    # tractable subset
    #pdb_ids = [276, 4629, 10701]
    pdb_ids = [46495, 23812, 41729] #pick first 3

    active_sites = []
    for id in pdb_ids:
        filepath = os.path.join("data", "%i.pdb"%id)
        active_sites.append(io.read_active_site(filepath))

    result = [[active_sites[2]],[active_sites[0],active_sites[1]]]

    assert cluster.cluster_hierarchically(active_sites,k=2)[0] == result
