# hierarchical clustering helper functions
import numpy as np
import pandas as pd
from .utils import ActiveSite, Cluster
from .chemistry import compute_coordinates

## CLUSTER DISTANCE UTILITIES
def compute_centroid_coordinates(cluster):
    '''average the dims (savs), lengths, and aa group proportions for all the
    sites in the given cluster, and return the values as a 3-tuple'''
    #shortcut for most usual use case:
    #print("called compute centroid coordinates on",cluster,cluster.members)
    if (len(cluster.members) == 1) and (type(cluster.members[0]) == ActiveSite):
        s = cluster.members[0]
        dim,chem = (s.dim_coordinates,s.chem_coordinates) if s.has_coordinates else compute_coordinates(s)
        cluster.centroid_coordinates = (dim,len(s.residues),chem)
        cluster.has_coordinates = True
        return (dim,len(s.residues),chem)

    else:
        dim_values = np.zeros(len(cluster.members))
        len_values = np.zeros(len(cluster.members))
        chem_values = np.ndarray(shape=(len(cluster.members),6))

        for i,c in enumerate(cluster.members):
            if type(c) == Cluster:
                #recurse to bottom of nested clusters
                dim,length,chem = c.centroid_coordinates if c.has_coordinates else compute_centroid_coordinates(c)
                dim_values[i] = dim
                len_values[i] = length
                chem_values[i] = chem
            elif type(c) == ActiveSite:
                dim,chem = (c.dim_coordinates,c.chem_coordinates) if c.has_coordinates else compute_coordinates(c)
                dim_values[i] = c.dim_coordinates
                len_values[i] = len(c.residues)
                chem_values[i] = c.chem_coordinates
            else:
                raise TypeError("Cluster members must be ActiveSite or Cluster. Found %s of type %s in cluster %s" % (c,type(c),cluster) )

        # store centroid coordinates in cluster class variable
        centroid_dim = np.mean(dim_values)
        centroid_len = np.mean(len_values)
        centroid_chem = np.mean(chem_values,axis=0)
        cluster.centroid_coordinates = (centroid_dim,centroid_len,centroid_chem)
        cluster.has_coordinates = True
        return (centroid_dim,centroid_len,centroid_chem)


def make_centroid_site(cluster):
    '''make centroid coordinates into a dummy ActiveSite object, which has
    coordinates but no residue data'''
    #print("called make centroid site on",cluster)
    # get centroid coordinates
    dim,length,chem = cluster.centroid_coordinates if cluster.has_coordinates else compute_centroid_coordinates(cluster)

    # make ActiveSite and fill values
    centroid_site = ActiveSite(cluster.name+'_centroid')
    centroid_site.residues = np.zeros(int(round(length))) #have to round to use as a normal ActiveSite, alas...
    centroid_site.dim_coordinates = dim
    centroid_site.chem_coordinates = chem
    centroid_site.has_coordinates = True  #running this through compute_coordinates will not work!
    return centroid_site
