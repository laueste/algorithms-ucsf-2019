# some quick helper functions for clusterings
import numpy as np
import pandas as pd
from .utils import Atom, Residue, ActiveSite

## HELPER FUNCTION
def proportion(a,b):
    '''returns the difference between a and b
    divided by the larger of the two values'''
    # divide by the larger value to ensure
    # the answer is always less than 1
    return np.abs(a-b) / max(a,b)

## DIMENSIONALITY CALCULATIONS
def compute_dimensions(site):
    # take absoute value because we want distances!
    all_atoms = [ np.abs(a.coords) for r in site.residues for a in r.atoms ]
    x,y,z = list(zip(*all_atoms))
    return (max(x)-min(x),max(y)-min(y),max(z)-min(z))

def compute_sav(dim):
    x,y,z = dim
    surface_area = 2 * (x*y + x*z + y*z)
    volume = x*y*z
    return surface_area / volume

## AMINO ACID CALCULATIONS
aas = {
    "ALA":'H', #hydrophobic
    "ILE":'H',
    "LEU":'H',
    "MET":'H',
    "PHE":'H',
    "TRP":'H',
    "TYR":'H',
    "VAL":'H',
    "CYS":'C', #cysteine
    "SEC":'P', #polar uncharged
    "SER":'P',
    "THR":'P',
    "ASN":'P',
    "GLN":'P',
    "ARG":'E', #polar positive
    "HIS":'E',
    "LYS":'E',
    "ASP":'N', #polar negative
    "GLU":'N',
    "GLY":'B', #bent
    "PRO":'B'
}

def compute_aa_proportions(site):
    categories = [ aas[r.type] for r in site.residues ]
    counts = [ categories.count(c) for c in ["H","C","P","E","N","B"] ]
    return np.array(counts) / len(site.residues) #normalize


## SITE MODIFICATION
def compute_coordinates(site):
    '''modifies site in-place to record its relevant metrics that we use
    for computing site similarity. Returns coordinates also.'''
    dim = compute_dimensions(site)
    site.dim_coordinates = compute_sav(dim)
    site.chem_coordinates = compute_aa_proportions(site) #np.arr of aa %s
    site.has_coordinates = True
    return (site.dim_coordinates,site.chem_coordinates)


## SIMILARITY COMPUTATIONS

# DIMENSIONS
def compute_dim_similarity(site_a,site_b):
    '''returns a floating-point number for a 60-40
    combination of the similarities between surface-area-
    to-volume ratio and length for input sites a and b'''
    # a more sophisticated algorithm would
    # have a more interesting weighting than 60-40,
    # but I unfortunately don't have time to do a fun
    # ML/weight-learning process by friday!
    # (not using 50-50 because SAV ratio also includes
    # some information about the total size, since
    # smaller objects have higher SAV than larger
    # objects of the same proportion)

    # compute site coordinates if not present
    if site_a.has_coordinates == False:
        compute_coordinates(site_a)
    if site_b.has_coordinates == False:
        compute_coordinates(site_b)

    sav_a = site_a.dim_coordinates
    sav_b = site_b.dim_coordinates

    # rather than purely comparing volume, I want to
    # capture some measure of shape, so let's use
    # the surface-area-to-volume ratio as well, where
    # the minimum is spherelike and the max is linear
    len_a = len(site_a.residues)
    len_b = len(site_b.residues)

    return 0.6*proportion(sav_a,sav_b) + 0.4*proportion(len_a,len_b)


## CHEMICAL COMPOSITION
def compute_aa_distance(site_a,site_b):
    '''finds the euclidean distance between the two vectors
    formed by the proportions of each amino acid category in
    the site's total composition'''
    # compute site coordinates if needed
    if site_a.has_coordinates == False:
        compute_coordinates(site_a)
    if site_b.has_coordinates == False:
        compute_coordinates(site_b)

    percents_a = site_a.chem_coordinates
    percents_b = site_b.chem_coordinates
    dist = np.linalg.norm(percents_a-percents_b)
    #frobenius norm is aka euclidean norm
    return dist

def compute_aa_similarity(site_a,site_b):
    '''computes euclidean distance d and then converts to a
    similarity score (float between 0 and 1) by dividing by
    the largest possible euclidean distance between two unit
    vectors in 6-D space'''
    max_euclidean_dist = 1.4142135623731
    #computed by np.linalg.norm(np.array([1,0,0,0,0,0],np.array([0,1,0,0,0,0])))
    return 1 - (compute_aa_distance(site_a,site_b) / max_euclidean_dist)
