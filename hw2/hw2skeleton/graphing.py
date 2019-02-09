import matplotlib
matplotlib.use('TkAgg') #to fix a weird bug that causes an NSException anytime plt.____ is called
import matplotlib.pyplot as plt
import numpy as np
from .utils import Atom, Residue, ActiveSite, Cluster

def plot_similarity(similarity_values): #used previously, currently not called
    '''makes a histogram of the input list of similarity values
    to reveal the general data distribution'''
    fig,ax = plt.subplots()
    ax.hist(similarity_values,bins=50)
    ax.plot()
    ax.set_title("Distribution of Similarity Values")
    plt.savefig("similarity_histogram.png",bbox_inches="tight")

def plot_sites():
    # TODO:
    return

def plot_clusters():
    ## TODO:
    return
