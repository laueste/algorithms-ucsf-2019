import matplotlib
matplotlib.use('TkAgg') #to fix a weird bug that causes an NSException anytime plt.____ is called
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .utils import Atom, Residue, ActiveSite, Cluster
from .chemistry import compute_coordinates,compute_dimensions,compute_sav,compute_aa_proportions

def plot_similarity(similarity_values): #used previously, currently not called
    '''makes a histogram of the input list of similarity values
    to reveal the general data distribution'''
    fig,ax = plt.subplots()
    ax.hist(similarity_values,bins=50)
    ax.plot()
    ax.set_title("Distribution of Similarity Values")
    plt.savefig("similarity_histogram.png",bbox_inches="tight")

def simple_plot_sites(active_sites):
    x = [ s.dim_coordinates if s.has_coordinates else compute_coordinates(s)[0] for s in active_sites ]
    #y = [ s.chem_coordinates[0] for s in active_sites ] #since just computed above
    y = [ len(s.residues) for s in active_sites ]
    fig,ax = plt.subplots()
    ax.scatter(x,y)
    plt.savefig("SAV vs Length")

def plot_pca(active_sites):
    '''Makes a PCA-reduced scatterplot using all the data we considered in the
    clustering algorithms, unweighted. Note that this space is NOT exactly the
    same space as the clusters were created on, so the mapping of cluster colors
    to points in this space is not exactly the same as plotting the clusters
    in their own space directly, (and in fact doing clustering post-pca would
    have actually been my preferred method to do this, oops!), but there is not
    much new information going into the PCA, so this seems like a reasonable
    approximation.'''
    cols = ["name","length","x_dim","y_dim","z_dim","SAV","nonpolar","cysteine",
    "polar_unch","positive","negative","bent"]
    data = { c:[] for c in cols }
    for s in active_sites:
        data['name'].append(s.name)
        data['length'].append(len(s.residues))
        x,y,z = compute_dimensions(s)
        data['x_dim'].append(x)
        data['y_dim'].append(y)
        data['z_dim'].append(z)
        data['SAV'].append(compute_sav((x,y,z)))
        chem = compute_aa_proportions(s)
        data['nonpolar'].append(chem[0])
        data['cysteine'].append(chem[1])
        data['polar_unch'].append(chem[2])
        data['positive'].append(chem[3])
        data['negative'].append(chem[4])
        data['bent'].append(chem[5])
    df = pd.DataFrame.from_dict(data)
    # VISUALIZATION CODE FROM:
    # https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
    x = df.loc[:,cols[1:]].values #everything but the site name
    #standardize
    x = StandardScaler().fit_transform(x)
    #pca
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(x)
    pca_df = pd.concat([pd.DataFrame(data=pcs,columns=['pc1','pc2']),df[['name']]],axis=1)
    fig,ax = plt.subplots()
    pca_df.plot(kind='scatter',x='pc1',y='pc2')
    plt.savefig("PCA Decomposition.png")
    return pca_df

def plot_pca_clusters(active_sites,clustering,type='Hierarchical'):
    pca_df = plot_pca(active_sites)
    pca_df = pca_df.sort_values(by='name')
    site_clusters = { s.name:0 for s in active_sites }
    for j in range(len(clustering)):
        for k in range(len(clustering[j])):
            site_clusters[clustering[j][k].name] = j
    pca_df['cluster'] = pca_df['name'].apply(lambda n: site_clusters[n])
    pca_df.plot.scatter(x='pc1',y='pc2',c='cluster',colormap='viridis',
    title='PCA, with %s Clusters' % type, figsize=(9,6))
    plt.savefig("PCA with %s Clusters.png" % type)
