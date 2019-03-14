import numpy as np
import matplotlib
matplotlib.use('TkAgg') #to fix a weird bug that causes an NSException anytime plt.____ is called
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from .utils import hamming_distance
from .io import write_seqs

def screen_by_distance(pos_seqs,neg_seqs,threshold=7):
    """Write a file for the subset of negative sequences that are at most 5bp
    away (by hamming distance) from any positive sequence"""
    distances = np.zeros(len(neg_seqs))
    close_negatives = []
    for i,s in enumerate(neg_seqs):
        if i%10000 == 0:
            print(i)
        hd = 100
        for p in pos_seqs:
            hd = min(hd,hamming_distance(s,p))
        distances[i] = hd
        if hd <= threshold: #threshold chosen just by looking at the histogram
            close_negatives.append(s)
    fig,ax = plt.subplots()
    ax.hist(distances)
    ax.set_xlabel('Minimum Hamming Distance from Any Positive Seq')
    ax.set_ylabel('Negative Seq Counts')
    ax.set_title("Histogram of Hamming Distances for Negative Seqs, with Cutoff")
    plt.axvline(x=threshold,linestyle=":",color='r')
    plt.savefig('%s.png' % "HammingDistHistogram")
    print(np.mean(distances))
    write_seqs("./data/rap1-lieb-close-negatives.txt",close_negatives)

def screen_by_distance_reverse(all,close):
    """oops, forgot to make a file for the specifically far sequences..."""
    far_negatives = []
    for i,s in enumerate(all):
        if i%10000 == 0:
            print(i)
        if s not in close:
            far_negatives.append(s)
    write_seqs("./data/rap1-lieb-far-negatives.txt",far_negatives)

def make_dataset():
    return

# TODO save choices of sequences as INDEXES eventually?
def undersample(pos,close,far,neg_ratio=2,proportion_far=0.1):
    """
    Returns a dataset undersampled to the given ratio (pos:neg is 1:neg_ratio),
    with the given proportion of the negative samples taken from the far negatives

    Input: list of positive seqs, list of close negative seqs, list of far negative seqs
    Output: 2-column matrix of seq,binary_label, dictionary of description
    """
    info = {
        "type": 'undersampling',
        "neg_ratio": neg_ratio,
        "proportion_far_neg": proportion_far,
        "length": len(pos) + len(pos)*neg_ratio
    }
    total_n_neg = len(pos)*neg_ratio
    far_neg = np.random.choice(far,int(total_n_neg*proportion_far))
    close_neg = np.random.choice(close,total_n_neg-len(far_neg))
    all_pos = np.random.choice(pos,len(pos))
    pos_data = np.column_stack((all_pos,np.ones(len(all_pos))))
    neg_data = np.column_stack((close_neg+far_neg,np.zeros(len(far_neg)+len(close_neg))))
    dataset = np.concatenate((pos_data,neg_data))
    #N by 2 array of data, 1rst col is sequences, 2nd col is labels
    np.random.shuffle(dataset) #shuffle the order of the rows
    return dataset, info

def oversample(pos,close,far,neg_ratio=5,total_samples=60000):
    """
    Returns a dataset where the positive samples are oversampled up to make the
    given ratio for the given total number of samples (should be greater than
    the number of close positives, which is around 46k)

    Input: list of positive seqs, list of close negative seqs, list of far negative seqs
    Output: 2-column matrix of seq,binary_label, dictionary of description
    """
    info = {
        "type": 'oversampling',
        "neg_ratio": neg_ratio,
        "proportion_far_neg": ((total_samples/(neg_ratio+1))*neg_ratio - len(close))/total_samples,
        "length": total_samples
    }
    total_n_neg = (total_samples/(neg_ratio+1))*neg_ratio
    far_neg = np.random.choice(far,total_n_neg-len(close))
    close_neg = np.random.choice(close,total_n_neg-len(far_neg))
    all_pos = np.random.choice(pos,total_samples-total_n_neg)
    pos_data = np.column_stack((all_pos,np.ones(len(all_pos))))
    neg_data = np.column_stack((close_neg+far_neg,np.zeros(len(far_neg)+len(close_neg))))
    dataset = np.concatenate((pos_data,neg_data))
    #N by 2 array of data, 1rst col is sequences, 2nd col is labels
    np.random.shuffle(dataset) #shuffle the order of the rows
    return dataset, info

def underover(pos,close,far,neg_ratio=2,n_positives=500,proportion_far=0.1):
    """
    Returns a dataset with the given number of positives (oversampled as needed)
    and the given proprortion of pos:neg sequences (undersampled as needed),
    with the given proprortion of far negatives within the negative samples

    Input: list of positive seqs, list of close negative seqs, list of far negative seqs
    Output: 2-column matrix of seq,binary_label, dictionary of description
    """
    info = {
        "type": 'under/over-sampling',
        "neg_ratio": neg_ratio,
        "proportion_far_neg": proportion_far,
        "length": n_positives + n_positives*neg_ratio
    }
    total_n_neg = n_positives*neg_ratio
    far_neg = np.random.choice(far,int(total_n_neg*proportion_far))
    close_neg = np.random.choice(close,len(close))
    all_pos = np.random.choice(pos,n_positives)
    pos_data = np.column_stack((all_pos,np.ones(len(all_pos))))
    neg_data = np.column_stack((close_neg+far_neg,np.zeros(len(far_neg)+len(close_neg))))
    dataset = np.concatenate((pos_data,neg_data))
    #N by 2 array of data, 1rst col is sequences, 2nd col is labels
    np.random.shuffle(dataset) #shuffle the order of the rows
    return dataset, info
