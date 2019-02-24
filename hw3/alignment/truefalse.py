# investigating the true/false positive rates according to part I Q1 of the hwk
# Uses the BLOSUM50 matrix by default
import numpy as np
from .align import align

def align_pairs(seq_pairs,score_matrix,normalize=False,traceback=False):
    """
    Calculates alignment scores for each pair of sequences in the input lists
    using the given score matrix
    Input: list of 2-tuples of sequence objects to align,
           scoring matrix,
           boolean option to normalize by length of the shorter seq
    Output: 2D array sorted in descending order where each row is
            [score,name_of_seq1,name_of_seq2]. If traceback is true, the names
            of the sequences are actually the alignment sequence results, not
            just the names.
    """
    pair_scores = np.zeros(len(seq_pairs),dtype=(object,3))
    for i,p in enumerate(seq_pairs):
        print(i,p)
        s1,s2 = p
        if traceback == True: #return alignment sequences instead of just names
            query_name,target_name,align_score = align(s1.sequence,s2.sequence,
                                      score_matrix, do_traceback=True )
        else:
            query_name = s1.name
            target_name = s2.name
            align_score = align( s1.sequence, s2.sequence,
                                      score_matrix, do_traceback=False )
        if normalize == True:
            align_score = align_score / min(len(s1.sequence),len(s2.sequence))
        pair_scores[i][0] = align_score
        pair_scores[i][1] = query_name
        pair_scores[i][2] = target_name
    #sort descending by 0th element of each sub-array (the score)
    pair_scores = pair_scores[np.argsort(-pair_scores[:,0])]
    return pair_scores

def calculate_false_positive_rate(pos_pair_scores,neg_pair_scores,true_pos_frac=0.7):
    """
    Input: list of 3-tuples with alignments for positive matches,
           list of 3-tuples with alignments for negative matches,
           desired fraction of true positives (aka what fraction of the
           positive matches should "pass")
    Output: false positive rate given the input true positive rate (float)
    """
    # Get cutoff score
    #  If want 70% true positives, go down the sorted list of scored pairs and
    #  find the score for the pair just below the  top 70% of scores.
    #  Ex, get the score of the 35th item from a list of 50.
    cutoff = max(0,int(round(len(pos_pair_scores) * true_pos_frac)) - 1)
    cutoff_score = pos_pair_scores[cutoff][0]  #grab just the score
    print(true_pos_frac,cutoff_score,pos_pair_scores[cutoff])
    false_positive_count = len([n for n in neg_pair_scores if n[0] >= cutoff_score])
    false_positive_rate = false_positive_count / len(neg_pair_scores)
    return false_positive_rate

def calculate_true_positive_rate(pos_pair_scores,neg_pair_scores,false_pos_frac=0.1):
    """
    Input: list of 3-tuples with alignments for positive matches,
           list of 3-tuples with alignments for negative matches,
           desired fraction of false positives (aka what fraction of the
           negative matches should "pass")
    Output: true positive rate given the input false positive rate (float)
    """
    # Get cutoff score
    #  If want 70% true positives, go down the sorted list of scored pairs and
    #  find the score for the pair just below the  top 70% of scores.
    #  Ex, get the score of the 35th item from a list of 50.
    cutoff = round(len(neg_pair_scores) * false_pos_frac)
    cutoff_score = neg_pair_scores[cutoff][0]  #grab just the score
    true_positive_count = len([n for n in pos_pair_scores if n[0] >= cutoff_score])
    true_positive_rate = true_positive_count / len(pos_pair_scores)
    return true_positive_rate

def modify_scoring_matrix(score_matrix,gap_open,gap_ext):
    """
    Makes a copy of the input scoring matrix that has the input gap opening and
    gap extension penalties instead of the native ones.
    """
    ## TODO: ideally we would find some way to compress the gap opening penalty
    ## to ('N', '*')/('*', 'N') instead of this constant-time cost to fix a
    ## large number of keys every time!
    ## TODO: implement as in-place modification to save space?
    new_matrix = {}
    for k,v in score_matrix.items():
        if k == ('*','*'):
            new_matrix[k] = gap_ext
        if '*' in k and k != ('*','*'):
            new_matrix[k] = gap_open
        else:
            new_matrix[k] = v
    return new_matrix



### PROCESS OF ANALYSIS FOR DETERMINING OPTIMAL GAP PENALTIES
# NOTE
# The starting gap opening penalty is -5
# The starting gap extension penatly is 1
# To potentially save some computing time, start searching a smaller range,
# and if monotonically better/worse in a given direction, search more in
# that direction rather than starting out by doing the entire search space
# where gap opening goes from -20 to -1 and gap extension goes from 1 to 5

## DATA SCRATCHPAD
#key: open_cost extend_cost FP_fraction
# For just the first 5 sequences:
# everything at 0.6

# up to first 10
# -5 1 0.6
# -5 5 0.6
# -20 1 0.5
# -20 5 0.5
# up to first 25
# -4 1 0.6
# -4 5 0.6
# -20 1 0.72
# -20 5 0.72
#

# with small set cutoff 140
# """-1 1 0.45
# -1 5 0.45
# -5 1 0.4
# -5 5 0.4
# -10 1 0.4
# -10 5 0.4"""

# -2 1 0.45
# -3 1 0.45
# -4 1 0.45
# -5 1 0.4
# -6 1 0.45
# -8 1 0.4

#cutoff 160
# -4 1 0.5
# -4 3 0.5
# -5 1 0.4090909090909091
# -5 3 0.4090909090909091
# -6 1 0.4090909090909091
# -6 3 0.4090909090909091
# -7 1 0.45454545454545453
# -7 3 0.45454545454545453
# -8 1 0.45454545454545453
# -8 3 0.45454545454545453
# -9 1 0.45454545454545453
# -9 3 0.45454545454545453

# cutoff 200
# -4 1 0.3333333333333333
# -5 1 0.375
# -6 1 0.375
# -7 1 0.375
# -8 1 0.375

# all 50 (no cutoff)
# -3 1 0.5
# -4 1 0.5
# -5 1 0.52

# -3 1 0.5
# -3 2 0.5
# -3 5 0.5
# -4 1 0.5
# -4 2 0.5
# -4 5 0.5

# based off of talking to some folks, who got -8, 5 as their max,
# calculated using cutoff 250aas small subset:
# -4 1 0.1875
# -4 5 0.1875
# -8 1 0.1875
# -8 5 0.1875

# alright, the extension cost literally does not matter, let's choose -4, 1

def calculate_small_subset(seqs,cutoff=100):
    """
    Makes a subset of the given list of sequence pairs for which none of
    the sequences being evaluated are longer than the cutoff length. Used for
    coarse-grained testing before running the expensive, full-set tests.
    """
    subset = []
    for s1,s2 in seqs:
        if len(s1.sequence) < cutoff and len(s2.sequence) < cutoff:
            subset.append((s1,s2))
    return subset

def scan_gap_scoring(pos_seqs,neg_seqs,score_matrix,gap_open_range,gap_ext_range,cutoff=0.7):
    """
    Input: list of tuples of sequences that are positive matches,
           list of tuples of sequences that are negative matches,
           scoring matrix to use,
           range of gap opening penalties to test
           range of gap extension scores to test
           optional cutoff for false positive rate calculation
    Output: dictionary that stores the false positive rate (for tp rate 0.7)
            for each of the gap opening and gap extension penalty combinations
            in the input ranges.
    """
    scores = []
    # For doing coarse-grained tests that take less time to run
    p_subset = calculate_small_subset(pos_seqs,cutoff=250)
    n_subset = calculate_small_subset(neg_seqs,cutoff=250)
    print(len(p_subset),len(n_subset))
    for o in gap_open_range:
        for x in gap_ext_range:
            print("calculating for gap open, gap extension:",o,x)
            new_score_matrix = modify_scoring_matrix(score_matrix,o,x)
            pos_scores = align_pairs(p_subset,new_score_matrix)
            neg_scores = align_pairs(n_subset,new_score_matrix)
            fpr = calculate_false_positive_rate(pos_scores,neg_scores,true_pos_frac=cutoff)
            scores.append({
                'gap_open_cost': o,
                'gap_ext_cost': x,
                'false_positive_rate': fpr
            })
    for d in scores:
        print(d['gap_open_cost'],d['gap_ext_cost'],d['false_positive_rate'])
    return scores
