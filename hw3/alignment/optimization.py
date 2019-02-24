# optimize a matrix dynamically for maximum TP rate at the lowest FP rates
import numpy as np
from .truefalse import calculate_true_positive_rate
from .align import rescore_alignment

aas = 'ARNDCQEGHILKMFPSTWYVBZX' #we already optimized gap vals so leave * alone
amino_acid_combinations = []
for aa1 in aas:
    for aa2 in aas:
        amino_acid_combinations.append((aa1,aa2))

def apply_matrix_changes(base_matrix,changes):
    """
    Returns a tuple that contains a copy of the base matrix with
    the input changes applied + the list of changes.
    Changes should be of the form (('AA1', 'AA2'),score_value)
    """
    new_matrix = base_matrix.copy()
    for pair,new_score in changes:
        new_matrix[pair] = new_score #make sure matrix is symmetric!
        new_matrix[pair[::-1]] = new_score
    return (new_matrix,changes)

def create_matrix_library(base_matrix,upreg=True,step_frac=0.5):
    """"""
    library = [] #store matrices as (matrix,[(change1),(change2),...])
    for pair in amino_acid_combinations:
        new_matrix = base_matrix.copy()
        old_score = new_matrix[pair]
        new_score = old_score*(1.0+step_frac) if upreg == True else old_score*step_frac
        new_matrix[pair] = new_score
        new_matrix[pair[::-1]] = new_score
        library.append((new_matrix,[(pair,new_score)]))
    return library

def make_combined_changes_library(base_matrix,list_of_changes):
    """library is list of tuples: (matrix,[(change1),(change2),...]),
    make new library with merged changes from every combination of 2 entries
    from the list of changes"""
    new_library = []
    for i in range(len(list_of_changes)):
        ch1 = list_of_changes[i]
        for j in range(i+1,len(list_of_changes)):
            ch2 = list_of_changes[j]
            new_library.append(apply_matrix_changes(base_matrix,ch1+ch2))
    return new_library

def calculate_objective_function(pos_scores,neg_scores):
    """Calculates the sum of the true positive fractions across false positive
    fractions 0.0, 0.1, 0.2, and 0.3, using the input list of positive match
    scores and negative match scores. The maximum possible score is 4.0."""
    score = 0.0
    for fpr in [0.0,0.1,0.2,0.3]:
        score += calculate_true_positive_rate(pos_scores,neg_scores,false_pos_frac=fpr)
    return score

def test_matrix_scores(score_matrix,pos_alignments,neg_alignments):
    """"""
    pos_scores = [ (rescore_alignment(p[0],p[1],score_matrix),p[0],p[1]) for p in pos_alignments ]
    neg_scores = [ (rescore_alignment(p[0],p[1],score_matrix),p[0],p[1]) for p in neg_alignments ]
    return calculate_objective_function(pos_scores,neg_scores)

def merge_iteration(base_matrix,results_list,pos_align,neg_align,best_cutoff=3):
    """"""
    # takes input as  [(score,changes),...]
    changes_list = [ changes for score,changes in results_list ]
    merged_library = make_combined_changes_library(base_matrix,changes_list)
    best_merged = np.zeros(best_cutoff,dtype=(int,2)) # (score,index)
    for i,m_entry in enumerate(merged_library):
        m,changes = m_entry
        score = test_matrix_scores(m,pos_align,neg_align)
        if score > best_merged[-1][0]:
            best_merged[-1] = [score,i]
            best_merged = best_merged[np.argsort(-best_merged[:,0])] #sort by score
    best_merged_changes = [(score,merged_library[i][1]) for score,i in best_merged]
    print("Best Merged Weights:")
    for p in best_merged_changes:
        print(p)
    return best_merged_changes # returns [(score,changes),...]

def updown_iteration(upreg_library,downreg_library,pos_align,neg_align,
                                        best_cutoff=3):
    """"""
    best_upreg = np.zeros(best_cutoff,dtype=(int,2)) # (score,index)
    best_downreg = np.zeros(best_cutoff,dtype=(int,2)) # (score,index)
    for i,m_entry in enumerate(upreg_library):
        m,changes = m_entry
        score = test_matrix_scores(m,pos_align,neg_align)
        if score > best_upreg[-1][0]:
            best_upreg[-1] = [score,i]
            best_upreg = best_upreg[np.argsort(-best_upreg[:,0])] #sort by score
    for m,ch in downreg_library:
        m,changes = m_entry
        score = test_matrix_scores(m,pos_align,neg_align)
        if score > best_downreg[-1][0]:
            best_downreg[-1] = [score,i]
            best_dowreg = best_downreg[np.argsort(-best_downreg[:,0])] #sort by score
    best_upreg_changes = [(score,upreg_library[i][1]) for score,i in best_upreg]
    best_downreg_changes = [(score,downreg_library[i][1]) for score,i in best_downreg]
    print("Best Increased Weights:")
    for p in best_upreg_changes:
        print(p)
    print("Best Decreased Weights:")
    for p in best_downreg_changes:
        print(p)

    return best_upreg_changes,best_downreg_changes


def test_optimization(score_matrix,pos_alignments,neg_alignments,best_cutoff=3,
                        step_frac=0.5,total_iterations=2,merge_iterations=2):
    score = 0
    matrix = score_matrix
    upreg_library = create_matrix_library(matrix,upreg=True)
    downreg_library = create_matrix_library(matrix,upreg=False)
    best_up,best_down = updown_iteration(upreg_library,downreg_library,pos_alignments,neg_alignments,best_cutoff=best_cutoff)
    print(best_up)
    print(best_down)
    m = merge_iteration(matrix,best_up+best_down,pos_alignments,neg_alignments,best_cutoff=best_cutoff)
    print(m)



def optimize(score_matrix,pos_alignments,neg_alignments,best_cutoff=3,
                        step_frac=0.3,total_iterations=2,merge_iterations=2):
    """"""
    score = 0
    matrix = score_matrix
    for i in range(total_iterations):
        upreg_library = create_matrix_library(matrix,upreg=True)
        downreg_library = create_matrix_library(matrix,upreg=False)

        best_up,best_down = updown_iteration(upreg_library,
                                            downreg_library,
                                            pos_alignments,
                                            neg_alignments,
                                            best_cutoff=best_cutoff)
        best_changes = best_up+best_down
        for j in range(merge_iterations):

            best_changes = merge_iteration(matrix,
                                            best_changes,
                                            pos_alignments,
                                            neg_alignments,
                                            best_cutoff=best_cutoff)
            print(j,best_changes[0])

        matrix,changes = apply_matrix_changes(matrix,best_changes[0][1])
        score = best_changes[0][0]
        print('-',i)
    return matrix,changes


## evolutionary genetic algorithm
