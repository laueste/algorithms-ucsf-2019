import sys
import argparse
from .io import read_sequence_fasta,read_pairs_file,read_scoring_matrix,write_alignment,write_matrix
from .utils import Sequence
from .align import align,print_alignment,rescore_alignment
from .truefalse import align_pairs,scan_gap_scoring,modify_scoring_matrix
from .roc import compute_roc,graph_recorded_rocs,graph_normalized_roc_comparison
from .optimization import create_matrix_library,optimize

# USAGE:
# python -m alignment <query seq file> <target seq file> <scoring matrix file> [-T <True/False>]

## Parse User Input
parser = argparse.ArgumentParser(usage="python -m alignment <query seq file> <target seq file> <scoring matrix file> [-R]")
parser.add_argument("query_seq_filepath",type=str,
                        help="Fasta file of query sequence to align")
parser.add_argument("target_seq_filepath",type=str,
                        help="Fasta file of target sequence to align with")
parser.add_argument("scoring_matrix_filepath",default=None,type=str,
help="File for scoring matrix to use, whitespace-delimited, # read as comments")
parser.add_argument("-R",default=False,action='store_true',
help="If -R is specified, the query and target will be read as raw input")
parser.add_argument("-F",default=False,action='store_true',
help='if -F is specified, the query and target are positive and negative pair\
files, and only the first one is required')

# set variables from input
args = parser.parse_args()
matrix = read_scoring_matrix(args.scoring_matrix_filepath)
if args.F == True: # batch analysis

    ## # NOTE:
    ## Due to the changing nature of the different homework questions,
    ## and my lack of ability to spend extra time working on modulization
    ## for the different homework sections, this block of code under the -F
    ## tag is essentially a script used to conduct the analyses for multiple
    ## successive homework portions. I have left in the code for each section
    ## as I used it, commented out because re-running the expensive analyses
    ## of prior sections would not be feasible. It should be read as a stepwise
    ## record of my process rather than as "single-shot" code that completes
    ## all of the homework sections at once.

    # Read in data
    pos_pair_seqs = read_pairs_file(args.query_seq_filepath)
    neg_pair_seqs = read_pairs_file(args.target_seq_filepath)

    # PART I
    #optimal penalties determined empirically with scan_gap_scoring, example below
    #scan_gap_scoring(pos_pair_seqs,neg_pair_seqs,matrix,[-4,-8],[1])
    # optimal penalties determined: -4 opening penalty, 1 gap extension score
    # (-3 and -8 opening penalties and 1 through 5 extension scores
    # were observed to have equally low FP rates for TP=0.7, so -4,1 was chosen
    # from that subset mostly arbitrarily.)

    # make ROC curves
    #new_score_matrix = modify_scoring_matrix(matrix,-4,1)
    #pos_scores = align_pairs(pos_pair_seqs,new_score_matrix)
    #neg_scores = align_pairs(neg_pair_seqs,new_score_matrix)
    #roc = compute_roc(pos_scores_nrm,neg_scores_nrm,stepsize=0.01)
    #graph_recorded_rocs()
    # matrix with lowest FP at max TP: BLOSUM50.

    # Compare best matrix score with normalized scores
    #pos_scores_nrm = align_pairs(pos_pair_seqs,new_score_matrix,normalize=True)
    #neg_scores_nrm = align_pairs(neg_pair_seqs,new_score_matrix,normalize=True)
    #roc = compute_roc(pos_scores_nrm,neg_scores_nrm,stepsize=0.01)
    #graph_normalized_roc_comparison()


    # PART II
    new_score_matrix = modify_scoring_matrix(matrix,-4,1)
    # traceback=True makes the names in the score tuples into the full
    # alignment sequences
    pos_scores = align_pairs(pos_pair_seqs[:15],new_score_matrix,traceback=True)
    neg_scores = align_pairs(neg_pair_seqs[:15],new_score_matrix,traceback=True)
    pos_alignments = [ (q,t) for s,q,t in pos_scores ]
    neg_alignments = [ (q,t) for s,q,t in neg_scores ]
    opt_matrix,changes = optimize(new_score_matrix,pos_alignments,neg_alignments,best_cutoff=5,
                            step_frac=0.2,total_iterations=3,merge_iterations=2)

    # write_matrix('optimized_matrix',opt_matrix,changes=changes,
    #                     base_matrix_name='MATIO with -4,1 gap penalties')
    # print(opt_matrix)
    # print()
    # print(changes)


else: # single analysis
    query,target = '',''
    if args.R == True:
        query = Sequence(sequence=args.query_seq_filepath,name='Query',header='raw_input')
        target = Sequence(sequence=args.target_seq_filepath,name='Target',header='raw_input')
    else:
        query = read_sequence_fasta(args.query_seq_filepath)
        target = read_sequence_fasta(args.target_seq_filepath)

    ## Align Sequences
    print("Query:",query.name,query.sequence[:min(len(query.sequence),9)]+'...')
    print("Target:",target.name,target.sequence[:min(len(target.sequence),9)]+'...')
    print("Score Matrix:",args.scoring_matrix_filepath.split('/')[-1])
    query_align,target_align,score = align(query.sequence,target.sequence,matrix)
    print('Alignment:')
    ascii = print_alignment(query_align,target_align)
    print(ascii)
    print('Alignment Score:',score)
    print(target_align)
    print(query_align)
    print('Rescore Score:',rescore_alignment(query_align,target_align,matrix))



    ## Write Results
    write_alignment("alignment_%s_%s.txt" % (query.name,target.name),
        query.name, target.name, args.scoring_matrix_filepath.split('/')[-1],
        score, ascii)
