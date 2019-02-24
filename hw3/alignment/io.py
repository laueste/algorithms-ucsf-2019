import os
from .utils import Sequence

def read_sequence_fasta(filepath):
    """
    Read in all of the active sites from the given directory.

    Input: FASTA file path
    Output: Sequence object
    """
    basename = os.path.basename(filepath)
    name = os.path.splitext(basename)

    if name[1] not in [".fa",".fasta"]: #there are two possible FASTA suffixes
        raise IOError("%s is not a FASTA file"%filepath)

    seq = Sequence(name[0])

    # open the file
    with open(filepath, "r") as f:
        file_sequence = ''
        for line in f:
            if line[0] == '>': #first line case
                seq.header = line[1:].strip()
            else:
                file_sequence += line.strip().upper() #join the 60-char lines
    seq.sequence = file_sequence
    return seq


def read_scoring_matrix(filepath):
    """
    Parse array of line-by-line strings of a scoring matrix
    into a dictionary of tuples

    Input: m=list of lists of integers corresponding to scores (line-by-line)
           content of a scoring matrix,
           key=list of strings that are the amino acid labels for the score rows
           in the list of lists m
    Output: scoring matrix as a dictionary of tuples
           (same data structure as Biopython uses for these matrices)
    """
    # TODO: make this store the minimal set rather 23 * 23 entries
    m,key = read_matrix_data(filepath)
    scoring_matrix = {}
    for i in range(len(key)):
        aa1 = key[i]
        for j in range(len(key)):
            aa2 = key[j]
            scoring_matrix[(aa1,aa2)] = m[i][j]
    return scoring_matrix


def read_matrix_data(filepath):
    """
    Read in all of the active sites from the given directory.

    Input: scoring matrix file path
    Output: scoring matrix content as an array of ints (to be further parsed),
            and the key (the list of amino acid characters that indexes each
            line)
    """
    basename = os.path.basename(filepath)
    name = os.path.splitext(basename)

    matrix_lines  = []
    key = []
    # open the file
    with open(filepath, "r") as f:
        for line in f:
            if line[0] != '#': #if #, is comment line so ignore
                if 'A' in line: #the top header line
                    key = line.strip().split() #get a list of just the amino acid chars
                else:
                    matrix_lines.append([int(n) for n in line.strip().split()])
    return matrix_lines,key


def read_pairs_file(filepath):
    """
    Read in a text file where each line is of form  "<filename> <filename>"
    and returns a list of tuples of Sequence objects: [(Sequence,Sequence), ...]
    Needs to be run in main hw3 directory, not subfolders!

    Input: file path to text file of filepath pairs
    Output: list of tuples of Sequence objects
    """
    basename = os.path.basename(filepath)
    name = os.path.splitext(basename)

    pairs = []
    with open(filepath, "r") as f:
        for line in f:
            seqpath1,seqpath2 = line.strip().split(' ')
            pairs.append( (read_sequence_fasta(seqpath1),
                            read_sequence_fasta(seqpath2)) )
    return pairs


def write_alignment(filename,query_name,target_name,scoring_name,score,alignment_string):
    """
    Write the clustered ActiveSite instances out to a file.

    Input: a filename, a name for the query and target, an alignment ASCII
    Output: none
    """
    out = open(filename,'w')
    out.write("Query: %s \n" % query_name)
    out.write("Target: %s \n" % target_name)
    out.write("Scoring Matrix: %s \n" % scoring_name)
    out.write("Score: %s \n" % score)
    out.write("Alignment, Target on top and Query below: \n")
    out.write(alignment_string+"\n")
    out.close()

def write_matrix(filename,matrix,changes='',base_matrix_name=''):
    """Write the input scoring matrix to a file to save its contents"""
    out = open(filename,'w')
    header = 'A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  Z  X  *'
    index = header.replace(" ","")
    out.write('# Output Matrix')
    out.write('# Changes from %s are:' % base_matrix_name)
    for c in changes:
        out.write('# %s' % str(c))
    out.write(header)
    for char1 in index:
        for char2 in index:
            out.write(str(matrix[(char1,char2)])+'  ')
        out.write("\n")
    out.write("\n")
    out.close()
