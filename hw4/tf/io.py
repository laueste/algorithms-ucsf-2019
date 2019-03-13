# functions to read in transcription factor sequence data from given files,
# also includes the filtering functions for the negative data, despite that
# not exactly being "io"
import numpy as np
import os

def read_seqs(filepath):
    """
    Read in sequences from a text file of newline-delimited sequences
    Input: file path as string
    Output: 1D array of sequences as strings, all uppercase
    """
    seqs = []
    with open(filepath,"r") as f:
        for line in f:
            seqs.append(line.strip().upper())
    return seqs

def write_seqs(filepath,seqs):
    """
    Write a list of sequences to a file.

    Input: a filename, an array-like of sequence strings to write
    Output: none
    """
    out = open(filepath,'w')
    for s in seqs:
        out.write(s+"\n")
    out.close()

def parse_negatives_fasta(filepath):
    """
    Read in fasta file of yeast upstream regions, break into 17-bp chunks

    Input: FASTA file path
    Output: List of 17-bp sequences
    """
    basename = os.path.basename(filepath)
    name = os.path.splitext(basename)

    if name[1] not in [".fa",".fasta"]: #there are two possible FASTA suffixes
        raise IOError("%s is not a FASTA file"%filepath)
    seqs = []
    buffer = ''
    # open the file
    with open(filepath, "r") as f:
        for line in f:
            if line[0] != '>': #first line case
                buffer += line.strip().upper() #join the 60-char lines
            if len(buffer) == 1140: #every 15 lines, splits evenly into 17bp
                seqs += [ buffer[i:i+17] for i in range(0,len(buffer),17)]
                buffer = ''
    buffer = buffer[:len(buffer)-len(buffer)%17]
    seqs += [ buffer[i:i+17] for i in range(0,len(buffer),17)]
    return seqs



# TODO: how do we need to handle the reverse complements of all these inputs?
# Do we WANT to catch rev comps of hits? Start with NOT doing so...



def filter_pos_hits(pos_hits_list,list_to_be_filtered):
    """Return all elements of the second list that do not appear in the first"""
    pos_set = set(pos_hits_list)
    input_set = set(list_to_be_filtered)
    difference = input_set.difference(pos_set)
    return np.array(list(difference))
    # filtered_hits = np.zeros(list_to_be_filtered) #initialize at max length to save time
    # counter = 0 #how long is the filtered list actually
    # for i in range(len(list_to_be_filtered)):
    #     s = list_to_be_filtered[i]
    #     if s not in pos_hits_list:
    #         filtered_hits[i] = s
    #         counter += 1
    # return filtered_hits[:counter].copy() #return array reduced to actual size of contents


def make_negatives_file(fasta_filepath,pos_filepath):
    """Write a file of negative 17-bp sequences using the input fasta array"""
    raw_negative_seqs = parse_negatives_fasta(fasta_filepath)
    pos_hits = read_seqs(pos_filepath)
    filtered = filter_pos_hits(pos_hits,raw_negative_seqs)
    write_seqs("./data/rap1-lieb-constructed-negatives.txt",filtered)
