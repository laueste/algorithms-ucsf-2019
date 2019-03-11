# functions and classes to assist in featurization and training


def basic_featurize_sequence(seq):
    """
    Implements super basic featurization where each sequence gets decomposed
    into an array of floats, one for each base in the sequence.

    T = 0.2
    C = 0.3
    A = 0.6
    G = 0.9

    This decomposition is just qualitatively determined; want the purines and
    pyrimidines to be close in value, want high GC content to look different
    than high AT content (it will here only when A/T and G/C equally likely...)

    Input: DNA sequence as a string
    Output: Dictionary of features as floating point numbers
    """

    return

def featurize_sequence(seq):
    """Implement a more interesting/sophisticated featurization?"""
    return


def stratify_training_data(train_data,n_classes,k_folds):
    """
    Implement creating the folds for K-fold stratified cross-validation

    Inputs: the data (as a ???), [n_classes as an int], k_folds to create as an int
    Outputs: an array of the input data set separated into K stratified folds
    """
    return


def scan_seq_sizes(window_size=8,start=0,stop=-1):
    """
    Builds a fast and not terribly accurate net and tests performance when fed
    in only a subset (window_size bases) of the original 17-bp sequence,
    and slides that window along the full sequence length to determine which
    window(s) contain the most pertinent information
    Ex. for window size 3:
    [A B C] D E F ... -> A [B C D] E F... -> A B [C D E] F... etc

    Inputs: window_size  size of the subsequence to test
            start  the position in the sequence to start the first window
            stop  the position in the sequence to end the final window
    """
    return
