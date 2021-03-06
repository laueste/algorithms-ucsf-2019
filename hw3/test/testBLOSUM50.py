# raw content from the Biopython source code at
# https://github.com/biopython/biopython/blob/master/Bio/SubsMat/MatrixInfo.py
# NOTE THIS DOES NOT CONTAIN GAP PENALTIES OR PERMUTATIONS OF AAs!
# aka, contains (A,B) but not (B,A), and does not contain (N, *)
# NOTE right now, the test checks to see that our read-in matrix of BLOSUM50
# contains all of the key/value pairs that are in this matrix, but does not
# check that our matrix is *limited* to just the ones in this file. 
test_BLOSUM50 = {
        ('W', 'F'): 1, ('L', 'R'): -3, ('S', 'P'): -1, ('V', 'T'): 0,
        ('Q', 'Q'): 7, ('N', 'A'): -1, ('Z', 'Y'): -2, ('W', 'R'): -3,
        ('Q', 'A'): -1, ('S', 'D'): 0, ('H', 'H'): 10, ('S', 'H'): -1,
        ('H', 'D'): -1, ('L', 'N'): -4, ('W', 'A'): -3, ('Y', 'M'): 0,
        ('G', 'R'): -3, ('Y', 'I'): -1, ('Y', 'E'): -2, ('B', 'Y'): -3,
        ('Y', 'A'): -2, ('V', 'D'): -4, ('B', 'S'): 0, ('Y', 'Y'): 8,
        ('G', 'N'): 0, ('E', 'C'): -3, ('Y', 'Q'): -1, ('Z', 'Z'): 5,
        ('V', 'A'): 0, ('C', 'C'): 13, ('M', 'R'): -2, ('V', 'E'): -3,
        ('T', 'N'): 0, ('P', 'P'): 10, ('V', 'I'): 4, ('V', 'S'): -2,
        ('Z', 'P'): -1, ('V', 'M'): 1, ('T', 'F'): -2, ('V', 'Q'): -3,
        ('K', 'K'): 6, ('P', 'D'): -1, ('I', 'H'): -4, ('I', 'D'): -4,
        ('T', 'R'): -1, ('P', 'L'): -4, ('K', 'G'): -2, ('M', 'N'): -2,
        ('P', 'H'): -2, ('F', 'Q'): -4, ('Z', 'G'): -2, ('X', 'L'): -1,
        ('T', 'M'): -1, ('Z', 'C'): -3, ('X', 'H'): -1, ('D', 'R'): -2,
        ('B', 'W'): -5, ('X', 'D'): -1, ('Z', 'K'): 1, ('F', 'A'): -3,
        ('Z', 'W'): -2, ('F', 'E'): -3, ('D', 'N'): 2, ('B', 'K'): 0,
        ('X', 'X'): -1, ('F', 'I'): 0, ('B', 'G'): -1, ('X', 'T'): 0,
        ('F', 'M'): 0, ('B', 'C'): -3, ('Z', 'I'): -3, ('Z', 'V'): -3,
        ('S', 'S'): 5, ('L', 'Q'): -2, ('W', 'E'): -3, ('Q', 'R'): 1,
        ('N', 'N'): 7, ('W', 'M'): -1, ('Q', 'C'): -3, ('W', 'I'): -3,
        ('S', 'C'): -1, ('L', 'A'): -2, ('S', 'G'): 0, ('L', 'E'): -3,
        ('W', 'Q'): -1, ('H', 'G'): -2, ('S', 'K'): 0, ('Q', 'N'): 0,
        ('N', 'R'): -1, ('H', 'C'): -3, ('Y', 'N'): -2, ('G', 'Q'): -2,
        ('Y', 'F'): 4, ('C', 'A'): -1, ('V', 'L'): 1, ('G', 'E'): -3,
        ('G', 'A'): 0, ('K', 'R'): 3, ('E', 'D'): 2, ('Y', 'R'): -1,
        ('M', 'Q'): 0, ('T', 'I'): -1, ('C', 'D'): -4, ('V', 'F'): -1,
        ('T', 'A'): 0, ('T', 'P'): -1, ('B', 'P'): -2, ('T', 'E'): -1,
        ('V', 'N'): -3, ('P', 'G'): -2, ('M', 'A'): -1, ('K', 'H'): 0,
        ('V', 'R'): -3, ('P', 'C'): -4, ('M', 'E'): -2, ('K', 'L'): -3,
        ('V', 'V'): 5, ('M', 'I'): 2, ('T', 'Q'): -1, ('I', 'G'): -4,
        ('P', 'K'): -1, ('M', 'M'): 7, ('K', 'D'): -1, ('I', 'C'): -2,
        ('Z', 'D'): 1, ('F', 'R'): -3, ('X', 'K'): -1, ('Q', 'D'): 0,
        ('X', 'G'): -2, ('Z', 'L'): -3, ('X', 'C'): -2, ('Z', 'H'): 0,
        ('B', 'L'): -4, ('B', 'H'): 0, ('F', 'F'): 8, ('X', 'W'): -3,
        ('B', 'D'): 5, ('D', 'A'): -2, ('S', 'L'): -3, ('X', 'S'): -1,
        ('F', 'N'): -4, ('S', 'R'): -1, ('W', 'D'): -5, ('V', 'Y'): -1,
        ('W', 'L'): -2, ('H', 'R'): 0, ('W', 'H'): -3, ('H', 'N'): 1,
        ('W', 'T'): -3, ('T', 'T'): 5, ('S', 'F'): -3, ('W', 'P'): -4,
        ('L', 'D'): -4, ('B', 'I'): -4, ('L', 'H'): -3, ('S', 'N'): 1,
        ('B', 'T'): 0, ('L', 'L'): 5, ('Y', 'K'): -2, ('E', 'Q'): 2,
        ('Y', 'G'): -3, ('Z', 'S'): 0, ('Y', 'C'): -3, ('G', 'D'): -1,
        ('B', 'V'): -4, ('E', 'A'): -1, ('Y', 'W'): 2, ('E', 'E'): 6,
        ('Y', 'S'): -2, ('C', 'N'): -2, ('V', 'C'): -1, ('T', 'H'): -2,
        ('P', 'R'): -3, ('V', 'G'): -4, ('T', 'L'): -1, ('V', 'K'): -3,
        ('K', 'Q'): 2, ('R', 'A'): -2, ('I', 'R'): -4, ('T', 'D'): -1,
        ('P', 'F'): -4, ('I', 'N'): -3, ('K', 'I'): -3, ('M', 'D'): -4,
        ('V', 'W'): -3, ('W', 'W'): 15, ('M', 'H'): -1, ('P', 'N'): -2,
        ('K', 'A'): -1, ('M', 'L'): 3, ('K', 'E'): 1, ('Z', 'E'): 5,
        ('X', 'N'): -1, ('Z', 'A'): -1, ('Z', 'M'): -1, ('X', 'F'): -2,
        ('K', 'C'): -3, ('B', 'Q'): 0, ('X', 'B'): -1, ('B', 'M'): -3,
        ('F', 'C'): -2, ('Z', 'Q'): 4, ('X', 'Z'): -1, ('F', 'G'): -4,
        ('B', 'E'): 1, ('X', 'V'): -1, ('F', 'K'): -4, ('B', 'A'): -2,
        ('X', 'R'): -1, ('D', 'D'): 8, ('W', 'G'): -3, ('Z', 'F'): -4,
        ('S', 'Q'): 0, ('W', 'C'): -5, ('W', 'K'): -3, ('H', 'Q'): 1,
        ('L', 'C'): -2, ('W', 'N'): -4, ('S', 'A'): 1, ('L', 'G'): -4,
        ('W', 'S'): -4, ('S', 'E'): -1, ('H', 'E'): 0, ('S', 'I'): -3,
        ('H', 'A'): -2, ('S', 'M'): -2, ('Y', 'L'): -1, ('Y', 'H'): 2,
        ('Y', 'D'): -3, ('E', 'R'): 0, ('X', 'P'): -2, ('G', 'G'): 8,
        ('G', 'C'): -3, ('E', 'N'): 0, ('Y', 'T'): -2, ('Y', 'P'): -3,
        ('T', 'K'): -1, ('A', 'A'): 5, ('P', 'Q'): -1, ('T', 'C'): -1,
        ('V', 'H'): -4, ('T', 'G'): -2, ('I', 'Q'): -3, ('Z', 'T'): -1,
        ('C', 'R'): -4, ('V', 'P'): -3, ('P', 'E'): -1, ('M', 'C'): -2,
        ('K', 'N'): 0, ('I', 'I'): 5, ('P', 'A'): -1, ('M', 'G'): -3,
        ('T', 'S'): 2, ('I', 'E'): -4, ('P', 'M'): -3, ('M', 'K'): -2,
        ('I', 'A'): -1, ('P', 'I'): -3, ('R', 'R'): 7, ('X', 'M'): -1,
        ('L', 'I'): 2, ('X', 'I'): -1, ('Z', 'B'): 2, ('X', 'E'): -1,
        ('Z', 'N'): 0, ('X', 'A'): -1, ('B', 'R'): -1, ('B', 'N'): 4,
        ('F', 'D'): -5, ('X', 'Y'): -1, ('Z', 'R'): 0, ('F', 'H'): -1,
        ('B', 'F'): -4, ('F', 'L'): 1, ('X', 'Q'): -1, ('B', 'B'): 5
}
