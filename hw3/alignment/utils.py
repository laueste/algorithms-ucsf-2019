# Some utility classes to represent a fasta sequence and a scoring matrix

class Sequence:
    """
    A simple class for a sequence (protein or DNA) from a FASTA file
    """

    def __init__(self, name, sequence='', header=''):
        self.name = name #the file name
        self.header = header #the name in the file following '>'
        self.sequence = sequence

    # Overload the __repr__ operator to make printing simpler.
    def __repr__(self):
        return self.name

# class ScoringMatrix:
#     """
#     A simple class for an amino acid residue
#     """
#
#     def __init__(self, type):
#         self.type = type
#         self.coords = (0.0, 0.0, 0.0)
#
#     # Overload the __repr__ operator to make printing simpler.
#     def __repr__(self):
#         return self.type
