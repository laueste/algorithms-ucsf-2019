# Some utility classes to represent a PDB structure

class Atom:
    """
    A simple class for an amino acid residue
    """

    def __init__(self, type):
        self.type = type
        self.coords = (0.0, 0.0, 0.0)

    # Overload the __repr__ operator to make printing simpler.
    def __repr__(self):
        return self.type

class Residue:
    """
    A simple class for an amino acid residue
    """

    def __init__(self, type, number):
        self.type = type
        self.number = number
        self.atoms = []

    # Overload the __repr__ operator to make printing simpler.
    def __repr__(self):
        return "{0} {1}".format(self.type, self.number)

class ActiveSite:
    """
    A simple class for an active site
    """

    def __init__(self, name):
        self.name = name
        self.residues = []

    # Overload the __repr__ operator to make printing simpler.
    def __repr__(self):
        return self.name

class Cluster:
    """
    A simple class to represent a cluster (can contain one or more
    ActiveSites or other clusters)
    """
    def __init__(self,members=[]):
        self.members = members
        self.name = "c" + str(members)

    def __repr__(self):
        # Overload the __repr__ operator to make printing simpler.
        return self.name

    def compute_centroid(self):
        
