# You can do all of this in the `__main__.py` file, but this file exists
# to shows how to do relative import functions from another python file in
# the same directory as this one.
import numpy as np
import matplotlib
matplotlib.use('TkAgg') #without this, plt.anything causes an NSException uncaught cascade...?
import matplotlib.pyplot as plt
from .algs import quicksort, bubblesort
from .time import time_test,time_graphs
from .counts import counts_test,counts_graphs

def run_stuff():
    """
    This function is called in `__main__.py`
    """
    print("This is `run()` from ", __file__)

    x = np.random.randint(100,size=10)
    print("Unsorted input: ", x)

    print("Bubble sort output: ", bubblesort(x))
    print("Quick sort output: ", quicksort(x))

    bt,qt = time_test()
    time_graphs(bt,qt)

    bc,qc = counts_test()
    counts_graphs(bc,qc)
