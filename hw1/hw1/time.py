# use timeit to measure duration of different
# sorting functions with varied input lengths
import time
import numpy as np
import matplotlib.pyplot as plt
from .algs import quicksort, bubblesort

def time_alg(a):
    times = []
    # range of lengths to test is 100, 200, ... 1000
    for n in range(100,1100,100):
        # create a set of 100 random vectors of that size
        x = [ np.random.randint(100,size=n) ]*100
        t0 = time.time()
        sorted = [ a(e) for e in x ]
        t1 = time.time()
        print("100 vectors of length %s: %s sec" % (n, t1-t0))
        times.append(t1-t0)
    return times

def time_test():
    print("** Timing test for Bubblesort and Quicksort **")
    print("- BUBBLESORT -")
    bt = time_alg(bubblesort)
    print("")
    print("- QUICKSORT -")
    qt = time_alg(quicksort)
    print("")
    print("** end **")
    return bt,qt

def time_graphs(bt,qt):
    x = [ n for n in range(100,1100,100)]
    x2 = [ n for n in range(100,1001,1)]
    n2 = [ (n**2)/40000 for n in x2 ]
    nlogn = [ (n*np.log2(n))/2000 for n in x2 ]

    ## make graphs, save to png
    fig3,ax3 = plt.subplots(figsize=(9,6))
    ax3.scatter(x,bt,label='bubblesort')
    ax3.scatter(x,qt,marker="s",label='quicksort')
    ax3.plot(x2,nlogn,color="black",label="C*NlogN")
    ax3.plot(x2,n2,color="black",linestyle="--",label="C*N^2")
    ax3.set_xlabel("Length N of Input Vector")
    ax3.set_ylabel("Time to Sort 100 Vectors (seconds)")
    ax3.set_title('Timing Comparison, Linear Scale')
    ax3.legend()
    plt.savefig("fig3.png",bbox_inches="tight")

    fig4,ax4 = plt.subplots(figsize=(9,6))
    ax4.set_yscale('log')
    ax4.scatter(x,bt,label='bubblesort')
    ax4.scatter(x,qt,label='quicksort')
    ax4.plot(x2,nlogn,color="gray",label='C*NlogN')
    ax4.plot(x2,n2,color="gray",linestyle="--",label='C*N^2')
    ax4.set_xlabel("Length N of Input Vector")
    ax4.set_ylabel("Time to Sort 100 Vectors (seconds)")
    ax4.set_title('Timing Comparison, Log Scale')
    ax4.legend()
    plt.savefig("fig4.png",bbox_inches="tight")
