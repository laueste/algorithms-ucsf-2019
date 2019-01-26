# implement algorithms with counting
# of conditionals and Assignments
import numpy as np
import matplotlib
matplotlib.use('TkAgg') #to fix a weird bug that causes an NSException anytime plt.____ is called
import matplotlib.pyplot as plt

def bubblesort_count(x):
    """
    This function takes in a list x and sorts it by
    iterating through the elements pairwise and swapping
    the order of the pairs according to the sorting rule,
    repeating this entire process once for each element
    in the input list
    """
    assigments_count = 0
    conditionals_count = 0
    for n in range(len(x)):
        assigments_count += 1
        for i in range(len(x)-1): #comparing adjacent, so iterate up to 2nd-to-last
            assigments_count += 1
            if x[i] > x[i+1]:
                conditionals_count +=1
                assigments_count += 3
                saveBit = x[i+1]
                x[i+1] = x[i]
                x[i] = saveBit
    return x, (assigments_count,conditionals_count)
    ## Number of Assignments:
    ## O(N^2)  (x, n*N, i*N*N-1)
    ## Number of Conditionals:
    ## O(N^2)  (pairwise comparison ln20)*N*N-1

def quicksort_count(x):
    """
    Outer main function for quicksort to allow the count
    variables to persist across the recursion
    """
    ac = 0
    cc = 0

    def quicksort_inner(x):
        """
        This function takes in a list x and sorts it by picking the
        first element as the pivot and then iterating through the rest
        of the list and sorting the elements into one of two daughter lists,
        larger than the pivot and smaller than the pivot. Then the same
        algorithm is run recursively on the daughter lists until finished.
        """
        nonlocal ac
        nonlocal cc
        cc += 1
        if len(x) < 2:
            return x
        else:
            ac += 3
            pivot = x[0]
            lt = np.array([])
            gt = np.array([])
            for n in x[1:]:
                cc += 1
                if n < pivot:
                    ac += 1
                    lt = np.append(lt,n)
                else:
                    ac += 1
                    gt = np.append(gt,n)
        ac += 1
        return np.concatenate((quicksort_inner(lt),np.array([pivot]),quicksort_inner(gt)),axis=0)
        ## Number of Assignments:
        ##  O(NlogN)  ((x, pivot, lt*N, gt*N, n*N) * logN)
        ## Number of Conditionals:
        ##  O(NlogN)   (base case ln38)*logN, (partition ln44)*N*logN

    return quicksort_inner(x),(ac,cc)

def counts_alg(a):
    assignment_counts = []
    conditional_counts = []
    # range of lengths to test is 100, 200, ... 1000
    for n in range(100,1100,100):
        # create a set of 100 random vectors of that size
        x = [ np.random.randint(100,size=n) ]*100
        ac_x = []
        cc_x = []
        for v in x:
            result,counts = a(v)
            ac_x.append(counts[0])
            cc_x.append(counts[1])
        ac = np.mean(ac_x)
        cc = np.mean(cc_x)
        assignment_counts.append(ac)
        conditional_counts.append(cc)
        print("100 vectors of length %s: %s A %s C" % (n, ac,cc))
    return assignment_counts,conditional_counts


def counts_test():
    print("** Operation Counts Test for Bubblesort and Quicksort **")
    print("- BUBBLESORT -")
    bcRes = counts_alg(bubblesort_count)
    print("")
    print("- QUICKSORT -")
    qcRes = counts_alg(quicksort_count)
    print("")
    print("** end **")
    return (bcRes,qcRes)

def counts_graphs(bcRes,qcRes):
    bc_a,bc_c = bcRes
    qc_a,qc_c = qcRes
    x = [ n for n in range(100,1100,100)]
    x2 = [ n for n in range(100,1001,1)]
    n2 = [ n**2 for n in x2 ]
    nlogn = [ 2*n*np.log2(n) for n in x2 ]

    ## make graphs, save to png
    # Note that there is a LOT of tweaking of the constants C that
    # make the curves line up well to the data (just scaling/translation)
    fig1,(ax1a,ax1b) = plt.subplots(1,2,figsize=(18,6))
    ax1a.scatter(x,bc_a,label='bubblesort assignments')
    ax1b.scatter(x,bc_c,label='bubblesort conditionals')
    ax1a.scatter(x,qc_a,marker="s",label='quicksort assignments')
    ax1b.scatter(x,qc_c,marker="s",label='quicksort conditionals')
    ax1a.plot(x2,[n/1.2 for n in nlogn],color="black",label='C*NlogN')
    ax1a.plot(x2,n2,color="black",linestyle="--",label='C*N^2')
    ax1b.plot(x2,[n/1.5 for n in nlogn],color="black",label='C*NlogN')
    ax1b.plot(x2,[n/350 for n in n2],color="black",linestyle="--",label='C*N^2')
    ax1a.set_xlabel("Length N of Input Vector")
    ax1a.set_ylabel("Assignments Used to Sort 100 Vectors")
    ax1a.set_title('Assignments Comparison, Linear Scale')
    ax1a.legend()
    ax1b.set_xlabel("Length N of Input Vector")
    ax1b.set_ylabel("Conditionals Used to Sort 100 Vectors")
    ax1b.set_title('Conditionals Comparison, Linear Scale')
    ax1b.legend()
    plt.savefig("fig1.png",bbox_inches="tight")

    fig2,(ax2a,ax2b) = plt.subplots(1,2,figsize=(18,6))
    ax2a.set_yscale('log')
    ax2b.set_yscale('log')
    ax2a.scatter(x,bc_a,label='bubblesort assignments')
    ax2b.scatter(x,bc_c,label='bubblesort conditionals')
    ax2a.scatter(x,qc_a,marker="s",label='quicksort assignments')
    ax2b.scatter(x,qc_c,marker="s",label='quicksort conditionals')
    ax2a.plot(x2,[n/1.2 for n in nlogn],color="gray",label='C*NlogN')
    ax2a.plot(x2,n2,color="gray",linestyle="--",label='C*N^2')
    ax2b.plot(x2,[n/1.5 for n in nlogn],color="gray",label='C*NlogN')
    #ax2b.plot(x2,[(n*np.log2(n) + np.log2(n))/1.5 for n in x2],color="gray",linestyle=":",label='C*(NlogN+logN)')
    ax2b.plot(x2,[n/375 for n in n2],color="gray",linestyle="--",label='C*N^2')
    ax2a.set_xlabel("Length N of Input Vector")
    ax2a.set_ylabel("Assignments Used to Sort 100 Vectors")
    ax2a.set_title('Assignments Comparison, Log Scale')
    ax2a.legend()
    ax2b.set_xlabel("Length N of Input Vector")
    ax2b.set_ylabel("Conditionals Used to Sort 100 Vectors")
    ax2b.set_title('Conditionals Comparison, Log Scale')
    ax2b.legend()
    plt.savefig("fig2.png",bbox_inches="tight")

    fig5,ax5 = plt.subplots(figsize=(9,6))
    ax5.set_ylim(top=4000)
    ax5.scatter(x,bc_c,label='bubblesort conditionals')
    ax5.scatter(x,qc_c,marker="s",label='quicksort conditionals')
    ax5.scatter(x,[n/20 for n in qc_c],marker='s',label='K*(quicksort conditionals)')
    ax5.plot(x2,[n/30 for n in nlogn],color="gray",linestyle=":",label='D*NlogN')
    ax5.plot(x2,[n/1.5 for n in nlogn],color="gray",label='C*NlogN')
    #ax5.plot(x2,[(n*np.log2(n) + np.log2(n))/1.5 for n in x2],color="gray",linestyle=":",label='C*(NlogN+logN)')
    ax5.plot(x2,[n/375 for n in n2],color="gray",linestyle="--",label='C*N^2')
    ax5.set_xlabel("Length N of Input Vector")
    ax5.set_ylabel("Conditionals Used to Sort 100 Vectors")
    ax5.set_title('Conditionals Comparison, Linear Scale (Addendum)')
    ax5.legend()
    plt.savefig("fig5.png",bbox_inches="tight")
