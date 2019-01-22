import numpy as np

def pointless_sort(x):
    """
    This function always returns the same values to show how testing
    works, check out the `test/test_alg.py` file to see.
    """
    return np.array([1,2,3])

def bubblesort(x):
    """
    This function takes in a list x and sorts it by
    iterating through the elements pairwise and swapping
    the order of the pairs according to the sorting rule,
    repeating this entire process once for each element
    in the input list
    """
    for n in range(len(x)):
        for i in range(len(x)-1): #comparing adjacent, so iterate up to 2nd-to-last
            if x[i] > x[i+1]:
                saveBit = x[i+1]
                x[i+1] = x[i]
                x[i] = saveBit
    return x
    ## Number of Assignments:
    ## O(N^2)  (x, n*N, i*N*N-1)
    ## Number of Conditionals:
    ## O(N^2)  (pairwise comparison)*N*N-1

def quicksort(x):
    """
    This function takes in a list x and sorts it by picking the
    first element as the pivot and then iterating through the rest
    of the list and sorting the elements into one of two daughter lists,
    larger than the pivot and smaller than the pivot. Then the same
    algorithm is run recursively on the daughter lists until finished.
    """
    if len(x) < 2:
        return x
    else:
        pivot = x[0]
        lt,gt = np.array([])
        for n in x[1:]:
            if n < pivot:
                lt = np.append(lt,n)
            else:
                gt = np.append(gt,n)
    return np.concatenate((quicksort(lt),np.array([pivot]),quicksort(gt)),axis=0)
    ## Number of Assignments:
    ##  O(NlogN)  ((x, pivot, lt*N, gt*N, n*N) * logN)
    ## Number of Conditionals:
    ##  O(NlogN)   (base case)*logN, (partition)*N*logN
