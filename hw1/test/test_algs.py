import numpy as np
from hw1 import algs

def test_pointless_sort():
    # generate random vector of length 10
    x = np.random.rand(10)

    # check that pointless_sort always returns [1,2,3]
    assert np.array_equal(algs.pointless_sort(x), np.array([1,2,3]))

    # generate a new random vector of length 10
    x = np.random.rand(10)

    # check that pointless_sort still returns [1,2,3]
    assert np.array_equal(algs.pointless_sort(x), np.array([1,2,3]))

def test_bubblesort():
    # Actually test bubblesort here. It might be useful to think about
    # some edge cases for your code, where it might fail. Some things to
    # think about: (1) does your code handle 0-element arrays without
    # failing, (2) does your code handle characters?

    # test odd entries
    x = np.array([1000,2,48,234,789])
    assert np.array_equal(algs.bubblesort(x),np.array([2,48,234,789,1000]))

    # test even entries
    x = np.array([1000,2,48,234,789,3])
    assert np.array_equal(algs.bubblesort(x),np.array([2,3,48,234,789,1000]))

    # test repeated entries
    x = np.array([1,2,4,0,1])
    assert np.array_equal(algs.bubblesort(x),np.array([0,1,1,2,4]))

    # test multiple different repeats
    x = np.array([5,7,8,2,1,5,2,2,1,8,5,2])
    assert np.array_equal(algs.bubblesort(x),np.array([1,1,2,2,2,2,5,5,5,7,8,8]))

    # check that the zero-length vector produces a reasonable output
    x = np.array([])
    assert np.array_equal(algs.bubblesort(x),np.array([]))

    # test 1-length vector
    x = np.array([5])
    assert np.array_equal(algs.bubblesort(x),np.array([5]))

    # test character arrays, which should (according to python < / > rules)
    # be put into alphabetical order (uppercase first)
    x = np.array(['a','b','z','y','C'])
    assert np.array_equal(algs.bubblesort(x),np.array(['C','a','b','y','z']))


def test_quicksort():

    # simple initial test
    x = np.array([9,8,7,6])
    assert np.array_equal(algs.quicksort(x),np.array([6,7,8,9]))

    # same but with an odd number of elements
    x = np.array([9,8,101,7,6])
    assert np.array_equal(algs.quicksort(x),np.array([6,7,8,9,101]))

    # test repeated elements
    x = np.array([1,2,4,0,1])
    assert np.array_equal(algs.quicksort(x),np.array([0,1,1,2,4]))

    # test multiple different repeats
    x = np.array([5,7,8,2,1,5,2,2,1,8,5,2])
    assert np.array_equal(algs.quicksort(x),np.array([1,1,2,2,2,2,5,5,5,7,8,8]))

    # check that the zero-length vector produces a reasonable output
    x = np.array([])
    assert np.array_equal(algs.quicksort(x),np.array([]))

    # test 1-length vector
    x = np.array([5])
    assert np.array_equal(algs.quicksort(x),np.array([5]))

    # test character arrays, which should (according to python < / > rules)
    # be put into alphabetical order (uppercase first)
    x = np.array(['a','b','z','y','C'])
    assert np.array_equal(algs.quicksort(x),np.array(['C','a','b','y','z']))
    x = np.array([1,2,4,0,1])
