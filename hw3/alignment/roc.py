# file for making ROC curves and testing multiple matrices (Part I Q2 & Q3)
import matplotlib
matplotlib.use('TkAgg') #to fix a weird bug that causes an NSException anytime plt.____ is called
import matplotlib.pyplot as plt
import numpy as np
from .truefalse import calculate_false_positive_rate

def compute_roc(pos_scores,neg_scores,stepsize=0.05):
    """
    For a given set of positive and negative alignment scores, computes the
    false positive rate for a range of true positive rates from 0 to 1,
    with the number of points determined by the input stepsize
    (default 0.05 -> 20 points)
    Input: array of tuples of [score,s1name,s2name] for positive alignments,
           array of tuples of [score,s1name,s2name] for negative alignments,
           optional stepsize to determine number of points to calculate
    Output: pair of lists for x(true pos frac) and y(false pos frac) ROC values
    """
    x = []
    y = []
    for i in np.linspace(0,1,1/stepsize):
        x.append(i)
        y.append(calculate_false_positive_rate(pos_scores,neg_scores,true_pos_frac=i))
    print('ROC x',x)
    print('ROC y',y)
    return x,y

def graph_rocs(xs,ys,labels,
                title='graph',colors=['C'+str(i) for i in range(10)]):
    """
    Graphs x_i vs y_i with lable labels_i for all i in the lists xs,ys,labels.
    Makes sure that the graph is square.

    Input: list of lists of: x-values, y-values, and labels, optional title
    Output: none, saves a plot
    """
    fig,ax = plt.subplots(figsize=(6,6))
    ax.axis('equal')
    for x,y,l,c in zip(xs,ys,labels,colors):
        ax.plot(x,y,c,label=l)
    ax.set_xlabel('True Positive Fraction')
    ax.set_ylabel('False Positive Fraction')
    ax.set_title(title)
    ax.legend()
    plt.savefig('%s.png' % title)


#### SPECIFIC MATRIX SCORES (determined empirically and saved here) ####


## BLOSUM50:

b50x = [0.0, 0.010101010101010102, 0.020202020202020204, 0.030303030303030304, 0.04040404040404041, 0.05050505050505051, 0.06060606060606061, 0.07070707070707072, 0.08080808080808081, 0.09090909090909091, 0.10101010101010102, 0.11111111111111112, 0.12121212121212122, 0.13131313131313133, 0.14141414141414144, 0.15151515151515152, 0.16161616161616163, 0.17171717171717174, 0.18181818181818182, 0.19191919191919193, 0.20202020202020204, 0.21212121212121213, 0.22222222222222224, 0.23232323232323235, 0.24242424242424243, 0.25252525252525254, 0.26262626262626265, 0.27272727272727276, 0.2828282828282829, 0.29292929292929293, 0.30303030303030304, 0.31313131313131315, 0.32323232323232326, 0.33333333333333337, 0.3434343434343435, 0.3535353535353536, 0.36363636363636365, 0.37373737373737376, 0.38383838383838387, 0.393939393939394, 0.4040404040404041, 0.4141414141414142, 0.42424242424242425, 0.43434343434343436, 0.4444444444444445, 0.4545454545454546, 0.4646464646464647, 0.4747474747474748, 0.48484848484848486, 0.494949494949495, 0.5050505050505051, 0.5151515151515152, 0.5252525252525253, 0.5353535353535354, 0.5454545454545455, 0.5555555555555556, 0.5656565656565657, 0.5757575757575758, 0.5858585858585859, 0.595959595959596, 0.6060606060606061, 0.6161616161616162, 0.6262626262626263, 0.6363636363636365, 0.6464646464646465, 0.6565656565656566, 0.6666666666666667, 0.6767676767676768, 0.686868686868687, 0.696969696969697, 0.7070707070707072, 0.7171717171717172, 0.7272727272727273, 0.7373737373737375, 0.7474747474747475, 0.7575757575757577, 0.7676767676767677, 0.7777777777777778, 0.787878787878788, 0.797979797979798, 0.8080808080808082, 0.8181818181818182, 0.8282828282828284, 0.8383838383838385, 0.8484848484848485, 0.8585858585858587, 0.8686868686868687, 0.8787878787878789, 0.888888888888889, 0.8989898989898991, 0.9090909090909092, 0.9191919191919192, 0.9292929292929294, 0.9393939393939394, 0.9494949494949496, 0.9595959595959597, 0.9696969696969697, 0.9797979797979799, 0.98989898989899, 1.0]
b50y = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.04, 0.08, 0.08, 0.08, 0.08, 0.1, 0.1, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.18, 0.18, 0.28, 0.28, 0.3, 0.3, 0.32, 0.32, 0.32, 0.32, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.4, 0.4, 0.46, 0.46, 0.46, 0.46, 0.46, 0.46, 0.48, 0.48, 0.48, 0.48, 0.5, 0.5, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.54, 0.54, 0.62, 0.62, 0.64, 0.64, 0.7, 0.7, 0.7, 0.7, 0.74, 0.74, 0.76, 0.76, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9]


## BLOSUM62

b62x = [0.0, 0.010101010101010102, 0.020202020202020204, 0.030303030303030304, 0.04040404040404041, 0.05050505050505051, 0.06060606060606061, 0.07070707070707072, 0.08080808080808081, 0.09090909090909091, 0.10101010101010102, 0.11111111111111112, 0.12121212121212122, 0.13131313131313133, 0.14141414141414144, 0.15151515151515152, 0.16161616161616163, 0.17171717171717174, 0.18181818181818182, 0.19191919191919193, 0.20202020202020204, 0.21212121212121213, 0.22222222222222224, 0.23232323232323235, 0.24242424242424243, 0.25252525252525254, 0.26262626262626265, 0.27272727272727276, 0.2828282828282829, 0.29292929292929293, 0.30303030303030304, 0.31313131313131315, 0.32323232323232326, 0.33333333333333337, 0.3434343434343435, 0.3535353535353536, 0.36363636363636365, 0.37373737373737376, 0.38383838383838387, 0.393939393939394, 0.4040404040404041, 0.4141414141414142, 0.42424242424242425, 0.43434343434343436, 0.4444444444444445, 0.4545454545454546, 0.4646464646464647, 0.4747474747474748, 0.48484848484848486, 0.494949494949495, 0.5050505050505051, 0.5151515151515152, 0.5252525252525253, 0.5353535353535354, 0.5454545454545455, 0.5555555555555556, 0.5656565656565657, 0.5757575757575758, 0.5858585858585859, 0.595959595959596, 0.6060606060606061, 0.6161616161616162, 0.6262626262626263, 0.6363636363636365, 0.6464646464646465, 0.6565656565656566, 0.6666666666666667, 0.6767676767676768, 0.686868686868687, 0.696969696969697, 0.7070707070707072, 0.7171717171717172, 0.7272727272727273, 0.7373737373737375, 0.7474747474747475, 0.7575757575757577, 0.7676767676767677, 0.7777777777777778, 0.787878787878788, 0.797979797979798, 0.8080808080808082, 0.8181818181818182, 0.8282828282828284, 0.8383838383838385, 0.8484848484848485, 0.8585858585858587, 0.8686868686868687, 0.8787878787878789, 0.888888888888889, 0.8989898989898991, 0.9090909090909092, 0.9191919191919192, 0.9292929292929294, 0.9393939393939394, 0.9494949494949496, 0.9595959595959597, 0.9696969696969697, 0.9797979797979799, 0.98989898989899, 1.0]
b62y = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.04, 0.08, 0.08, 0.1, 0.1, 0.1, 0.1, 0.14, 0.14, 0.18, 0.18, 0.24, 0.24, 0.26, 0.26, 0.32, 0.32, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.36, 0.36, 0.36, 0.36, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.46, 0.46, 0.46, 0.46, 0.46, 0.46, 0.48, 0.48, 0.48, 0.48, 0.5, 0.5, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.58, 0.58, 0.6, 0.6, 0.66, 0.66, 0.66, 0.66, 0.72, 0.72, 0.72, 0.72, 0.74, 0.74, 0.78, 0.78, 0.84, 0.84, 0.92, 0.92, 0.92, 0.92, 0.94]

## PAM100
pam100x = [0.0, 0.010101010101010102, 0.020202020202020204, 0.030303030303030304, 0.04040404040404041, 0.05050505050505051, 0.06060606060606061, 0.07070707070707072, 0.08080808080808081, 0.09090909090909091, 0.10101010101010102, 0.11111111111111112, 0.12121212121212122, 0.13131313131313133, 0.14141414141414144, 0.15151515151515152, 0.16161616161616163, 0.17171717171717174, 0.18181818181818182, 0.19191919191919193, 0.20202020202020204, 0.21212121212121213, 0.22222222222222224, 0.23232323232323235, 0.24242424242424243, 0.25252525252525254, 0.26262626262626265, 0.27272727272727276, 0.2828282828282829, 0.29292929292929293, 0.30303030303030304, 0.31313131313131315, 0.32323232323232326, 0.33333333333333337, 0.3434343434343435, 0.3535353535353536, 0.36363636363636365, 0.37373737373737376, 0.38383838383838387, 0.393939393939394, 0.4040404040404041, 0.4141414141414142, 0.42424242424242425, 0.43434343434343436, 0.4444444444444445, 0.4545454545454546, 0.4646464646464647, 0.4747474747474748, 0.48484848484848486, 0.494949494949495, 0.5050505050505051, 0.5151515151515152, 0.5252525252525253, 0.5353535353535354, 0.5454545454545455, 0.5555555555555556, 0.5656565656565657, 0.5757575757575758, 0.5858585858585859, 0.595959595959596, 0.6060606060606061, 0.6161616161616162, 0.6262626262626263, 0.6363636363636365, 0.6464646464646465, 0.6565656565656566, 0.6666666666666667, 0.6767676767676768, 0.686868686868687, 0.696969696969697, 0.7070707070707072, 0.7171717171717172, 0.7272727272727273, 0.7373737373737375, 0.7474747474747475, 0.7575757575757577, 0.7676767676767677, 0.7777777777777778, 0.787878787878788, 0.797979797979798, 0.8080808080808082, 0.8181818181818182, 0.8282828282828284, 0.8383838383838385, 0.8484848484848485, 0.8585858585858587, 0.8686868686868687, 0.8787878787878789, 0.888888888888889, 0.8989898989898991, 0.9090909090909092, 0.9191919191919192, 0.9292929292929294, 0.9393939393939394, 0.9494949494949496, 0.9595959595959597, 0.9696969696969697, 0.9797979797979799, 0.98989898989899, 1.0]
pam100y = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.04, 0.08, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.16, 0.16, 0.26, 0.26, 0.28, 0.28, 0.3, 0.3, 0.32, 0.32, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.36, 0.36, 0.36, 0.36, 0.38, 0.38, 0.38, 0.38, 0.46, 0.46, 0.46, 0.46, 0.46, 0.46, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.52, 0.52, 0.52, 0.52, 0.54, 0.54, 0.54, 0.54, 0.56, 0.56, 0.64, 0.64, 0.64, 0.64, 0.72, 0.72, 0.72, 0.72, 0.74, 0.74, 0.78, 0.78, 0.84, 0.84, 0.9, 0.9, 0.94, 0.94, 0.96]

## PAM250

pam250x = [0.0, 0.010101010101010102, 0.020202020202020204, 0.030303030303030304, 0.04040404040404041, 0.05050505050505051, 0.06060606060606061, 0.07070707070707072, 0.08080808080808081, 0.09090909090909091, 0.10101010101010102, 0.11111111111111112, 0.12121212121212122, 0.13131313131313133, 0.14141414141414144, 0.15151515151515152, 0.16161616161616163, 0.17171717171717174, 0.18181818181818182, 0.19191919191919193, 0.20202020202020204, 0.21212121212121213, 0.22222222222222224, 0.23232323232323235, 0.24242424242424243, 0.25252525252525254, 0.26262626262626265, 0.27272727272727276, 0.2828282828282829, 0.29292929292929293, 0.30303030303030304, 0.31313131313131315, 0.32323232323232326, 0.33333333333333337, 0.3434343434343435, 0.3535353535353536, 0.36363636363636365, 0.37373737373737376, 0.38383838383838387, 0.393939393939394, 0.4040404040404041, 0.4141414141414142, 0.42424242424242425, 0.43434343434343436, 0.4444444444444445, 0.4545454545454546, 0.4646464646464647, 0.4747474747474748, 0.48484848484848486, 0.494949494949495, 0.5050505050505051, 0.5151515151515152, 0.5252525252525253, 0.5353535353535354, 0.5454545454545455, 0.5555555555555556, 0.5656565656565657, 0.5757575757575758, 0.5858585858585859, 0.595959595959596, 0.6060606060606061, 0.6161616161616162, 0.6262626262626263, 0.6363636363636365, 0.6464646464646465, 0.6565656565656566, 0.6666666666666667, 0.6767676767676768, 0.686868686868687, 0.696969696969697, 0.7070707070707072, 0.7171717171717172, 0.7272727272727273, 0.7373737373737375, 0.7474747474747475, 0.7575757575757577, 0.7676767676767677, 0.7777777777777778, 0.787878787878788, 0.797979797979798, 0.8080808080808082, 0.8181818181818182, 0.8282828282828284, 0.8383838383838385, 0.8484848484848485, 0.8585858585858587, 0.8686868686868687, 0.8787878787878789, 0.888888888888889, 0.8989898989898991, 0.9090909090909092, 0.9191919191919192, 0.9292929292929294, 0.9393939393939394, 0.9494949494949496, 0.9595959595959597, 0.9696969696969697, 0.9797979797979799, 0.98989898989899, 1.0]
pam250y =[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.04, 0.08, 0.08, 0.1, 0.1, 0.12, 0.12, 0.12, 0.12, 0.18, 0.18, 0.26, 0.26, 0.26, 0.26, 0.3, 0.3, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.36, 0.36, 0.38, 0.38, 0.38, 0.38, 0.42, 0.42, 0.42, 0.42, 0.44, 0.44, 0.46, 0.46, 0.46, 0.46, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.52, 0.52, 0.52, 0.52, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56, 0.66, 0.66, 0.66, 0.66, 0.68, 0.68, 0.7, 0.7, 0.78, 0.78, 0.8, 0.8, 0.84, 0.84, 0.9, 0.9, 0.9, 0.9, 0.92]

## MATIO
matx = [0.0, 0.010101010101010102, 0.020202020202020204, 0.030303030303030304, 0.04040404040404041, 0.05050505050505051, 0.06060606060606061, 0.07070707070707072, 0.08080808080808081, 0.09090909090909091, 0.10101010101010102, 0.11111111111111112, 0.12121212121212122, 0.13131313131313133, 0.14141414141414144, 0.15151515151515152, 0.16161616161616163, 0.17171717171717174, 0.18181818181818182, 0.19191919191919193, 0.20202020202020204, 0.21212121212121213, 0.22222222222222224, 0.23232323232323235, 0.24242424242424243, 0.25252525252525254, 0.26262626262626265, 0.27272727272727276, 0.2828282828282829, 0.29292929292929293, 0.30303030303030304, 0.31313131313131315, 0.32323232323232326, 0.33333333333333337, 0.3434343434343435, 0.3535353535353536, 0.36363636363636365, 0.37373737373737376, 0.38383838383838387, 0.393939393939394, 0.4040404040404041, 0.4141414141414142, 0.42424242424242425, 0.43434343434343436, 0.4444444444444445, 0.4545454545454546, 0.4646464646464647, 0.4747474747474748, 0.48484848484848486, 0.494949494949495, 0.5050505050505051, 0.5151515151515152, 0.5252525252525253, 0.5353535353535354, 0.5454545454545455, 0.5555555555555556, 0.5656565656565657, 0.5757575757575758, 0.5858585858585859, 0.595959595959596, 0.6060606060606061, 0.6161616161616162, 0.6262626262626263, 0.6363636363636365, 0.6464646464646465, 0.6565656565656566, 0.6666666666666667, 0.6767676767676768, 0.686868686868687, 0.696969696969697, 0.7070707070707072, 0.7171717171717172, 0.7272727272727273, 0.7373737373737375, 0.7474747474747475, 0.7575757575757577, 0.7676767676767677, 0.7777777777777778, 0.787878787878788, 0.797979797979798, 0.8080808080808082, 0.8181818181818182, 0.8282828282828284, 0.8383838383838385, 0.8484848484848485, 0.8585858585858587, 0.8686868686868687, 0.8787878787878789, 0.888888888888889, 0.8989898989898991, 0.9090909090909092, 0.9191919191919192, 0.9292929292929294, 0.9393939393939394, 0.9494949494949496, 0.9595959595959597, 0.9696969696969697, 0.9797979797979799, 0.98989898989899, 1.0]
maty = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.02, 0.06, 0.06, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.14, 0.14, 0.18, 0.18, 0.3, 0.3, 0.3, 0.3, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.38, 0.38, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.46, 0.46, 0.46, 0.46, 0.46, 0.46, 0.5, 0.5, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.54, 0.54, 0.62, 0.62, 0.66, 0.66, 0.66, 0.66, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72, 0.84, 0.84, 0.86, 0.86, 0.94, 0.94, 0.96, 0.96, 0.96]

## BLOSUM50 with Normalized Scores
b50Nx = [0.0, 0.010101010101010102, 0.020202020202020204, 0.030303030303030304, 0.04040404040404041, 0.05050505050505051, 0.06060606060606061, 0.07070707070707072, 0.08080808080808081, 0.09090909090909091, 0.10101010101010102, 0.11111111111111112, 0.12121212121212122, 0.13131313131313133, 0.14141414141414144, 0.15151515151515152, 0.16161616161616163, 0.17171717171717174, 0.18181818181818182, 0.19191919191919193, 0.20202020202020204, 0.21212121212121213, 0.22222222222222224, 0.23232323232323235, 0.24242424242424243, 0.25252525252525254, 0.26262626262626265, 0.27272727272727276, 0.2828282828282829, 0.29292929292929293, 0.30303030303030304, 0.31313131313131315, 0.32323232323232326, 0.33333333333333337, 0.3434343434343435, 0.3535353535353536, 0.36363636363636365, 0.37373737373737376, 0.38383838383838387, 0.393939393939394, 0.4040404040404041, 0.4141414141414142, 0.42424242424242425, 0.43434343434343436, 0.4444444444444445, 0.4545454545454546, 0.4646464646464647, 0.4747474747474748, 0.48484848484848486, 0.494949494949495, 0.5050505050505051, 0.5151515151515152, 0.5252525252525253, 0.5353535353535354, 0.5454545454545455, 0.5555555555555556, 0.5656565656565657, 0.5757575757575758, 0.5858585858585859, 0.595959595959596, 0.6060606060606061, 0.6161616161616162, 0.6262626262626263, 0.6363636363636365, 0.6464646464646465, 0.6565656565656566, 0.6666666666666667, 0.6767676767676768, 0.686868686868687, 0.696969696969697, 0.7070707070707072, 0.7171717171717172, 0.7272727272727273, 0.7373737373737375, 0.7474747474747475, 0.7575757575757577, 0.7676767676767677, 0.7777777777777778, 0.787878787878788, 0.797979797979798, 0.8080808080808082, 0.8181818181818182, 0.8282828282828284, 0.8383838383838385, 0.8484848484848485, 0.8585858585858587, 0.8686868686868687, 0.8787878787878789, 0.888888888888889, 0.8989898989898991, 0.9090909090909092, 0.9191919191919192, 0.9292929292929294, 0.9393939393939394, 0.9494949494949496, 0.9595959595959597, 0.9696969696969697, 0.9797979797979799, 0.98989898989899, 1.0]
b50Ny = [0.16, 0.16, 0.16, 0.28, 0.28, 0.28, 0.28, 0.3, 0.3, 0.32, 0.32, 0.32, 0.32, 0.4, 0.4, 0.44, 0.44, 0.46, 0.46, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.56, 0.56, 0.56, 0.56, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.64, 0.64, 0.66, 0.66, 0.7, 0.7, 0.7, 0.7, 0.78, 0.78, 0.82, 0.82, 0.84, 0.84, 0.84, 0.84, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 1.0]



def graph_recorded_rocs():
    labels = ["BLOSUM50","BLOSUM62","PAM100","PAM250","MATIO"]
    xs = [b50x,b62x,pam100x,pam250x,matx]
    ys = [b50y,b62y,pam100y,pam250y,maty]
    graph_rocs(xs,ys,labels,title="ROC by Matrix")

def graph_normalized_roc_comparison():
    labels = ["Raw Scores","Normalized by Length"]
    xs = [b50x,b50Nx]
    ys = [b50y,b50Ny]
    graph_rocs(xs,ys,labels,title="BLOSUM50 ROC for Raw vs Normalized Scores")
