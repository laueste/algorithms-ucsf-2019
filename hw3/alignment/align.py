# implementation of smith-waterman sequence alignment
import numpy as np

# Initialize Alignment Matrix
def initialize(query,target):
    """
    Initializes the alignment matrix for the two input sequences. Query always
    goes across the top (columns), target goes vertically along the left (rows)

    Input: two strings to be aligned
    Output: the alignment matrix to operate on, of dimensions
            (query-length+1)rows x (target-length+1)cols, filled with
            2-tuples of zeros (0,0) to hold (score,traceback_direction)
    """
    # add 1 length for the gap option
    return np.zeros((len(query)+1,len(target)+1),dtype=(int,2))

## ASCII EXAMPLE OF AN ALIGNMENT MATRIX
#  T-> j  0 1 2 3 4     <- char index aka row/col indices
#         * A A T G     <- target sequence (starts with gap = *)
#  Q  0 * 0 0 0 0 0     <- this 0th row and col stay 0
#  |  1 A 0 5 5 0 2
#  V  2 T 0 0 5 9 5
#  i  3 G 0 0 1 5 18    <- maximum score=18, start traceback here

[0, 0, 0, 0, 0],
[0, 2, 2, 4, 4],
[0, 4, 0, 0, 2],
[0, 4, 1, 1, 0]

# Compute Penalty Score for any Specific Gap
def compute_gap_penalty(gapSize,startChar,score_matrix):
    """
    Computes the score for a gap of gapSize and starting with startChar.
    """
    # one penalty value to open the gap, another value to continue the gap
    #return score_matrix[(startChar,'*')] + (gapSize)*score_matrix[('*','*')]
    # NOTE: had I been interpreting this wrong all along??? previously used ^^
    # but I think actually we aren't supposed to be able to get gap penalties
    # from the matrix itself?? I will store gap extension penalty in */*...
    return score_matrix[(startChar,'*')] - (gapSize)*score_matrix[('*','*')]


## NOTE on computing scores:
##    Since SM algorithm restricts the lowest value for any cell to zero,
##    the two gap computation fns return at lowest -1, because
##    once their scores are below 0, they will not be the option chosen
##    for the overall value of the cell.

# Compute the three score possibilities (match, gap in Q, gap in T)
def compute_match_score(row,col,queryChar,targetChar,align_matrix,score_matrix):
    """
    Computes H_i,j + S(Q,T),
    the score of matching the current query and target chars plus the score of
    the best alignment of all preceeding characters once the current two chars
    are paired.
    Input: row and col (ints), query and target chars (string),
           align & score matrices
    Output: the score as an integer
    """
    current_score = score_matrix[(queryChar,targetChar)]
    prev_score = align_matrix[row-1][col-1][0] #score is 1rst item in cell tuple
    # we get to assume that this is always filled in, because we initialize the
    # 0th row and column and always fill L->R top->bottom
    return current_score+prev_score

def compute_query_gap_score(row,col,queryChar,targetChar,align_matrix,score_matrix):
    """
    Computes max( H_(i-k),j + gap_penalty*k, 0<k<=i ),
    the cost of a gap in the query sequence matched with the current char in
    the target sequence. No cost for initial gaps.
    Input: row and col (ints), query and target chars (string),
           align & score matrices
    Output: the score as an integer
    """
    ## TODO: Does the gp have to change its startChar every lookback?? maybe...!
    max_score = -1
    if col == 1: #then this is the first char, so just an initial gap
        return score_matrix[(queryChar,targetChar)]
    for k in range(1,row):
        gp = compute_gap_penalty(k,queryChar,score_matrix)
        max_score = max(max_score, gp+align_matrix[row-k][col][0])
    return max_score

def compute_target_gap_score(row,col,queryChar,targetChar,align_matrix,score_matrix):
    """
    Computes max( H_i,(j-k) + gap_penalty*k, 0<k<=i ),
    the cost of a gap in the target sequence matched with the current char in
    the query sequence. No cost for initial gaps.
    Input: row and col (ints), query and target chars (string),
           align & score matrices
    Output: the score as an integer
    """
    max_score = -1
    if row == 1: #then this is the first char, so just an initial gap
        return score_matrix[(queryChar,targetChar)]
    for k in range(1,col):
        gp = compute_gap_penalty(k,targetChar,score_matrix)
        max_score = max(max_score, gp+align_matrix[row][col-k][0])
    return max_score


# Create Complete Alignment Matrix
def make_align_matrix(query_seq,target_seq,score_matrix):
    """
    Fills in an alignment matrix for the given query and target sequences,
    according to the input score matrix.

    Input: query and target sequences (str), score matrix (dictionary of tuples)
    Output: an alignment matrix filled in according to the Smith-Waterman
            algorithm, returned as a 2D array of tuples
    """
    align_matrix = initialize(query_seq,target_seq)
    global_max_score = 0 #record start for traceback to save search cost later
    global_max_score_coordinates = (0,0)


    # iterate through query and target starting at 1 not 0, since the 0th row
    # and col are for the starting gap/start of the sequence. Starting gaps are
    # not penalized and the 0 score is used in traceback to know when to end,
    # so leave the 0th row and column values at their initialized values of 0.
    for r,q in enumerate(query_seq,start=1):
        for c,t in enumerate(target_seq,start=1):

            # Directions Key:
            # the second element of each tuple in each cell of align_matrix
            # is the traceback direction, used in the final step.`
            # 0 = match, go back diagonally \
            # 1 = gap in query, go back left --
            # 2 = gap in target, go back up |
            # 4 = this is the default zero, do not go anywhere (will win ties)

            match_score = (compute_match_score(r,c,q,t,align_matrix,score_matrix), 0)
            query_gap_score = (compute_query_gap_score(r,c,q,t,align_matrix,score_matrix), 1)
            target_gap_score = (compute_target_gap_score(r,c,q,t,align_matrix,score_matrix), 2)

            max_score = max((0,4),match_score,query_gap_score,target_gap_score,
                                key=lambda n:n[0])

            # update goal state record
            if max_score[0] > global_max_score:
                global_max_score = max_score[0]
                global_max_score_coordinates = (r,c)

            align_matrix[r][c] = max_score

    return align_matrix,global_max_score,global_max_score_coordinates


def print_alignment(query,target):
    """
    Pretty-prints out two aligned sequences, target on top and query below
    Input: query and target alignment sequences as strings
    Output: a 3-line string displaying the alignment. Ex:
        A A T G              C G A T G
          | | |      or      |   | | |
        - A T G              C - A T G

    """
    bridge = ''
    longer_seq = ''
    shorter_seq = ''
    longer = query if len(query) > len(target) else target
    shorter = target if query == longer else query
    for i in range(1,len(longer)+1):
        if i > len(shorter) or shorter[-i] == '*':
            shorter_seq = '- '+shorter_seq
            longer_seq = longer[-i]+' '+longer_seq
            bridge = '  '+bridge
        elif longer[-i] == '*':
            longer_seq = '- '+longer_seq
            shorter_seq = shorter[-i]+' '+shorter_seq
            bridge = '  '+bridge
        elif shorter[-i] == longer [-i]:
            shorter_seq = shorter[-i]+' '+shorter_seq
            longer_seq = longer[-i]+' '+longer_seq
            bridge = '| '+bridge
        else:
            shorter_seq = shorter[-i]+' '+shorter_seq
            longer_seq = longer[-i]+' '+longer_seq
            bridge = '  '+bridge
    query_seq = shorter_seq if shorter == query else longer_seq
    target_seq = longer_seq if longer == target else shorter_seq
    #yes there's an extra space at the end, but I'm fine with that
    return "\n".join([target_seq,bridge,query_seq])

# Decode a completely filled-in alignment matrix to return
# the sequence and the score of the best alignment
def traceback(query,target,complete_align_matrix,max_score=0,max_coordinates=()):
    """
    Trace back from highest scoring cell to a cell with a score of 0,
    recording the optimal sequence this path represents + its total score.
    Optionally can pass in the score and location of the highest scoring cell
    if it was recorded during the matrix-filling process.

    Input:  filled-in alignment matrix (2D array of tuples, (score,direction),
            that correspond to the Smith-Waterman alignment calculations for the
            query and target sequences of that matrix. Optionally, the score
            coordinates (as an int and tuple) of the highest-value cell
    Output: the sequence and the score of the optimal alignment from the input
            matrix
    """
    if max_coordinates: #if starting location not passed in, find it.
        for r in range(len(complete_align_matrix)):
            for c in range(len(complete_align_matrix[0])):
                if complete_align_matrix[r][c][0] > max_score:
                    max_score = complete_align_matrix[r][c][0]
                    max_coordinates = (r,c)

    row,col = max_coordinates
    current_cell = complete_align_matrix[row][col]
    score,direction = current_cell

    # Set variables for iteration
    query_align = ''
    target_align = ''
    queryChar = query[row-1]
    targetChar = target[col-1]

    # Walk path back through alignment matrix to a cell with score value 0
    timeout_counter = len(query)*len(target)+5 # to avoid endless loops
    while score > 0 and timeout_counter > 0:
        timeout_counter += -1
        # Build up the optimal alignment sequence, backwards
        #matrix has queryLen+1 rows and targetLen+1 cols, since both start at *
        target_align = targetChar + target_align
        query_align = queryChar + query_align
        # Determine which direction to go and then move
        if direction == 0: #match
            row,col = row-1,col-1
            queryChar = query[max(0,row-1)]
            targetChar = target[max(0,col-1)]
        if direction == 1: #gap in query
            row,col = row,col-1
            queryChar = '*'
            targetChar = target[max(0,col-1)]
        if direction == 2: #gap in target
            row,col = row-1,col
            queryChar = query[max(0,row-1)]
            targetChar = '*'
        score,direction = complete_align_matrix[row][col]

    # quick error handling
    if timeout_counter == 0:
        print("Row, Col, Location, Direction:",row,col,current_cell,direction)
        print("Target:",target)
        print("Query:",query)
        print("Query Alignment, Target Alignment:",query_align,target_align)
        raise RuntimeError("Traceback timed out, with state values shown above")

    # otherwise, return the completed alignments
    return query_align,target_align,max_score

# Algorithm Main Function
def align(query,target,score_matrix,do_traceback=True):
    """
    Aligns two sequences via the Smith-Waterman algorithm, using the input
    scoring matrix to evaluate costs

    Input: query and target sequences as strings, a score matrix as a dictionary
           of tuples.
    Output: ASCII alignment depiction + alignment score
    """
    #quick check to ensure that we have sequences to operate on
    if len(query) < 1 or len(target) < 1:
        raise ValueError("Query and target sequences cannot be empty.\
         Received %s query and %s target." % (query,target))

    # Complete the alignment matrix
    matrix,score,best_cell = make_align_matrix(query,target,score_matrix)

    if do_traceback == False: # shortcut for analyses that only want score
        return score

    # Traceback to find alignments
    q_seq,t_seq,score = traceback(query,target,matrix,score,best_cell)
    return q_seq,t_seq,score


def rescore_alignment(query_align,target_align,score_matrix):
    """
    Re-scores a static input alignment (query and target sequences) based on
    the input scoring matrix. Assumes no starting gap penalty.
    Input: query and target alignment sequences (post initial alignment),
           scoring matrix to rescore with
    Output: alignment score (int) according to the input score matrix
    """
    score = 0
    shorter = query_align if len(query_align) < len(target_align) else target_align
    longer = target_align if shorter == query_align else query_align
    current_gap_s = 0
    current_gap_l = 0
    for i in range(0,len(shorter)+1): #go backwards b/c same way score was initially made
        sChar,lChar = shorter[-i],longer[-i]
        if current_gap_s > 0: #if currently in a gap in shorter
            if sChar == '*': #gap continues
                current_gap_s += 1
            else: # gap has ended, add gap extension penalties
                score += -1*current_gap_s*score_matrix[('*','*')]
                current_gap_s = 0 #reset to non-gap state
        elif current_gap_l > 0: #if currently in a gap in longer
            if lChar == '*':
                current_gap_l += 1
            else: # gap has ended, add gap extension penalties
                score += -1*current_gap_l*score_matrix[('*','*')]
                current_gap_l = 0 #reset to non-gap state
        else: # not currently in a gap
            if sChar == '*': #we have just then started a gap in shorter
                score += score_matrix[(sChar,'*')] #add gap opening penalty
                current_gap_s += 1
            elif lChar == '*': #we have just then started a gap in longer
                score += score_matrix[(sChar,'*')] #add gap opening penalty
                current_gap_l += 1
            else:
                score += max(0,score_matrix[(sChar,lChar)])
        #print(sChar,lChar,current_gap_s,current_gap_l,score)
    #print([ score_matrix[(shorter[i],longer[i])] for i in range(len(shorter)) ])
    return score
