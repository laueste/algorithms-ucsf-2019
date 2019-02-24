from alignment import align,io
import pytest

test_matrix = { #TODO implementation now needs both (N,M) and (M,N) to be keys
('A','A'):5,('T','T'):5,('G','G'):8,
('A','T'):0,('T','A'):0,
('G','T'):-2,('T','G'):-2,
('A','G'):0,('G','A'):0,
('A','*'):-5,('G','*'):-5,('T','*'):-5,('*','A'):-5,('*','T'):-5,('*','G'):-5,
('*', '*'):1
}

@pytest.mark.parametrize("query,target,matrix", [("ATG","AATG",test_matrix)])

def test_matrix_building(query, target, matrix):
    alignment_scores_only = [ [0, 0, 0, 0, 0],
                              [0, 5, 5, 0, 0],
                              [0, 0, 5, 10, 4],
                              [0, 0, 0, 4, 18] ]
    direction_scores_only = [ [0, 0, 0, 0, 0],
                              [0, 0, 0, 4, 4],
                              [0, 4, 0, 0, 2],
                              [0, 4, 4, 1, 0] ]
    output_matrix,score,coords = align.make_align_matrix(query,target,matrix)
    assert len(output_matrix) == len(query)+1
    assert len(output_matrix[0]) == len(target)+1
    assert alignment_scores_only == [[e[0] for e in r] for r in output_matrix]
    assert direction_scores_only  == [[e[1] for e in r] for r in output_matrix]
    assert score == 18
    assert coords == (3,4)

@pytest.mark.parametrize("q,t,answer",
[("*ATG","AATG","A A T G \n  | | | \n- A T G "),
('AACTATGG','C*ATG*','- - C - A T G - \n    |   | | |   \nA A C T A T G G ')])

def test_printing(q,t,answer):
    assert answer == align.print_alignment(q,t)

## TODO TESTS THAT I WANT TO IMPLEMENT BUT MAY RUN OUT OF TIME FOR
def test_one_char_seq_match():
    # test 'A' vs 'A', real score is 5
    assert True == True

def test_one_char_seq_mismatch():
    # test 'A' vs 'T', real score is 0
    assert True == True

def test_aligned_length():
    # make sure that the aligned sequences are the same length
    assert True == True

def test_traceback_scoring():
    # test that the reported score corresponds to scoring the reported alignments
    # using rescore_alignment
    assert True == True

def test_traceback():
    # make sure that the alignment fits what we expect for the toy example
    assert True == True

# @pytest.mark.parametrize("file1,file2,matrixFile",
# [("./sequences/prot-0004.fa","./sequences/prot-0008.fa","./matrices/BLOSUM50")])
# # NOTE: THIS DOES NOT CURRENTLY PASS! SEE WRITEUP PART II
# def test_rescore(file1,file2,matrixFile):
#     score_matrix = io.read_scoring_matrix(matrixFile)
#     query = io.read_sequence_fasta(file1)
#     target = io.read_sequence_fasta(file2)
#     results = align.align(query.sequence,target.sequence,score_matrix)
#     query_align,target_align,score = results
#     assert score == align.rescore_alignment(query_align,target_align,score_matrix)
