from alignment import optimization
import pytest

test_matrix = {
('A','A'):5,('T','T'):5,('G','G'):8,
('A','T'):0,('T','A'):0,
('G','T'):-2,('T','G'):-2,
('A','G'):0,('G','A'):0
}

@pytest.mark.parametrize( "matrix,changes", [(test_matrix,[(('A','A'),10)]),
(test_matrix,[(('A','T'),3)]),(test_matrix,[(('A','T'),1),(('G','G'),12)])] )

def test_matrix_changing(matrix,changes):
    new_matrix,new_changes = optimization.apply_matrix_changes(matrix,changes)
    for pair,score in changes:
        assert new_matrix[pair] == score
        assert new_matrix[pair[::-1]] == score #ensure matrix symmetry
    assert changes == new_changes


def test_matrix_symmetry():
    return
