
# Standard library
from unittest.mock import patch

# 3rd party packages
import pytest

# Local source
from parametrization_clean.domain.selection.tournament import TournamentSelect


@patch('random.sample')
@pytest.mark.usefixtures('get_individuals')
def test_tournament(sample_mock, get_individuals):
    get_individuals[0].cost = 5142
    get_individuals[1].cost = 4387
    get_individuals[2].cost = 5789
    get_individuals[3].cost = 4698
    sample_mock.return_value = get_individuals[0], get_individuals[2]
    assert TournamentSelect.selection(get_individuals) == get_individuals[0]
    sample_mock.return_value = get_individuals
    assert TournamentSelect.selection(get_individuals, tournament_size=4) == get_individuals[1]
