
# Standard library

# 3rd party packages

# Local source
from domain.selection.factory import selection_factory
from domain.selection.tournament import TournamentSelect


def test_get_tournament():
    assert selection_factory('tournament') == TournamentSelect
