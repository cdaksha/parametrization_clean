
# Standard library

# 3rd party packages

# Local source
from domain.selection.factory import SelectionFactory
from domain.selection.tournament import TournamentSelect


def test_get_tournament():
    assert SelectionFactory.create_executor('tournament') == TournamentSelect
