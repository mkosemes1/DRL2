"""
tests/test_field_cell.py
========================
Tests unitaires pour la classe FieldCell.

Vérifie les valeurs par défaut et la mutabilité des attributs
de la classe FieldCell utilisée pour la grille du champ agricole.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agri_drone_env import FieldCell


# ─── Tests Valeurs par Défaut ─────────────────────────────────────

class TestFieldCellDefaults:
    """Vérifie que tous les attributs de FieldCell ont les bonnes valeurs par défaut."""

    def test_field_cell_defaults_healthy(self):
        """Vérifie que healthy est True par défaut."""
        cell = FieldCell()
        assert cell.healthy is True

    def test_field_cell_defaults_wet(self):
        """Vérifie que wet est True par défaut."""
        cell = FieldCell()
        assert cell.wet is True

    def test_field_cell_defaults_sprayed(self):
        """Vérifie que sprayed est False par défaut."""
        cell = FieldCell()
        assert cell.sprayed is False

    def test_field_cell_defaults_watered(self):
        """Vérifie que watered est False par défaut."""
        cell = FieldCell()
        assert cell.watered is False

    def test_field_cell_defaults_visited(self):
        """Vérifie que visited est False par défaut."""
        cell = FieldCell()
        assert cell.visited is False

    def test_field_cell_all_defaults_combined(self):
        """Vérifie que tous les attributs par défaut sont corrects en une seule assertion."""
        cell = FieldCell()
        assert (cell.healthy, cell.wet, cell.sprayed, cell.watered, cell.visited) == (
            True, True, False, False, False
        )


# ─── Tests Mutabilité ─────────────────────────────────────────────

class TestFieldCellAttributesCanBeSet:
    """Vérifie que tous les attributs de FieldCell sont modifiables."""

    def test_field_cell_set_healthy(self):
        """Vérifie que l'attribut healthy peut être défini à False."""
        cell = FieldCell()
        cell.healthy = False
        assert cell.healthy is False

    def test_field_cell_set_wet(self):
        """Vérifie que l'attribut wet peut être défini à False."""
        cell = FieldCell()
        cell.wet = False
        assert cell.wet is False

    def test_field_cell_set_sprayed(self):
        """Vérifie que l'attribut sprayed peut être défini à True."""
        cell = FieldCell()
        cell.sprayed = True
        assert cell.sprayed is True

    def test_field_cell_set_watered(self):
        """Vérifie que l'attribut watered peut être défini à True."""
        cell = FieldCell()
        cell.watered = True
        assert cell.watered is True

    def test_field_cell_set_visited(self):
        """Vérifie que l'attribut visited peut être défini à True."""
        cell = FieldCell()
        cell.visited = True
        assert cell.visited is True

    def test_field_cell_independent_instances(self):
        """Vérifie que deux instances de FieldCell sont indépendantes."""
        cell1 = FieldCell()
        cell2 = FieldCell()
        cell1.healthy = False
        cell1.sprayed = True
        cell1.visited = True
        # cell2 ne doit pas être affecté par les modifications de cell1
        assert cell2.healthy is True
        assert cell2.sprayed is False
        assert cell2.visited is False
