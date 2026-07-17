"""
tests/test_normalization.py
============================
Tests unitaires pour les fonctions de normalisation (normalize/denormalize).

Vérifie le mapping correct entre [min, max] et [-1, 1],
le clipping, la casse min==max, et la propriété de roundtrip.
"""

import sys
import os
import pytest

# Ajouter le répertoire environment/ au path pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.normalization import normalize, denormalize


# ─── Tests de base pour normalize() ────────────────────────────────

class TestNormalize:
    """Tests pour la fonction normalize()."""

    def test_normalize_midpoint(self):
        """Le milieu de l'intervalle doit être normalisé à 0.0."""
        assert normalize(5.0, 0.0, 10.0) == pytest.approx(0.0)

    def test_normalize_min(self):
        """La borne inférieure doit être normalisée à -1.0."""
        assert normalize(0.0, 0.0, 10.0) == pytest.approx(-1.0)

    def test_normalize_max(self):
        """La borne supérieure doit être normalisée à 1.0."""
        assert normalize(10.0, 0.0, 10.0) == pytest.approx(1.0)

    def test_normalize_below_min(self):
        """Une valeur en dessous de min doit être clipée à -1.0."""
        assert normalize(-5.0, 0.0, 10.0) == pytest.approx(-1.0)

    def test_normalize_above_max(self):
        """Une valeur au dessus de max doit être clipée à 1.0."""
        assert normalize(20.0, 0.0, 10.0) == pytest.approx(1.0)

    def test_normalize_negative_range(self):
        """Teste la normalisation avec un intervalle négatif [-5, 5]."""
        assert normalize(-5.0, -5.0, 5.0) == pytest.approx(-1.0)
        assert normalize(0.0, -5.0, 5.0) == pytest.approx(0.0)
        assert normalize(5.0, -5.0, 5.0) == pytest.approx(1.0)

    def test_normalize_zero_range(self):
        """Si min == max, retourne 0.0 (évitement division par zéro)."""
        assert normalize(5.0, 5.0, 5.0) == pytest.approx(0.0)


# ─── Tests de base pour denormalize() ──────────────────────────────

class TestDenormalize:
    """Tests pour la fonction denormalize()."""

    def test_denormalize_midpoint(self):
        """denormalize(0) doit retourner le milieu de l'intervalle."""
        assert denormalize(0.0, 0.0, 10.0) == pytest.approx(5.0)

    def test_denormalize_min(self):
        """denormalize(-1) doit retourner la borne inférieure."""
        assert denormalize(-1.0, 0.0, 10.0) == pytest.approx(0.0)

    def test_denormalize_max(self):
        """denormalize(1) doit retourner la borne supérieure."""
        assert denormalize(1.0, 0.0, 10.0) == pytest.approx(10.0)

    def test_denormalize_negative_range(self):
        """Teste la dénormalisation avec un intervalle négatif."""
        assert denormalize(0.0, -5.0, 5.0) == pytest.approx(0.0)
        assert denormalize(-1.0, -5.0, 5.0) == pytest.approx(-5.0)
        assert denormalize(1.0, -5.0, 5.0) == pytest.approx(5.0)


# ─── Tests de roundtrip ───────────────────────────────────────────

class TestRoundtrip:
    """Tests de roundtrip normalize/denormalize."""

    def test_normalize_roundtrip_inside_range(self):
        """denormalize(normalize(x)) == x pour x dans [min, max]."""
        for x in [0.0, 2.5, 5.0, 7.5, 10.0]:
            normalized = normalize(x, 0.0, 10.0)
            denormalized = denormalize(normalized, 0.0, 10.0)
            assert denormalized == pytest.approx(x)

    def test_normalize_roundtrip_negative_range(self):
        """Roundtrip fonctionne avec un intervalle négatif."""
        for x in [-5.0, -2.5, 0.0, 2.5, 5.0]:
            normalized = normalize(x, -5.0, 5.0)
            denormalized = denormalize(normalized, -5.0, 5.0)
            assert denormalized == pytest.approx(x)

    def test_normalize_roundtrip_clipped(self):
        """Pour les valeurs hors range, le roundtrip est limité par le clipping."""
        # Au-delà de max -> clip à 1.0 -> denormalize(1) = max
        assert denormalize(normalize(20.0, 0.0, 10.0), 0.0, 10.0) == pytest.approx(10.0)
        # En dessous de min -> clip à -1.0 -> denormalize(-1) = min
        assert denormalize(normalize(-10.0, 0.0, 10.0), 0.0, 10.0) == pytest.approx(0.0)
