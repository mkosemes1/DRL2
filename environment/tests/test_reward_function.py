"""
tests/test_reward_function.py
==============================
Tests unitaires pour RewardCalculator et RewardConfig.

Vérifie la fonction de récompense composite pour la tâche d'irrigation
(water task) ainsi que les méthodes compute() et compute_agri().

Cas testés :
  - Arrosage et remplissage du réservoir
  - Pénalité temporelle proportionnelle aux groupes non arrosés
  - Shaping de distance (rapprochement / éloignement)
  - Mission accomplie
  - Somme totale et cohérence des termes
  - Retour tuple et clés attendues
  - Garde d'infini pour le distance shaping
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reward.reward_function import RewardCalculator, RewardConfig


# ─── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def reward_config():
    """Configuration par défaut de la récompense."""
    return RewardConfig()


@pytest.fixture
def calculator(reward_config):
    """Calculateur de récompense initialisé."""
    return RewardCalculator(reward_config)


# ─── Tests compute_water_task ────────────────────────────────────

class TestComputeWaterTask:
    """Tests pour la méthode compute_water_task()."""

    def test_watering_reward(self, calculator):
        """Vérifie que just_watered=True produit terms['watering'] == 5.0."""
        _, terms = calculator.compute_water_task(
            tank_level=100.0,
            prev_dist=10.0,
            curr_dist=8.0,
            just_watered=True,
            just_refilled=False,
            all_watered=False,
            num_unwatered=3,
        )
        assert terms["watering"] == 5.0

    def test_refill_reward(self, calculator):
        """Vérifie que just_refilled=True produit terms['refill'] == 1.0."""
        _, terms = calculator.compute_water_task(
            tank_level=100.0,
            prev_dist=10.0,
            curr_dist=8.0,
            just_watered=False,
            just_refilled=True,
            all_watered=False,
            num_unwatered=3,
        )
        assert terms["refill"] == 1.0

    def test_refill_no_reward_at_high_level(self, calculator):
        """Vérifie que la récompense de remplissage est donnée même si tank_level >= 98.

        Le seuil de vérification est dans l'environnement, pas dans
        le calculateur de récompense.
        """
        _, terms = calculator.compute_water_task(
            tank_level=99.0,
            prev_dist=10.0,
            curr_dist=8.0,
            just_watered=False,
            just_refilled=True,
            all_watered=False,
            num_unwatered=3,
        )
        assert terms["refill"] == 1.0

    def test_time_penalty(self, calculator):
        """Vérifie que time_penalty == -0.02 * num_unwatered."""
        num_unwatered = 5
        _, terms = calculator.compute_water_task(
            tank_level=100.0,
            prev_dist=10.0,
            curr_dist=8.0,
            just_watered=False,
            just_refilled=False,
            all_watered=False,
            num_unwatered=num_unwatered,
        )
        assert terms["time_penalty"] == pytest.approx(-0.02 * num_unwatered)

    def test_distance_shaping_closer(self, calculator):
        """Vérifie distance_shaping == 0.05 quand le drone se rapproche."""
        _, terms = calculator.compute_water_task(
            tank_level=100.0,
            prev_dist=10.0,
            curr_dist=5.0,
            just_watered=False,
            just_refilled=False,
            all_watered=False,
            num_unwatered=2,
        )
        assert terms["distance_shaping"] == pytest.approx(0.05)

    def test_distance_shaping_further(self, calculator):
        """Vérifie distance_shaping == 0.0 quand le drone s'éloigne (curr >= prev)."""
        _, terms = calculator.compute_water_task(
            tank_level=100.0,
            prev_dist=5.0,
            curr_dist=10.0,
            just_watered=False,
            just_refilled=False,
            all_watered=False,
            num_unwatered=2,
        )
        assert terms["distance_shaping"] == 0.0

    def test_distance_shaping_equal(self, calculator):
        """Vérifie distance_shaping == 0.0 quand les distances sont égales."""
        _, terms = calculator.compute_water_task(
            tank_level=100.0,
            prev_dist=7.0,
            curr_dist=7.0,
            just_watered=False,
            just_refilled=False,
            all_watered=False,
            num_unwatered=2,
        )
        assert terms["distance_shaping"] == 0.0

    def test_distance_shaping_all_watered(self, calculator):
        """Vérifie distance_shaping == 0.0 quand tous les groupes sont arrosés."""
        _, terms = calculator.compute_water_task(
            tank_level=100.0,
            prev_dist=0.0,
            curr_dist=0.0,
            just_watered=False,
            just_refilled=False,
            all_watered=True,
            num_unwatered=0,
        )
        assert terms["distance_shaping"] == 0.0

    def test_mission_complete(self, calculator):
        """Vérifie mission_complete == 100.0 quand all_watered=True."""
        _, terms = calculator.compute_water_task(
            tank_level=50.0,
            prev_dist=0.0,
            curr_dist=0.0,
            just_watered=False,
            just_refilled=False,
            all_watered=True,
            num_unwatered=0,
        )
        assert terms["mission_complete"] == 100.0

    def test_mission_not_complete(self, calculator):
        """Vérifie mission_complete == 0.0 quand all_watered=False."""
        _, terms = calculator.compute_water_task(
            tank_level=50.0,
            prev_dist=10.0,
            curr_dist=8.0,
            just_watered=False,
            just_refilled=False,
            all_watered=False,
            num_unwatered=2,
        )
        assert terms["mission_complete"] == 0.0

    def test_total_reward_sum(self, calculator):
        """Vérifie que la récompense totale est la somme de tous les termes."""
        total, terms = calculator.compute_water_task(
            tank_level=80.0,
            prev_dist=12.0,
            curr_dist=9.0,
            just_watered=True,
            just_refilled=True,
            all_watered=False,
            num_unwatered=3,
        )
        expected_total = sum(terms.values())
        assert total == pytest.approx(expected_total)

    def test_return_type(self, calculator):
        """Vérifie que le retour est un tuple (float, dict)."""
        result = calculator.compute_water_task(
            tank_level=100.0,
            prev_dist=10.0,
            curr_dist=8.0,
            just_watered=False,
            just_refilled=False,
            all_watered=False,
            num_unwatered=1,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], dict)

    def test_all_terms_present(self, calculator):
        """Vérifie que le dict terms contient exactement les 5 clés attendues."""
        _, terms = calculator.compute_water_task(
            tank_level=100.0,
            prev_dist=10.0,
            curr_dist=8.0,
            just_watered=True,
            just_refilled=True,
            all_watered=True,
            num_unwatered=0,
        )
        expected_keys = {"watering", "refill", "time_penalty", "distance_shaping", "mission_complete"}
        assert set(terms.keys()) == expected_keys

    def test_no_watering_no_refill(self, calculator):
        """Vérifie que quand tous les flags sont False, seul time_penalty est non nul
        (et éventuellement distance_shaping)."""
        _, terms = calculator.compute_water_task(
            tank_level=100.0,
            prev_dist=10.0,
            curr_dist=8.0,
            just_watered=False,
            just_refilled=False,
            all_watered=False,
            num_unwatered=3,
        )
        # Les termes qui doivent être nuls
        assert terms["watering"] == 0.0
        assert terms["refill"] == 0.0
        assert terms["mission_complete"] == 0.0
        # La pénalité temporelle est toujours non nulle si num_unwatered > 0
        assert terms["time_penalty"] != 0.0
        # Le distance_shaping peut être 0.05 si le drone se rapproche
        assert terms["distance_shaping"] in (0.0, 0.05)

    def test_no_watering_no_refill_zero_unwatered(self, calculator):
        """Vérifie que quand num_unwatered=0 et tous les flags False, tous les termes sont nuls."""
        _, terms = calculator.compute_water_task(
            tank_level=100.0,
            prev_dist=0.0,
            curr_dist=0.0,
            just_watered=False,
            just_refilled=False,
            all_watered=False,
            num_unwatered=0,
        )
        assert terms["watering"] == 0.0
        assert terms["refill"] == 0.0
        assert terms["time_penalty"] == 0.0
        assert terms["distance_shaping"] == 0.0
        assert terms["mission_complete"] == 0.0

    def test_distance_shaping_infinity_guard_prev(self, calculator):
        """Vérifie que distance_shaping == 0.0 quand prev_dist == inf."""
        _, terms = calculator.compute_water_task(
            tank_level=100.0,
            prev_dist=float("inf"),
            curr_dist=5.0,
            just_watered=False,
            just_refilled=False,
            all_watered=False,
            num_unwatered=2,
        )
        assert terms["distance_shaping"] == 0.0

    def test_distance_shaping_infinity_guard_curr(self, calculator):
        """Vérifie que distance_shaping == 0.0 quand curr_dist == inf."""
        _, terms = calculator.compute_water_task(
            tank_level=100.0,
            prev_dist=10.0,
            curr_dist=float("inf"),
            just_watered=False,
            just_refilled=False,
            all_watered=False,
            num_unwatered=2,
        )
        assert terms["distance_shaping"] == 0.0

    def test_distance_shaping_both_infinity(self, calculator):
        """Vérifie que distance_shaping == 0.0 quand les deux distances sont inf."""
        _, terms = calculator.compute_water_task(
            tank_level=100.0,
            prev_dist=float("inf"),
            curr_dist=float("inf"),
            just_watered=False,
            just_refilled=False,
            all_watered=False,
            num_unwatered=2,
        )
        assert terms["distance_shaping"] == 0.0

    def test_all_rewards_combined(self, calculator):
        """Vérifie le calcul complet avec arrosage + remplissage + distance + mission."""
        total, terms = calculator.compute_water_task(
            tank_level=50.0,
            prev_dist=10.0,
            curr_dist=5.0,
            just_watered=True,
            just_refilled=True,
            all_watered=False,
            num_unwatered=3,
        )
        # watering=5.0, refill=1.0, time_penalty=-0.06, distance_shaping=0.05, mission=0.0
        assert terms["watering"] == 5.0
        assert terms["refill"] == 1.0
        assert terms["time_penalty"] == pytest.approx(-0.06)
        assert terms["distance_shaping"] == pytest.approx(0.05)
        assert terms["mission_complete"] == 0.0
        assert total == pytest.approx(5.0 + 1.0 + (-0.06) + 0.05 + 0.0)


# ─── Tests compute() ────────────────────────────────────────────

class TestCompute:
    """Tests pour la méthode compute() (récompense de navigation)."""

    def test_compute_returns_tuple(self, calculator):
        """Vérifie que compute() retourne un tuple (float, dict)."""
        result = calculator.compute(
            distance_old=10.0,
            distance_new=8.0,
            heading_error=0.5,
            action=np.zeros(4),
            angular_rates=np.zeros(3),
            collided=False,
            out_of_bounds=False,
            flipped=False,
            reached_goal=False,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], dict)

    def test_compute_total_equals_sum(self, calculator):
        """Vérifie que le total est la somme des termes dans compute()."""
        total, terms = calculator.compute(
            distance_old=10.0,
            distance_new=5.0,
            heading_error=0.3,
            action=np.array([0.5, 0.1, 0.0, 0.0]),
            angular_rates=np.array([0.1, 0.0, 0.0]),
            collided=False,
            out_of_bounds=False,
            flipped=False,
            reached_goal=False,
        )
        assert total == pytest.approx(sum(terms.values()))

    def test_compute_collision_penalty(self, calculator):
        """Vérifie que collided=True applique la pénalité de collision."""
        _, terms = calculator.compute(
            distance_old=10.0,
            distance_new=8.0,
            heading_error=0.0,
            action=np.zeros(4),
            angular_rates=np.zeros(3),
            collided=True,
            out_of_bounds=False,
            flipped=False,
            reached_goal=False,
        )
        assert terms["collision"] == -300.0

    def test_compute_goal_reward(self, calculator):
        """Vérifie que reached_goal=True produit un bonus de goal."""
        _, terms = calculator.compute(
            distance_old=1.0,
            distance_new=0.0,
            heading_error=0.0,
            action=np.zeros(4),
            angular_rates=np.zeros(3),
            collided=False,
            out_of_bounds=False,
            flipped=False,
            reached_goal=True,
        )
        assert terms["goal"] == 300.0


# ─── Tests compute_agri() ───────────────────────────────────────

class TestComputeAgri:
    """Tests pour la méthode compute_agri() (récompense agricole étendue)."""

    def test_compute_agri_returns_tuple(self, calculator):
        """Vérifie que compute_agri() retourne un tuple (float, dict)."""
        result = calculator.compute_agri(
            distance_old=10.0,
            distance_new=8.0,
            heading_error=0.5,
            action=np.zeros(4),
            angular_rates=np.zeros(3),
            collided=False,
            out_of_bounds=False,
            flipped=False,
            reached_goal=False,
            battery_level=500.0,
            maladies_traitees=5,
            sec_arrosees=3,
            total_malades=10,
            total_sec=8,
            pesticide_used=1.0,
            water_used=1.0,
            visited_percentage=0.5,
            returning_home=False,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], dict)

    def test_compute_agri_health_bonus(self, calculator):
        """Vérifie le bonus de santé proportionnel aux maladies traitées."""
        _, terms = calculator.compute_agri(
            distance_old=10.0,
            distance_new=8.0,
            heading_error=0.0,
            action=np.zeros(4),
            angular_rates=np.zeros(3),
            collided=False,
            out_of_bounds=False,
            flipped=False,
            reached_goal=False,
            battery_level=500.0,
            maladies_traitees=5,
            sec_arrosees=0,
            total_malades=10,
            total_sec=0,
            pesticide_used=0.0,
            water_used=0.0,
            visited_percentage=0.0,
            returning_home=False,
        )
        # 50% des maladies traitées → health = 2.0 * 0.5 = 1.0
        assert terms["health"] == pytest.approx(1.0)

    def test_compute_agri_low_battery_penalty(self, calculator):
        """Vérifie la pénalité batterie faible sans retour à la base."""
        _, terms = calculator.compute_agri(
            distance_old=10.0,
            distance_new=8.0,
            heading_error=0.0,
            action=np.zeros(4),
            angular_rates=np.zeros(3),
            collided=False,
            out_of_bounds=False,
            flipped=False,
            reached_goal=False,
            battery_level=5.0,
            maladies_traitees=0,
            sec_arrosees=0,
            total_malades=0,
            total_sec=0,
            pesticide_used=0.0,
            water_used=0.0,
            visited_percentage=0.0,
            returning_home=False,
        )
        assert terms["low_battery"] == -1.0

    def test_compute_agri_no_low_battery_when_returning(self, calculator):
        """Vérifie qu'il n'y a pas de pénalité batterie quand on retourne à la base."""
        _, terms = calculator.compute_agri(
            distance_old=10.0,
            distance_new=8.0,
            heading_error=0.0,
            action=np.zeros(4),
            angular_rates=np.zeros(3),
            collided=False,
            out_of_bounds=False,
            flipped=False,
            reached_goal=False,
            battery_level=5.0,
            maladies_traitees=0,
            sec_arrosees=0,
            total_malades=0,
            total_sec=0,
            pesticide_used=0.0,
            water_used=0.0,
            visited_percentage=0.0,
            returning_home=True,
        )
        assert terms["low_battery"] == 0.0
