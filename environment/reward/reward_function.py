"""

Fonction de récompense composite, entièrement paramétrable.

R = R_progress + R_goal + R_heading + R_smooth + R_energy
    + R_stability + R_time - R_collision - R_out - R_flip
    + (bonus agricoles : santé, irrigation, exploration, etc.)
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class RewardConfig:
    """Configuration paramétrable de la fonction de récompense composite.

    Regroupe les pondérations et seuils pour les trois familles de termes
    de récompense : navigation, agriculture et tâche d'irrigation (water task).

    Attributes:
        k_progress: Poids de la progression vers l'objectif.
        goal_reward: Récompense de fin d'épisode si l'objectif est atteint.
        goal_radius: Rayon de détection de l'objectif (en mètres).
        heading_weight: Poids de l'erreur de cap.
        smooth_weight: Poids du lissage des actions.
        energy_alpha: Coefficient de pénalité énergétique.
        stability_weight: Poids de la stabilité angulaire.
        time_penalty: Pénalité temporelle constante par pas.
        collision_penalty: Pénalité en cas de collision avec un obstacle.
        out_of_bounds_penalty: Pénalité en cas de sortie des limites.
        flip_penalty: Pénalité en cas de retournement du drone.
        health_bonus: Récompense par unité de pourcentage de maladies traitées.
        irrigation_bonus: Récompense par unité de pourcentage de zones arrosées.
        exploration_bonus: Récompense par unité de pourcentage de cartographie.
        waste_penalty: Pénalité pour gaspillage de pesticide ou d'eau.
        low_battery_penalty: Pénalité si batterie faible sans retour à la base.
        watering_reward: Récompense lors de l'arrosage d'un groupe de plantes.
        refill_reward: Récompense lors du remplissage du réservoir à la bassine.
        refill_threshold: Seuil de niveau du réservoir pour considérer un remplissage.
        time_penalty_per_group: Pénalité temporelle par groupe non arrosé.
        distance_shaping_reward: Récompense pour se rapprocher du groupe le plus proche.
        mission_complete_reward: Bonus de fin de mission quand tous les groupes sont arrosés.
    """
    # Termes de navigation
    k_progress: float = 5.0
    goal_reward: float = 300.0
    goal_radius: float = 0.5
    heading_weight: float = 0.5
    smooth_weight: float = 0.1
    energy_alpha: float = 0.05
    stability_weight: float = 0.3
    time_penalty: float = 0.01
    collision_penalty: float = 300.0
    out_of_bounds_penalty: float = 200.0
    flip_penalty: float = 250.0

    # Termes agricoles (nouveaux)
    health_bonus: float = 2.0          # par unité de pourcentage de maladies traitées
    irrigation_bonus: float = 2.0      # par unité de pourcentage de zones sèches arrosées
    exploration_bonus: float = 0.5     # par unité de pourcentage de cartographie
    waste_penalty: float = 0.5         # pénalité pour gaspillage de ressources
    low_battery_penalty: float = 1.0   # pénalité si batterie faible sans retour

    # --- Tâche d'irrigation (water task) ---
    watering_reward: float = 5.0           # récompense quand un groupe de plantes est arrosé
    refill_reward: float = 1.0             # récompense quand le réservoir est rempli
    refill_threshold: float = 98.0         # seuil pour considérer un remplissage
    time_penalty_per_group: float = 0.02   # pénalité temporelle par groupe non arrosé
    distance_shaping_reward: float = 0.05  # récompense pour se rapprocher d'un groupe
    mission_complete_reward: float = 100.0  # bonus quand tous les groupes sont arrosés


class RewardCalculator:
    """Calculateur de récompense composite pour l'environnement agricole.

    Fournit trois méthodes de calcul :
        - ``compute`` : récompense de navigation de base.
        - ``compute_agri`` : récompense étendue avec fonctions agricoles.
        - ``compute_water_task`` : récompense dédiée à la tâche d'irrigation
          avec gestion de réservoir d'eau et groupes de plantes.
    """

    def __init__(self, config: RewardConfig):
        """Initialise le calculateur de récompense.

        Args:
            config: Configuration des pondérations et seuils de récompense.
        """
        self.cfg = config
        self._prev_action = np.zeros(4)   # pour le lissage

    def reset(self):
        """Réinitialise la mémoire interne du calculateur de récompense.

        Remet à zéro le vecteur de l'action précédente utilisé pour le
        calcul du terme de lissage.
        """
        self._prev_action = np.zeros(4)

    # ------------------------------------------------------------------
    # Méthode originale (navigation simple, conservée pour compatibilité)
    def compute(
        self,
        distance_old: float,
        distance_new: float,
        heading_error: float,
        action: np.ndarray,
        angular_rates: np.ndarray,
        collided: bool,
        out_of_bounds: bool,
        flipped: bool,
        reached_goal: bool,
    ) -> tuple[float, dict]:
        """Calcule la récompense de navigation de base (sans fonctions agricoles).

        Combine les termes de progression, cap, lissage, énergie, stabilité,
        temps et pénalités terminales (collision, hors limites, retournement).

        Args:
            distance_old: Distance à l'objectif à l'étape précédente.
            distance_new: Distance à l'objectif à l'étape courante.
            heading_error: Erreur d'angle entre la direction courante et l'objectif (en radians).
            action: Action courante (tableau de forme ``(4,)``).
            angular_rates: Vitesses angulaires courantes (tableau de forme ``(3,)``).
            collided: True si une collision a eu lieu.
            out_of_bounds: True si le drone est hors des limites.
            flipped: True si le drone est retourné.
            reached_goal: True si l'objectif est atteint.

        Returns:
            Tuple ``(total_reward, terms_dict)`` où ``total_reward`` est la
            récompense totale (float) et ``terms_dict`` est un dictionnaire
            contenant la décomposition de chaque terme.
        """
        c = self.cfg
        terms = {}

        terms["progress"] = c.k_progress * (distance_old - distance_new)
        terms["goal"] = c.goal_reward if reached_goal else 0.0
        terms["heading"] = c.heading_weight * (1.0 - heading_error / np.pi) * 2.0 - c.heading_weight

        action_delta = np.linalg.norm(action - self._prev_action)
        terms["smooth"] = -c.smooth_weight * action_delta
        self._prev_action = action.copy()

        throttle_normalized = (action[0] + 1.0) / 2.0
        terms["energy"] = -c.energy_alpha * (throttle_normalized ** 2)
        terms["stability"] = -c.stability_weight * float(np.sum(np.square(angular_rates)))
        terms["time"] = -c.time_penalty
        terms["collision"] = -c.collision_penalty if collided else 0.0
        terms["out_of_bounds"] = -c.out_of_bounds_penalty if out_of_bounds else 0.0
        terms["flip"] = -c.flip_penalty if flipped else 0.0

        total = sum(terms.values())
        return float(total), terms

    # ------------------------------------------------------------------
    # Nouvelle méthode pour l'environnement agricole étendu
    def compute_agri(
        self,
        distance_old: float,
        distance_new: float,
        heading_error: float,
        action: np.ndarray,
        angular_rates: np.ndarray,
        collided: bool,
        out_of_bounds: bool,
        flipped: bool,
        reached_goal: bool,
        battery_level: float,
        maladies_traitees: int,
        sec_arrosees: int,
        total_malades: int,
        total_sec: int,
        pesticide_used: float,
        water_used: float,
        visited_percentage: float,
        returning_home: bool,
    ) -> tuple[float, dict]:
        """Calcule la récompense étendue pour les tâches agricoles.

        Ajoute aux termes de navigation les bonus de santé (maladies
        traitées), d'irrigation (zones sèches arrosées), d'exploration
        (cartographie), les pénalités de gaspillage de ressources et
        de batterie faible.

        Args:
            distance_old: Distance à l'objectif à l'étape précédente.
            distance_new: Distance à l'objectif à l'étape courante.
            heading_error: Erreur d'angle courante (en radians).
            action: Action courante (tableau de forme ``(4,)``).
            angular_rates: Vitesses angulaires courantes (tableau de forme ``(3,)``).
            collided: True si une collision a eu lieu.
            out_of_bounds: True si le drone est hors des limites.
            flipped: True si le drone est retourné.
            reached_goal: True si l'objectif est atteint.
            battery_level: Niveau de batterie restant (en Wh).
            maladies_traitees: Nombre de cellules malades pulvérisées.
            sec_arrosees: Nombre de cellules sèches arrosées.
            total_malades: Nombre total de cellules malades dans le champ.
            total_sec: Nombre total de cellules sèches dans le champ.
            pesticide_used: Quantité de pesticide utilisée.
            water_used: Quantité d'eau utilisée.
            visited_percentage: Pourcentage de cellules visitées (0.0–1.0).
            returning_home: True si le drone est en route vers la base.

        Returns:
            Tuple ``(total_reward, terms_dict)`` où ``total_reward`` est la
            récompense totale (float) et ``terms_dict`` est un dictionnaire
            contenant la décomposition de chaque terme.
        """
        c = self.cfg
        terms = {}

        # --- 1. Progression vers la cible (navigation) ---
        terms["progress"] = c.k_progress * (distance_old - distance_new)

        # --- 2. Bonus de santé (maladies traitées) ---
        if total_malades > 0:
            pct_maladies = maladies_traitees / total_malades
            terms["health"] = c.health_bonus * pct_maladies
        else:
            terms["health"] = 0.0

        # --- 3. Bonus d'irrigation (zones sèches arrosées) ---
        if total_sec > 0:
            pct_sec = sec_arrosees / total_sec
            terms["irrigation"] = c.irrigation_bonus * pct_sec
        else:
            terms["irrigation"] = 0.0

        # --- 4. Bonus d'exploration (cartographie) ---
        terms["exploration"] = c.exploration_bonus * visited_percentage

        # --- 5. Bonus terminal si objectif atteint ---
        terms["goal"] = c.goal_reward if reached_goal else 0.0

        # --- 6. Pénalités pour gaspillage de ressources ---
        # Pulvérisation inutile (plus de maladies ou aucune)
        if pesticide_used > 0 and (total_malades == 0 or maladies_traitees >= total_malades):
            terms["waste_pesticide"] = -c.waste_penalty
        else:
            terms["waste_pesticide"] = 0.0

        # Arrosage inutile (plus de zones sèches ou aucune)
        if water_used > 0 and (total_sec == 0 or sec_arrosees >= total_sec):
            terms["waste_water"] = -c.waste_penalty
        else:
            terms["waste_water"] = 0.0

        # --- 7. Pénalité si batterie faible et pas en retour ---
        if battery_level < 10.0 and not returning_home:
            terms["low_battery"] = -c.low_battery_penalty
        else:
            terms["low_battery"] = 0.0

        # --- 8. Termes de contrôle (identique à compute) ---
        terms["heading"] = c.heading_weight * (1.0 - heading_error / np.pi) * 2.0 - c.heading_weight

        action_delta = np.linalg.norm(action - self._prev_action)
        terms["smooth"] = -c.smooth_weight * action_delta
        self._prev_action = action.copy()

        throttle_normalized = (action[0] + 1.0) / 2.0
        terms["energy"] = -c.energy_alpha * (throttle_normalized ** 2)
        terms["stability"] = -c.stability_weight * float(np.sum(np.square(angular_rates)))
        terms["time"] = -c.time_penalty

        # --- 9. Pénalités terminales (crash, sortie, retournement) ---
        terms["collision"] = -c.collision_penalty if collided else 0.0
        terms["out_of_bounds"] = -c.out_of_bounds_penalty if out_of_bounds else 0.0
        terms["flip"] = -c.flip_penalty if flipped else 0.0

        # Somme de tous les termes
        total = sum(terms.values())
        return float(total), terms

    # ------------------------------------------------------------------
    # Méthode pour la tâche d'irrigation avec gestion de ressources
    def compute_water_task(
        self,
        tank_level: float,
        prev_dist: float,
        curr_dist: float,
        just_watered: bool,
        just_refilled: bool,
        all_watered: bool,
        num_unwatered: int,
    ) -> tuple[float, dict]:
        """Calcule la récompense pour la tâche d'irrigation avec gestion de réservoir.

        La récompense combine les éléments suivants :
            - **Arrosage** (``watering_reward``) : lorsque un groupe de plantes
              non arrosé est arrosé avec succès.
            - **Remplissage** (``refill_reward``) : lorsque le drone retourne
              à la bassine et remplit le réservoir.
            - **Pénalité temporelle** : proportionnelle au nombre de groupes
              restant à arroser, incitant l'agent à agir rapidement.
            - **Shaping de distance** : récompense bonus si le drone se
              rapproche du groupe non arrosé le plus proche.
            - **Mission accomplie** (``mission_complete_reward``) : bonus
              terminal lorsque tous les groupes sont arrosés.

        Args:
            tank_level: Niveau actuel du réservoir d'eau (0–100).
            prev_dist: Distance au groupe non arrosé le plus proche (étape précédente).
            curr_dist: Distance au groupe non arrosé le plus proche (étape courante).
            just_watered: True si un groupe vient d'être arrosé à cette étape.
            just_refilled: True si le réservoir vient d'être rempli à cette étape.
            all_watered: True si tous les groupes de plantes sont arrosés.
            num_unwatered: Nombre de groupes restant à arroser.

        Returns:
            Récompense totale calculée pour cette étape (float).
        """
        c = self.cfg
        terms = {}

        # Récompense d'arrosage
        terms["watering"] = c.watering_reward if just_watered else 0.0

        # Récompense de remplissage
        terms["refill"] = c.refill_reward if just_refilled else 0.0

        # Pénalité temporelle proportionnelle au nombre de groupes non arrosés
        terms["time_penalty"] = -c.time_penalty_per_group * num_unwatered

        # Shaping de distance : récompense pour se rapprocher du groupe le plus proche
        terms["distance_shaping"] = 0.0
        if num_unwatered > 0 and prev_dist < float("inf") and curr_dist < float("inf"):
            if curr_dist < prev_dist:
                terms["distance_shaping"] = c.distance_shaping_reward

        # Bonus de mission accomplie
        terms["mission_complete"] = c.mission_complete_reward if all_watered else 0.0

        total_reward = sum(terms.values())
        return float(total_reward), terms