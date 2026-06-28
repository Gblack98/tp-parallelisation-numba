"""
Version séquentielle.

On calcule la moyenne de chaque étudiant, moyenne = (M*5 + P*4 + A*2) / 11,
puis la moyenne de toute la promo. Les étudiants étant indépendants, la boucle
n'a aucune dépendance : c'est le calcul que l'on parallélisera ensuite.

Cette version est compilée par Numba (@njit) mais s'exécute sur un seul thread :
elle sert de référence pour mesurer le speedup.
"""

import numpy as np
from numba import njit

COEF_TOTAL = 5 + 4 + 2  # 11


@njit(cache=True)
def moyenne_promo_seq(M, P, A):
    """Moyenne de la promo, calculée étudiant par étudiant (séquentiel)."""
    total = 0.0
    n = M.shape[0]
    for i in range(n):
        total += (M[i] * 5 + P[i] * 4 + A[i] * 2) / COEF_TOTAL
    return total / n
