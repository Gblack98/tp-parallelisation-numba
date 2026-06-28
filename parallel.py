"""
Version parallèle.

Même calcul que la version séquentielle, mais la boucle est répartie sur tous les
threads avec Numba (@njit(parallel=True) et prange). La somme des moyennes est une
réduction : Numba la gère automatiquement, sans condition de course.

Chaque étudiant étant indépendant des autres, la parallélisation est exacte.
"""

from numba import njit, prange

COEF_TOTAL = 5 + 4 + 2  # 11


@njit(parallel=True, cache=True)
def moyenne_promo_par(M, P, A):
    """Moyenne de la promo, calculée en parallèle (prange + réduction)."""
    total = 0.0
    n = M.shape[0]
    for i in prange(n):
        total += (M[i] * 5 + P[i] * 4 + A[i] * 2) / COEF_TOTAL
    return total / n
