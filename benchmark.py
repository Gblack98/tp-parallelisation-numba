"""
Compare la version séquentielle et la version parallèle :
  - vérifie que les deux donnent le même résultat ;
  - mesure le temps de chacune (hors compilation JIT) ;
  - mesure la montée en charge sur 1, 2, 4 et 8 threads ;
  - calcule le speedup et en déduit la proportion parallélisable (loi d'Amdahl).

Le temps retenu est le meilleur (min) sur plusieurs exécutions : c'est la mesure
la plus stable pour un micro-benchmark, car elle filtre le bruit de la machine.

Usage :
    python benchmark.py --csv etudiants.csv --repeat 100
"""

import argparse
import time

import pandas as pd
from numba import get_num_threads, set_num_threads

from sequential import moyenne_promo_seq
from parallel import moyenne_promo_par


def chrono(fonction, args, repeat: int) -> float:
    temps = []
    for _ in range(repeat):
        debut = time.perf_counter()
        fonction(*args)
        temps.append(time.perf_counter() - debut)
    return min(temps)


def proportion_parallelisable(speedup: float, n_threads: int) -> float:
    """Loi d'Amdahl : S = 1 / ((1-p) + p/n)  =>  p = (1 - 1/S) / (1 - 1/n)."""
    return (1 - 1 / speedup) / (1 - 1 / n_threads)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="etudiants.csv")
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    M, P, A = df["M"].to_numpy(), df["P"].to_numpy(), df["A"].to_numpy()
    n_max = get_num_threads()
    print(f"{len(df):,} étudiants | {n_max} threads disponibles\n")

    # Compilation JIT déclenchée une fois, hors mesure.
    moy_seq = moyenne_promo_seq(M, P, A)
    moy_par = moyenne_promo_par(M, P, A)
    print(f"Moyenne de la promo (séquentiel) : {moy_seq:.6f}")
    print(f"Moyenne de la promo (parallèle)  : {moy_par:.6f}")
    print(f"Résultats identiques : {abs(moy_seq - moy_par) < 1e-6}\n")

    # Référence séquentielle.
    t_seq = chrono(moyenne_promo_seq, (M, P, A), args.repeat)
    print(f"Séquentiel : {t_seq * 1000:7.2f} ms\n")

    # Montée en charge de la version parallèle.
    print("Threads | Temps (ms) | Speedup")
    print("--------|------------|--------")
    threads = [t for t in (1, 2, 4, 8) if t <= n_max]
    meilleur = (1, 0.0)  # (n_threads, speedup)
    for nt in threads:
        set_num_threads(nt)
        moyenne_promo_par(M, P, A)  # re-spécialisation éventuelle
        t = chrono(moyenne_promo_par, (M, P, A), args.repeat)
        su = t_seq / t
        print(f"{nt:>7} | {t * 1000:10.2f} | {su:6.2f}x")
        if su > meilleur[1]:
            meilleur = (nt, su)

    n_best, speedup = meilleur
    p = proportion_parallelisable(speedup, n_best)
    print(f"\nMeilleur speedup : {speedup:.2f}x  ({n_best} threads)")
    print(f"Proportion parallélisable (Amdahl, n={n_best}) : {p:.3f}  (soit {p * 100:.1f} %)")


if __name__ == "__main__":
    main()
