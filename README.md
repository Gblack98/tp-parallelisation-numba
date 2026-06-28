# TP : Paralléliser un calcul de moyenne avec Numba

**Ibrahima Gabar Diop** · Machine Learning / Programmation parallèle (M1 MBDA) · Université Cheikh Hamidou Kane

Calcul de la moyenne d'une promo de N étudiants, en séquentiel puis en parallèle avec Numba, mesure du speedup et estimation de la proportion parallélisable par la loi d'Amdahl.

## Le calcul

Chaque étudiant a trois notes, pondérées par des coefficients :

| Matière | Coefficient |
|---|---|
| Maths (M) | 5 |
| Physique (P) | 4 |
| Anglais (A) | 2 |

Moyenne d'un étudiant : `(M*5 + P*4 + A*2) / 11`. On calcule cette moyenne pour chacun, puis la moyenne de toute la promo. Les étudiants sont indépendants : aucune note ne dépend d'une autre, donc la boucle se parallélise sans risque de résultat faux.

## Organisation

| Fichier | Rôle |
|---|---|
| `generate_data.py` | Génère N étudiants dans un CSV (notes tirées entre 0 et 20) |
| `sequential.py` | Version séquentielle, compilée avec `@njit` (un seul thread) |
| `parallel.py` | Version parallèle, `@njit(parallel=True)` avec `prange` |
| `benchmark.py` | Vérifie l'égalité des résultats, mesure les temps, le speedup et Amdahl |

La version parallèle utilise `prange` : Numba répartit la boucle sur les threads et gère la somme finale comme une réduction, sans condition de course.

## Résultats

Mesure sur **10 000 000 d'étudiants**, 4 cœurs physiques (8 threads logiques). Temps = meilleur sur 100 exécutions, compilation JIT exclue.

```
Moyenne séquentiel : 10.000366
Moyenne parallèle  : 10.000366   (résultats identiques)

Séquentiel : 18.28 ms

Threads | Temps (ms) | Speedup
--------|------------|--------
   1    |   17.20    |  1.06x
   2    |    9.35    |  1.96x
   4    |    8.66    |  2.11x
   8    |    9.88    |  1.85x
```

- À 1 thread, la version parallèle rejoint la séquentielle : l'overhead de `prange` est négligeable.
- À 2 threads, le speedup est presque idéal (1.96x).
- Le pic est à **4 threads (2.11x)**, soit le nombre de cœurs physiques.
- À 8 threads, le speedup baisse (1.85x) : les 4 threads supplémentaires sont des hyperthreads qui partagent les mêmes cœurs et la même bande passante mémoire.

## Speedup et loi d'Amdahl

Speedup = temps séquentiel / temps parallèle = **2.11x** (meilleur point, 4 threads).

Loi d'Amdahl : `S = 1 / ((1 - p) + p/n)`, avec `p` la proportion parallélisable et `n` le nombre de threads. On isole `p` :

```
p = (1 - 1/S) / (1 - 1/n)
p = (1 - 1/2.11) / (1 - 1/4) = 0.526 / 0.75 ≈ 0.70
```

**Proportion parallélisable estimée : ≈ 70 %.**

## Discussion

Le calcul est très simple (quelques opérations par étudiant) mais lit beaucoup de données : il est en partie **borné par la bande passante mémoire**. C'est pourquoi le speedup plafonne autour de 2x au lieu d'atteindre le nombre de cœurs, et pourquoi l'hyperthreading n'apporte rien. C'est le comportement attendu pour ce type de tâche : parallèle à 100 % en théorie, mais limitée en pratique par l'accès mémoire, ce que la loi d'Amdahl traduit par un `p` inférieur à 1.

Les valeurs exactes varient légèrement d'une exécution à l'autre selon la charge de la machine ; l'ordre de grandeur (speedup ~2x, `p` ~60-70 %) reste stable.

## Reproduire

```bash
pip install -r requirements.txt
python generate_data.py --n 10000000 --out etudiants.csv
python benchmark.py --csv etudiants.csv --repeat 100
```

## Dépôt

https://github.com/Gblack98/tp-parallelisation-numba
