# TP : Paralléliser un calcul de moyenne avec Numba

Calcul de la moyenne d'une promo de N étudiants, en séquentiel puis en parallèle avec Numba.

Le rapport complet (résultats, speedup, loi d'Amdahl) est dans `rapport.pdf`.

## Lancer

```bash
pip install -r requirements.txt
python generate_data.py --n 10000000 --out etudiants.csv
python benchmark.py --csv etudiants.csv --repeat 100
```

## Fichiers

- `generate_data.py` : génère les étudiants dans un CSV
- `sequential.py` : version séquentielle (`@njit`)
- `parallel.py` : version parallèle (`prange`)
- `benchmark.py` : speedup et loi d'Amdahl
