"""
Génère les données de N étudiants dans un fichier CSV.

Chaque étudiant a trois notes indépendantes : Maths (M), Physique (P), Anglais (A),
tirées au hasard entre 0 et 20.

Usage :
    python generate_data.py --n 1000000 --out etudiants.csv
"""

import argparse

import numpy as np
import pandas as pd


def generer(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "M": np.round(rng.uniform(0, 20, n), 2),
        "P": np.round(rng.uniform(0, 20, n), 2),
        "A": np.round(rng.uniform(0, 20, n), 2),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Génère les notes de N étudiants")
    parser.add_argument("--n", type=int, default=1_000_000, help="nombre d'étudiants")
    parser.add_argument("--out", type=str, default="etudiants.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generer(args.n, args.seed)
    df.to_csv(args.out, index=False)
    print(f"{len(df):,} étudiants écrits dans {args.out}")


if __name__ == "__main__":
    main()
