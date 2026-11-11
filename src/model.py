"""
Modelo predictivo de mortalidad en pisciculturas.

Objetivo: predecir el % de mortalidad del DÍA SIGUIENTE de un lote, usando solo
información disponible hoy (ambiente, alimentación, crecimiento y mortalidad
reciente). Sirve para anticipar desviaciones antes de que ocurran.

Decisiones de diseño:
  - Las variables de bitácora (falla/brote) NO se usan como entrada, porque
    el modelo debe inferir el problema desde los datos operacionales.
  - Validación temporal: se entrena con el pasado y se evalúa en el futuro
    (expanding window), con 1 día de separación para evitar fugas de información.
  - Se compara contra baselines simples; un modelo complejo debe superarlos.

Uso:
    python src/model.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42
N_FOLDS = 4
GAP_DIAS = 1
OBJETIVO = "mortalidad_siguiente_pct"

BASE_NUMERICAS = [
    "temperatura_c", "oxigeno_mgl", "ph", "amonio_mgl", "caudal_ls",
    "densidad_kgm3", "peso_prom_g", "edad_dias", "pct_alimento",
    "mortalidad_dia_pct", "sgr_pct_dia", "fcr",
]


# ---------------------------------------------------------------------------
# 1. VARIABLES (feature engineering)
# ---------------------------------------------------------------------------
def construir_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["lote", "fecha"]).reset_index(drop=True).copy()
    g = df.groupby("lote")

    df["pct_alimento"] = df["alimento_kg"] / df["biomasa_kg"] * 100

    # Rezagos de mortalidad y oxígeno (lo que pasó ayer y antes de ayer)
    for k in (1, 2, 3):
        df[f"mortalidad_lag{k}"] = g["mortalidad_dia_pct"].shift(k)
    for k in (1, 2):
        df[f"oxigeno_lag{k}"] = g["oxigeno_mgl"].shift(k)

    # Promedios y mínimos móviles (incluyen el día actual)
    df["mortalidad_media_3d"] = g["mortalidad_dia_pct"].transform(
        lambda s: s.rolling(3, min_periods=3).mean())
    df["mortalidad_media_7d"] = g["mortalidad_dia_pct"].transform(
        lambda s: s.rolling(7, min_periods=7).mean())
    df["oxigeno_min_3d"] = g["oxigeno_mgl"].transform(
        lambda s: s.rolling(3, min_periods=3).min())
    df["temperatura_media_7d"] = g["temperatura_c"].transform(
        lambda s: s.rolling(7, min_periods=7).mean())

    # Variable objetivo: mortalidad del día siguiente
    df[OBJETIVO] = g["mortalidad_dia_pct"].shift(-1)

    # Etapa productiva como variables 0/1
    df = pd.concat([df, pd.get_dummies(df["etapa"], prefix="etapa", dtype=float)], axis=1)
    return df


def columnas_modelo(df: pd.DataFrame) -> list[str]:
    derivadas = [
        "mortalidad_lag1", "mortalidad_lag2", "mortalidad_lag3",
        "oxigeno_lag1", "oxigeno_lag2",
        "mortalidad_media_3d", "mortalidad_media_7d",
        "oxigeno_min_3d", "temperatura_media_7d",
    ]
    etapas = [c for c in df.columns if c.startswith("etapa_")]
    return BASE_NUMERICAS + derivadas + etapas


# ---------------------------------------------------------------------------
# 2. VALIDACIÓN TEMPORAL
# ---------------------------------------------------------------------------
def folds_temporales(fechas: pd.Series, n_folds: int = N_FOLDS, gap: int = GAP_DIAS):
    """Genera (índices_train, índices_test) con ventana expansiva por fecha."""
    unicas = np.sort(fechas.unique())
    cortes = np.linspace(0.4 * len(unicas), len(unicas), n_folds + 1).astype(int)
    for i in range(n_folds):
        ini, fin = cortes[i], cortes[i + 1]
        train = fechas < unicas[max(ini - gap, 0)]
        test = (fechas >= unicas[ini]) & (fechas < unicas[fin] if fin < len(unicas) else True)
        yield np.flatnonzero(train), np.flatnonzero(test)


# ---------------------------------------------------------------------------
# 3. MODELOS
# ---------------------------------------------------------------------------
def crear_modelos():
    return {
        "Ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               Ridge(alpha=1.0)),
        "RandomForest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                  n_jobs=-1, random_state=SEED)),
        "GradientBoosting": HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=300, max_depth=4,
            min_samples_leaf=20, random_state=SEED),
    }


def metricas(y, pred) -> dict:
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "R2": r2_score(y, pred),
    }


# ---------------------------------------------------------------------------
# 4. EVALUACIÓN
# ---------------------------------------------------------------------------
def evaluar(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X, y = df[cols], df[OBJETIVO]
    filas = []
    for i, (tr, te) in enumerate(folds_temporales(df["fecha"]), start=1):
        # Baselines: no requieren entrenamiento
        filas.append({"fold": i, "modelo": "Baseline: mortalidad de hoy",
                      **metricas(y.iloc[te], df["mortalidad_dia_pct"].iloc[te])})
        filas.append({"fold": i, "modelo": "Baseline: media móvil 7d",
                      **metricas(y.iloc[te], df["mortalidad_media_7d"].iloc[te])})
        for nombre, modelo in crear_modelos().items():
            modelo.fit(X.iloc[tr], y.iloc[tr])
            filas.append({"fold": i, "modelo": nombre,
                          **metricas(y.iloc[te], modelo.predict(X.iloc[te]))})
    return pd.DataFrame(filas)


def resumen(res: pd.DataFrame) -> pd.DataFrame:
    r = res.groupby("modelo")[["MAE", "RMSE", "R2"]].agg(["mean", "std"])
    r.columns = [f"{a}_{b}" for a, b in r.columns]
    return r.sort_values("MAE_mean").round(4)


# ---------------------------------------------------------------------------
# 5. ENTRENAMIENTO FINAL E IMPORTANCIA DE VARIABLES
# ---------------------------------------------------------------------------
def importancia_variables(df, cols, out: Path):
    """Entrena un RandomForest con todo y grafica qué variables pesan más."""
    modelo = crear_modelos()["RandomForest"].fit(df[cols], df[OBJETIVO])
    imp = pd.Series(modelo[-1].feature_importances_, index=cols).sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    imp.tail(12).plot.barh(ax=ax, color="#2a7ab0")
    ax.set_title("Variables más importantes (RandomForest)")
    ax.set_xlabel("Importancia")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "importancia_variables.png", dpi=150)
    plt.close(fig)
    return modelo, imp.sort_values(ascending=False)


# ---------------------------------------------------------------------------
# 6. EJECUCIÓN
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Modelo predictivo de mortalidad")
    p.add_argument("--entrada", default="data/processed/indicadores_diarios.csv")
    p.add_argument("--salida", default="data/processed")
    p.add_argument("--modelos", default="models")
    p.add_argument("--figuras", default="reports/figuras")
    a = p.parse_args()

    df = construir_variables(pd.read_csv(a.entrada, parse_dates=["fecha"]))
    cols = columnas_modelo(df)
    df = df.dropna(subset=[OBJETIVO, "mortalidad_media_7d", "mortalidad_lag3"])
    df = df.sort_values(["fecha", "estanque"]).reset_index(drop=True)
    print(f"Filas para modelar: {len(df)} | variables: {len(cols)}")

    res = evaluar(df, cols)
    tabla = resumen(res)
    out = Path(a.salida)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "validacion_modelos.csv", index=False)
    print("\n== Validación temporal (promedio de %d folds) ==" % N_FOLDS)
    print(tabla.to_string())

    modelo, imp = importancia_variables(df, cols, Path(a.figuras))
    print("\nTop 8 variables:")
    print(imp.head(8).round(3).to_string())

    Path(a.modelos).mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": modelo, "columnas": cols},
                Path(a.modelos) / "modelo_mortalidad.joblib")
    print(f"\nModelo guardado en {Path(a.modelos).resolve()}")


if __name__ == "__main__":
    main()
