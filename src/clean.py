"""
Limpieza e integración de datos de piscicultura.

Pasos:
  1. Carga las 3 fuentes crudas (ambiental, productivo, eventos).
  2. Ambiental: elimina duplicados, marca como faltantes los valores fuera de
     rango físico (errores de sensor) e interpola huecos cortos (máx. 3 días).
  3. Marca en cada día si hubo falla de aireación o brote (según bitácora).
  4. Integra todo en un solo dataset diario por estanque.
  5. Valida el resultado y guarda data/processed/dataset_integrado.csv.

Uso:
    python src/clean.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Rangos plausibles para agua dulce de piscicultura; fuera de ellos = error de sensor
RANGOS = {
    "temperatura_c": (0, 25),
    "oxigeno_mgl": (0.5, 20),
    "ph": (5.5, 8.5),
}
MAX_HUECO_DIAS = 3
TIPOS_EVENTO = ["falla_aireacion", "brote_enfermedad"]


def cargar(raw: Path):
    ambiental = pd.read_csv(raw / "ambiental.csv", parse_dates=["fecha"])
    productivo = pd.read_csv(raw / "productivo.csv", parse_dates=["fecha"])
    eventos = pd.read_csv(raw / "eventos.csv", parse_dates=["fecha"])
    return ambiental, productivo, eventos


def limpiar_ambiental(df: pd.DataFrame):
    """Devuelve el DataFrame limpio y un reporte con lo que se corrigió."""
    reporte = {}
    n0 = len(df)
    df = (df.sort_values(["estanque", "fecha"])
            .drop_duplicates(["estanque", "fecha"])
            .reset_index(drop=True))
    reporte["duplicados_eliminados"] = n0 - len(df)

    # 1) valores imposibles -> NaN
    for col, (lo, hi) in RANGOS.items():
        fuera = df[col].notna() & ~df[col].between(lo, hi)
        reporte[f"{col}: fuera de rango"] = int(fuera.sum())
        df.loc[fuera, col] = np.nan

    # 2) registrar qué filas tendrán algún valor imputado
    cols = list(RANGOS)
    df["sensor_imputado"] = df[cols].isna().any(axis=1)
    reporte["filas con faltantes (antes de imputar)"] = int(df["sensor_imputado"].sum())

    # 3) interpolar por estanque, solo huecos cortos
    df[cols] = df.groupby("estanque")[cols].transform(
        lambda s: s.interpolate(limit=MAX_HUECO_DIAS, limit_direction="both")
    )
    reporte["faltantes que quedan"] = int(df[cols].isna().sum().sum())
    return df, reporte


def marcar_eventos(df: pd.DataFrame, eventos: pd.DataFrame) -> pd.DataFrame:
    """Marca los días en que estuvo activo cada tipo de evento."""
    df = df.copy()
    for tipo in TIPOS_EVENTO:
        df[tipo] = False
    for ev in eventos.itertuples():
        dias = pd.date_range(ev.fecha, periods=int(ev.duracion_dias))
        mask = (df["estanque"] == ev.estanque) & df["fecha"].isin(dias)
        df.loc[mask, ev.tipo] = True
    return df


def integrar(ambiental, productivo, eventos):
    ambiental, reporte = limpiar_ambiental(ambiental)
    df = productivo.merge(ambiental, on=["fecha", "estanque"],
                          how="left", validate="one_to_one")
    df = marcar_eventos(df, eventos)
    df = df.sort_values(["estanque", "fecha"]).reset_index(drop=True)
    return df, reporte


def validar(df: pd.DataFrame):
    """Chequeos básicos de consistencia; falla si algo está mal."""
    assert (df["peso_prom_g"] > 0).all(), "peso no positivo"
    assert (df["mortalidad_n"] >= 0).all(), "mortalidad negativa"
    assert not df.duplicated(["estanque", "fecha"]).any(), "filas duplicadas"
    assert df["temperatura_c"].between(*RANGOS["temperatura_c"]).all(), "temperatura fuera de rango"
    assert df["oxigeno_mgl"].between(*RANGOS["oxigeno_mgl"]).all(), "oxígeno fuera de rango"


def main():
    p = argparse.ArgumentParser(description="Limpieza e integración")
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--salida", default="data/processed")
    a = p.parse_args()

    ambiental, productivo, eventos = cargar(Path(a.raw))
    df, reporte = integrar(ambiental, productivo, eventos)
    validar(df)

    out = Path(a.salida)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "dataset_integrado.csv", index=False)

    print("== Reporte de limpieza ==")
    for k, v in reporte.items():
        print(f"{k:<45} {v}")
    print(f"\nDataset integrado: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"Días con falla de aireación: {int(df['falla_aireacion'].sum())}")
    print(f"Días con brote de enfermedad: {int(df['brote_enfermedad'].sum())}")
    print(f"Guardado en {(out / 'dataset_integrado.csv').resolve()}")


if __name__ == "__main__":
    main()
