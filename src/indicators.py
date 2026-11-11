"""
Indicadores operacionales de piscicultura.

Lee data/processed/dataset_integrado.csv y calcula:

  Diarios (por lote):
    - mortalidad_dia_pct   : % de peces muertos en el día
    - mortalidad_acum_pct  : % acumulado respecto de los peces iniciales
    - sgr_pct_dia          : tasa de crecimiento específico (ventana móvil)
    - fcr                  : factor de conversión (ventana móvil)

  Resumen por lote (un lote = un ciclo en un estanque):
    - mortalidad total, SGR global, FCR global, densidad máxima, días de
      eventos, y si el lote está completo.

Uso:
    python src/indicators.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

VENTANA = 14        # días de la ventana móvil para SGR y FCR
DIAS_CICLO = 200    # un lote completo tiene 200 días de datos


# ---------------------------------------------------------------------------
# 1. CARGA
# ---------------------------------------------------------------------------
def cargar(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["fecha"])
    return df.sort_values(["lote", "fecha"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. INDICADORES DIARIOS
# ---------------------------------------------------------------------------
def indicadores_diarios(df: pd.DataFrame, ventana: int = VENTANA) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("lote")

    # Peso y biomasa muerta "reales" (derivados de la biomasa, sin ruido de muestreo)
    df["peso_real_g"] = df["biomasa_kg"] * 1000 / df["peces"]
    df["biomasa_muerta_kg"] = df["mortalidad_n"] * df["peso_real_g"] / 1000

    # --- Mortalidad ---
    df["peces_iniciales_lote"] = g["peces"].transform("first")
    df["mortalidad_dia_pct"] = df["mortalidad_n"] / df["peces"] * 100
    df["mortalidad_acum_pct"] = (
        g["mortalidad_n"].cumsum() / df["peces_iniciales_lote"] * 100
    )

    # --- SGR (%/día): 100 * (ln(Pf) - ln(Pi)) / días ---
    peso_previo = g["peso_prom_g"].shift(ventana)
    df["sgr_pct_dia"] = 100 * np.log(df["peso_prom_g"] / peso_previo) / ventana

    # --- FCR = alimento entregado / biomasa ganada ---
    # Ganancia = cambio de biomasa + biomasa de los peces que murieron en la ventana.
    # El alimento y las muertes de los días t-ventana..t-1 explican el cambio
    # de biomasa entre t-ventana y t, por eso se usa shift(1).
    def suma_ventana(col):
        return g[col].transform(lambda s: s.shift(1).rolling(ventana).sum())

    df["alimento_ventana_kg"] = suma_ventana("alimento_kg")
    ganancia = (
        df["biomasa_kg"] - g["biomasa_kg"].shift(ventana) + suma_ventana("biomasa_muerta_kg")
    )
    df["ganancia_biomasa_kg"] = ganancia
    df["fcr"] = np.where(ganancia > 0, df["alimento_ventana_kg"] / ganancia, np.nan)
    return df


# ---------------------------------------------------------------------------
# 3. RESUMEN POR LOTE
# ---------------------------------------------------------------------------
def resumen_lotes(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("lote")
    res = g.agg(
        estanque=("estanque", "first"),
        inicio=("fecha", "min"),
        fin=("fecha", "max"),
        dias=("fecha", "count"),
        peces_iniciales=("peces", "first"),
        mortalidad_total=("mortalidad_n", "sum"),
        peso_inicial_g=("peso_real_g", "first"),
        peso_final_g=("peso_real_g", "last"),
        biomasa_inicial_kg=("biomasa_kg", "first"),
        biomasa_final_kg=("biomasa_kg", "last"),
        densidad_max_kgm3=("densidad_kgm3", "max"),
        dias_falla_aireacion=("falla_aireacion", "sum"),
        dias_brote=("brote_enfermedad", "sum"),
    )

    # Alimento y biomasa muerta de todos los días menos el último (que aún
    # no se refleja en la biomasa final), para que el FCR global sea consistente.
    interior = df[df["fecha"] < g["fecha"].transform("max")]
    gi = interior.groupby("lote")
    alimento_total = gi["alimento_kg"].sum()
    muerta_total = gi["biomasa_muerta_kg"].sum()

    res["alimento_total_kg"] = alimento_total
    res["mortalidad_pct"] = res["mortalidad_total"] / res["peces_iniciales"] * 100
    res["sgr_global_pct_dia"] = (
        100 * np.log(res["peso_final_g"] / res["peso_inicial_g"]) / (res["dias"] - 1)
    )
    ganancia = res["biomasa_final_kg"] - res["biomasa_inicial_kg"] + muerta_total
    res["fcr_global"] = res["alimento_total_kg"] / ganancia
    res["lote_completo"] = res["dias"] == DIAS_CICLO
    return res.reset_index()


# ---------------------------------------------------------------------------
# 4. EJECUCIÓN
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Cálculo de indicadores")
    p.add_argument("--entrada", default="data/processed/dataset_integrado.csv")
    p.add_argument("--salida", default="data/processed")
    a = p.parse_args()

    df = indicadores_diarios(cargar(Path(a.entrada)))
    lotes = resumen_lotes(df)

    out = Path(a.salida)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "indicadores_diarios.csv", index=False)
    cols = lotes.select_dtypes("float").columns
    lotes[cols] = lotes[cols].round(3)
    lotes.to_csv(out / "resumen_lotes.csv", index=False)

    comp = lotes[lotes["lote_completo"]]
    print(f"Lotes: {len(lotes)} ({len(comp)} completos)")
    print("\nPromedios de lotes completos:")
    print(f"  Mortalidad total : {comp['mortalidad_pct'].mean():.2f} %")
    print(f"  SGR global       : {comp['sgr_global_pct_dia'].mean():.2f} %/día")
    print(f"  FCR global       : {comp['fcr_global'].mean():.2f}")
    print(f"  Peso final       : {comp['peso_final_g'].mean():.0f} g")
    print(f"\nGuardado en {out.resolve()}")


if __name__ == "__main__":
    main()
