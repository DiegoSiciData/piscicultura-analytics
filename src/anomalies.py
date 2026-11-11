"""
Detección de desviaciones operacionales en pisciculturas.

Combina dos enfoques:
  1. Reglas operacionales (umbrales fijos): oxígeno bajo, temperatura alta,
     densidad alta.
  2. Estadística robusta: z-score modificado (mediana/MAD) calculado por
     etapa productiva, para mortalidad, FCR y SGR.

Luego agrupa los días con alerta en "episodios" y evalúa qué tan bien
detectan los eventos registrados en la bitácora (fallas de aireación y
brotes). La bitácora proviene del simulador, por eso sirve como "verdad".

Uso:
    python src/anomalies.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. PARÁMETROS
# ---------------------------------------------------------------------------
UMBRALES = {
    "oxigeno_min_mgl": 6.5,
    "caida_oxigeno_mgl": 1.5,   # caída respecto de lo habitual del estanque
    "temperatura_max_c": 14.0,
    "densidad_max_kgm3": 40.0,
}
Z_UMBRAL = 3.5          # límite habitual para z-score modificado
VENTANA_BASE = 14       # días previos que definen el "oxígeno habitual"
GAP_EPISODIO_DIAS = 1   # días sin alerta tolerados dentro de un mismo episodio
TOLERANCIA_DIAS = 2     # holgura al comparar episodios con la bitácora

# señales que cuentan como "desviación" (las de riesgo se reportan aparte)
SENALES_DESVIACION = ["regla_oxigeno", "est_oxigeno", "est_mortalidad", "est_sgr", "est_fcr"]
SENALES_RIESGO = ["riesgo_temperatura", "riesgo_densidad"]


# ---------------------------------------------------------------------------
# 2. FUNCIONES ESTADÍSTICAS
# ---------------------------------------------------------------------------
def z_robusto(s: pd.Series) -> pd.Series:
    """Z-score modificado: usa mediana y MAD, resistente a outliers."""
    mediana = s.median()
    mad = (s - mediana).abs().median()
    if not mad or np.isnan(mad):
        return pd.Series(0.0, index=s.index)
    return 0.6745 * (s - mediana) / mad


# ---------------------------------------------------------------------------
# 3. DETECCIÓN DIARIA
# ---------------------------------------------------------------------------
def detectar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["estanque", "fecha"]).reset_index(drop=True)

    # Reglas operacionales
    df["regla_oxigeno"] = df["oxigeno_mgl"] < UMBRALES["oxigeno_min_mgl"]
    df["riesgo_temperatura"] = df["temperatura_c"] > UMBRALES["temperatura_max_c"]
    df["riesgo_densidad"] = df["densidad_kgm3"] > UMBRALES["densidad_max_kgm3"]

    # Estadística robusta por etapa (cada etapa tiene su propio "normal")
    por_etapa = df.groupby("etapa")
    df["z_mortalidad"] = por_etapa["mortalidad_dia_pct"].transform(z_robusto)
    df["z_fcr"] = por_etapa["fcr"].transform(z_robusto)
    df["z_sgr"] = por_etapa["sgr_pct_dia"].transform(z_robusto)

    # Mortalidad: alza sostenida (2 días seguidos) para evitar falsas alarmas
    # por variación aleatoria de un solo día.
    alto = df["z_mortalidad"] > Z_UMBRAL
    df["est_mortalidad"] = alto & alto.groupby(df["lote"]).shift(1, fill_value=False)
    df["est_fcr"] = df["z_fcr"].abs() > Z_UMBRAL                    # alzas o bajas
    df["est_sgr"] = df["z_sgr"] < -Z_UMBRAL                         # solo caídas

    # Oxígeno: caída brusca respecto de la mediana de los 14 días previos
    base_oxigeno = df.groupby("estanque")["oxigeno_mgl"].transform(
        lambda x: x.shift(1).rolling(VENTANA_BASE, min_periods=7).median()
    )
    df["est_oxigeno"] = (base_oxigeno - df["oxigeno_mgl"]) > UMBRALES["caida_oxigeno_mgl"]

    df["n_senales"] = df[SENALES_DESVIACION].sum(axis=1)
    df["alerta"] = df["n_senales"] > 0
    return df


# ---------------------------------------------------------------------------
# 4. EPISODIOS
# ---------------------------------------------------------------------------
def episodios(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa días con alerta consecutivos (por estanque) en episodios."""
    a = df[df["alerta"]].sort_values(["estanque", "fecha"]).copy()
    salto = a.groupby("estanque")["fecha"].diff().dt.days
    a["episodio_id"] = (salto.isna() | (salto > GAP_EPISODIO_DIAS + 1)).cumsum()

    ep = a.groupby("episodio_id").agg(
        estanque=("estanque", "first"),
        inicio=("fecha", "min"),
        fin=("fecha", "max"),
        dias_alerta=("fecha", "count"),
        oxigeno_min=("oxigeno_mgl", "min"),
        mortalidad_max_pct=("mortalidad_dia_pct", "max"),
        **{s: (s, "max") for s in SENALES_DESVIACION},
    )
    ep["senales"] = ep[SENALES_DESVIACION].apply(
        lambda f: ",".join(k for k, v in f.items() if v), axis=1
    )
    return ep.drop(columns=SENALES_DESVIACION).reset_index()


# ---------------------------------------------------------------------------
# 5. EVALUACIÓN CONTRA LA BITÁCORA
# ---------------------------------------------------------------------------
def evaluar(ep: pd.DataFrame, eventos: pd.DataFrame, tol: int = TOLERANCIA_DIAS):
    ev = eventos.copy()
    ev["evento_id"] = range(len(ev))
    ev["ev_inicio"] = ev["fecha"]
    ev["ev_fin"] = ev["fecha"] + pd.to_timedelta(ev["duracion_dias"] - 1, unit="D")

    cruce = ev.merge(ep, on="estanque", how="inner")
    holgura = pd.Timedelta(days=tol)
    solapa = (cruce["inicio"] <= cruce["ev_fin"] + holgura) & (
        cruce["fin"] >= cruce["ev_inicio"] - holgura
    )
    cruce = cruce[solapa]

    detectados = ev["evento_id"].isin(cruce["evento_id"])
    ep_validos = ep["episodio_id"].isin(cruce["episodio_id"])

    por_tipo = (
        ev.assign(detectado=detectados).groupby("tipo")["detectado"].mean().round(3)
    )
    return {
        "eventos_bitacora": len(ev),
        "eventos_detectados": int(detectados.sum()),
        "recall": round(detectados.mean(), 3),
        "episodios_alerta": len(ep),
        "episodios_con_evento": int(ep_validos.sum()),
        "precision": round(ep_validos.mean(), 3) if len(ep) else np.nan,
        "recall_por_tipo": por_tipo.to_dict(),
    }


# ---------------------------------------------------------------------------
# 6. EJECUCIÓN
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Detección de desviaciones")
    p.add_argument("--entrada", default="data/processed/indicadores_diarios.csv")
    p.add_argument("--eventos", default="data/raw/eventos.csv")
    p.add_argument("--salida", default="data/processed")
    a = p.parse_args()

    df = pd.read_csv(a.entrada, parse_dates=["fecha"])
    eventos = pd.read_csv(a.eventos, parse_dates=["fecha"])

    df = detectar(df)
    ep = episodios(df)
    metricas = evaluar(ep, eventos)

    out = Path(a.salida)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "alertas_diarias.csv", index=False)
    ep.to_csv(out / "episodios_alerta.csv", index=False)

    print(f"Días con alerta: {int(df['alerta'].sum())} de {len(df)} "
          f"({df['alerta'].mean() * 100:.1f} %)")
    for s in SENALES_DESVIACION + SENALES_RIESGO:
        print(f"  {s:<20} {int(df[s].sum()):>5} días")
    print("\n== Evaluación contra bitácora ==")
    for k, v in metricas.items():
        print(f"  {k:<22} {v}")


if __name__ == "__main__":
    main()
