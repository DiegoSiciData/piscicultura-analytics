"""
Lógica de datos del dashboard (sin Streamlit, para poder probarla aparte).

Funciones para: cargar los resultados del pipeline, filtrar por estanque y
periodo, calcular KPIs y generar la predicción de riesgo con el modelo guardado.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.model import construir_variables

RAIZ = Path(__file__).resolve().parents[1]
ARCHIVOS = {
    "diario": RAIZ / "data/processed/alertas_diarias.csv",
    "lotes": RAIZ / "data/processed/resumen_lotes.csv",
    "episodios": RAIZ / "data/processed/episodios_alerta.csv",
    "validacion": RAIZ / "data/processed/validacion_modelos.csv",
    "modelo": RAIZ / "models/modelo_mortalidad.joblib",
    "figura": RAIZ / "reports/figuras/importancia_variables.png",
}

# Nivel de riesgo: mortalidad prevista vs. la mediana histórica de su etapa
RAZON_MEDIO = 1.5
RAZON_ALTO = 3.0


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def archivos_faltantes() -> list[str]:
    return [nombre for nombre, ruta in ARCHIVOS.items() if not ruta.exists()]


def cargar_diario() -> pd.DataFrame:
    return pd.read_csv(ARCHIVOS["diario"], parse_dates=["fecha"])


def cargar_lotes() -> pd.DataFrame:
    return pd.read_csv(ARCHIVOS["lotes"], parse_dates=["inicio", "fin"])


def cargar_episodios() -> pd.DataFrame:
    return pd.read_csv(ARCHIVOS["episodios"], parse_dates=["inicio", "fin"])


def cargar_validacion() -> pd.DataFrame:
    """Promedio de las métricas de validación temporal por modelo."""
    res = pd.read_csv(ARCHIVOS["validacion"])
    return (res.groupby("modelo")[["MAE", "RMSE", "R2"]].mean()
               .sort_values("MAE").round(4).reset_index())


def cargar_modelo() -> dict:
    return joblib.load(ARCHIVOS["modelo"])


# ---------------------------------------------------------------------------
# Filtros y KPIs
# ---------------------------------------------------------------------------
def filtrar_diario(df, estanque, desde, hasta) -> pd.DataFrame:
    mask = (df["estanque"] == estanque) & df["fecha"].between(desde, hasta)
    return df[mask].sort_values("fecha")


def filtrar_por_solape(df, desde, hasta, estanque=None) -> pd.DataFrame:
    """Filtra lotes o episodios (con columnas inicio/fin) que tocan el periodo."""
    mask = (df["inicio"] <= hasta) & (df["fin"] >= desde)
    if estanque is not None:
        mask &= df["estanque"] == estanque
    return df[mask]


def kpis_lotes(lotes: pd.DataFrame) -> dict | None:
    """KPIs promedio de lotes completos; None si no hay ninguno."""
    comp = lotes[lotes["lote_completo"]]
    if comp.empty:
        return None
    return {
        "lotes_completos": len(comp),
        "mortalidad_pct": comp["mortalidad_pct"].mean(),
        "fcr": comp["fcr_global"].mean(),
        "sgr": comp["sgr_global_pct_dia"].mean(),
        "peso_final_g": comp["peso_final_g"].mean(),
    }


# ---------------------------------------------------------------------------
# Predicción de riesgo
# ---------------------------------------------------------------------------
def predecir_riesgo(diario: pd.DataFrame, artefacto: dict) -> pd.DataFrame:
    """
    Predice la mortalidad del día siguiente para cada fila (estanque, día).
    La predicción hecha el día t corresponde al día t+1.
    """
    df = construir_variables(diario)
    df["mortalidad_prevista_pct"] = artefacto["modelo"].predict(df[artefacto["columnas"]])

    # Sin historia suficiente (primeros días del lote) no se muestra predicción
    sin_historia = df["mortalidad_lag3"].isna() | df["mortalidad_media_7d"].isna()
    df.loc[sin_historia, "mortalidad_prevista_pct"] = np.nan

    referencia = df.groupby("etapa")["mortalidad_dia_pct"].transform("median")
    df["razon_riesgo"] = df["mortalidad_prevista_pct"] / referencia.replace(0, np.nan)
    df["nivel_riesgo"] = pd.cut(
        df["razon_riesgo"], [-np.inf, RAZON_MEDIO, RAZON_ALTO, np.inf],
        labels=["Normal", "Medio", "Alto"])
    cols = ["fecha", "estanque", "lote", "etapa", "mortalidad_dia_pct",
            "mortalidad_prevista_pct", "razon_riesgo", "nivel_riesgo"]
    return df[cols].sort_values(["estanque", "fecha"]).reset_index(drop=True)
