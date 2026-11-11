"""
Dashboard de monitoreo de piscicultura (Streamlit).

Ejecutar desde la carpeta raíz del proyecto:
    streamlit run app/dashboard.py

Requiere haber corrido antes el pipeline (ver README).
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import dashboard_data as dd  # noqa: E402
from src.anomalies import UMBRALES  # noqa: E402

st.set_page_config(page_title="Piscicultura · Monitoreo", page_icon="🐟", layout="wide")
st.title("🐟 Monitoreo de piscicultura de agua dulce")
st.caption("Datos simulados · limpieza → indicadores → alertas → modelo predictivo")

# ---------------------------------------------------------------------------
# Verificación de archivos
# ---------------------------------------------------------------------------
faltan = dd.archivos_faltantes()
if faltan:
    st.error("Faltan resultados del pipeline: " + ", ".join(faltan))
    st.code(
        "python src/simulate_data.py\n"
        "python src/clean.py\n"
        "python src/indicators.py\n"
        "python src/anomalies.py\n"
        "python src/model.py"
    )
    st.stop()


# ---------------------------------------------------------------------------
# Carga (con caché para que no se relea en cada interacción)
# ---------------------------------------------------------------------------
@st.cache_data
def cargar_todo():
    return (dd.cargar_diario(), dd.cargar_lotes(),
            dd.cargar_episodios(), dd.cargar_validacion())


@st.cache_data
def calcular_riesgo():
    return dd.predecir_riesgo(dd.cargar_diario(), dd.cargar_modelo())


diario, lotes, episodios, validacion = cargar_todo()
riesgo = calcular_riesgo()

# ---------------------------------------------------------------------------
# Filtros (barra lateral)
# ---------------------------------------------------------------------------
st.sidebar.header("Filtros")
estanque = st.sidebar.selectbox("Estanque", sorted(diario["estanque"].unique()))
f_min, f_max = diario["fecha"].min().date(), diario["fecha"].max().date()
rango = st.sidebar.date_input("Periodo", value=(f_min, f_max),
                              min_value=f_min, max_value=f_max)
if len(rango) != 2:
    st.info("Selecciona fecha de inicio y de término en la barra lateral.")
    st.stop()
desde, hasta = pd.Timestamp(rango[0]), pd.Timestamp(rango[1])

d = dd.filtrar_diario(diario, estanque, desde, hasta)
if d.empty:
    st.warning("No hay datos para ese estanque y periodo.")
    st.stop()

tab_resumen, tab_estanque, tab_alertas, tab_pred = st.tabs(
    ["Resumen", "Estanque", "Alertas", "Predicción"])

# ---------------------------------------------------------------------------
# Pestaña 1: resumen de lotes
# ---------------------------------------------------------------------------
with tab_resumen:
    st.subheader("Resultados por lote")
    lotes_periodo = dd.filtrar_por_solape(lotes, desde, hasta)
    k = dd.kpis_lotes(lotes_periodo)
    if k is None:
        st.info("No hay lotes completos en el periodo seleccionado (todos los estanques).")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Lotes completos", k["lotes_completos"])
        c2.metric("Mortalidad media", f"{k['mortalidad_pct']:.2f} %")
        c3.metric("FCR medio", f"{k['fcr']:.2f}")
        c4.metric("Peso final medio", f"{k['peso_final_g']:.0f} g")
        st.caption("KPIs de lotes completos (200 días) de todos los estanques en el periodo.")

    lotes_est = dd.filtrar_por_solape(lotes, desde, hasta, estanque)
    st.markdown(f"**Lotes del estanque {estanque}**")
    st.dataframe(
        lotes_est[["lote", "inicio", "fin", "dias", "mortalidad_pct", "sgr_global_pct_dia",
                   "fcr_global", "peso_final_g", "lote_completo"]],
        hide_index=True)
    if not lotes_est.empty:
        st.bar_chart(lotes_est.set_index("lote")["mortalidad_pct"])

# ---------------------------------------------------------------------------
# Pestaña 2: series del estanque
# ---------------------------------------------------------------------------
with tab_estanque:
    st.subheader(f"Estanque {estanque}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temperatura media", f"{d['temperatura_c'].mean():.1f} °C")
    c2.metric("Oxígeno medio", f"{d['oxigeno_mgl'].mean():.1f} mg/L")
    c3.metric("Peces muertos", f"{int(d['mortalidad_n'].sum()):,}")
    c4.metric("Días con alerta", int(d["alerta"].sum()))

    serie = d.set_index("fecha")
    st.markdown("**Oxígeno disuelto (mg/L)**")
    oxi = serie[["oxigeno_mgl"]].copy()
    oxi["umbral_minimo"] = UMBRALES["oxigeno_min_mgl"]
    st.line_chart(oxi)
    st.markdown("**Temperatura (°C)**")
    st.line_chart(serie[["temperatura_c"]])
    st.markdown("**Mortalidad diaria (%)**")
    st.line_chart(serie[["mortalidad_dia_pct"]])
    st.markdown("**FCR y SGR (ventana móvil de 14 días)**")
    st.line_chart(serie[["fcr", "sgr_pct_dia"]])

# ---------------------------------------------------------------------------
# Pestaña 3: alertas
# ---------------------------------------------------------------------------
with tab_alertas:
    st.subheader("Episodios de alerta")
    todos = st.checkbox("Ver todos los estanques", value=False)
    ep = dd.filtrar_por_solape(episodios, desde, hasta, None if todos else estanque)
    st.metric("Episodios en el periodo", len(ep))
    if ep.empty:
        st.success("Sin episodios de alerta en el periodo.")
    else:
        st.dataframe(
            ep[["estanque", "inicio", "fin", "dias_alerta", "oxigeno_min",
                "mortalidad_max_pct", "senales"]],
            hide_index=True)
    st.caption("Señales: reglas de oxígeno, caída de oxígeno, mortalidad sostenida y FCR atípico.")

# ---------------------------------------------------------------------------
# Pestaña 4: predicción
# ---------------------------------------------------------------------------
with tab_pred:
    st.subheader("Mortalidad prevista para el día siguiente")
    r = riesgo[(riesgo["estanque"] == estanque)
               & riesgo["fecha"].between(desde, hasta)].dropna(subset=["mortalidad_prevista_pct"])
    if r.empty:
        st.info("Sin predicciones en este periodo (faltan días de historia del lote).")
    else:
        ultimo = r.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Último día evaluado", ultimo["fecha"].strftime("%Y-%m-%d"))
        c2.metric("Mortalidad prevista", f"{ultimo['mortalidad_prevista_pct']:.4f} %")
        c3.metric("Nivel de riesgo", str(ultimo["nivel_riesgo"]))
        st.line_chart(r.set_index("fecha")[["mortalidad_dia_pct", "mortalidad_prevista_pct"]])
        dias_alto = int((r["nivel_riesgo"] == "Alto").sum())
        st.caption(f"Días con riesgo alto en el periodo: {dias_alto}. "
                   "Nivel según la razón entre lo previsto y la mortalidad típica de la etapa.")

    st.warning(
        "Este modelo se entrenó con todo el histórico, por lo que las curvas de arriba "
        "son ilustrativas. La evaluación real es la validación temporal de abajo.")
    st.markdown("**Validación temporal (promedio de 4 tramos)**")
    st.dataframe(validacion, hide_index=True)
    st.image(str(dd.ARCHIVOS["figura"]), caption="Variables más importantes del modelo")
