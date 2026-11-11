# Análisis y predicción en piscicultura de agua dulce

Proyecto de portafolio en Python: integración de datos ambientales y
productivos de estanques de piscicultura (salmón), indicadores operacionales,
detección de desviaciones y un modelo predictivo de mortalidad.

> **Nota:** los datos son **simulados** (`src/simulate_data.py`). Reproducen
> relaciones realistas (temperatura, oxígeno, densidad, crecimiento TGC) e
> incluyen faltantes y outliers de sensor para practicar limpieza.

## Estructura

```
data/raw/         datos crudos (ambiental, productivo, eventos)
data/processed/   datos limpios e integrados
notebooks/        exploración y análisis
src/              código del pipeline
models/           modelo entrenado
reports/figuras/  gráficos del análisis
app/              dashboard (Streamlit)
```

## Fuentes de datos

| Archivo | Contenido |
|---|---|
| `ambiental.csv` | temperatura, oxígeno, pH, amonio, caudal (diario por estanque) |
| `productivo.csv` | peces, peso, biomasa, densidad, alimento, mortalidad |
| `eventos.csv` | bitácora: fallas de aireación y brotes de enfermedad |

## Cómo usar

Ejecución rápida (corre todo el pipeline en orden):

```bash
pip install -r requirements.txt
python src/run_pipeline.py
streamlit run app/dashboard.py
```

O paso a paso:

```bash
python src/simulate_data.py        # genera data/raw/*.csv
python src/clean.py                # genera data/processed/dataset_integrado.csv
python src/indicators.py           # genera indicadores_diarios.csv y resumen_lotes.csv
python src/anomalies.py            # genera alertas_diarias.csv y episodios_alerta.csv
python src/model.py                # valida modelos y guarda models/modelo_mortalidad.joblib
streamlit run app/dashboard.py     # abre el dashboard
```

## Plan del proyecto

- [x] 1. Simulador de datos
- [x] 2. Limpieza e integración (`src/clean.py`)
- [x] 3. Indicadores: FCR, SGR, mortalidad acumulada, densidad (`src/indicators.py`)
- [x] 4. Detección de desviaciones (`src/anomalies.py`)
- [x] 5. Modelo predictivo de mortalidad, con validación temporal de ventana expansiva (`src/model.py`)
- [x] 6. Dashboard en Streamlit (`app/dashboard.py`)
- [x] 7. Automatización del pipeline (`src/run_pipeline.py`)

## Tecnologías

Python · pandas · NumPy · scikit-learn · Streamlit
