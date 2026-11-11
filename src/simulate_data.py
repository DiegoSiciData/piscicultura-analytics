"""
Simulador de datos de una piscicultura de agua dulce (salmón).

Genera tres fuentes separadas, como ocurriría en una empresa real:
  - ambiental.csv  : sensores (temperatura, oxígeno, pH, amonio, caudal)
  - productivo.csv : peces, peso, biomasa, alimento, mortalidad
  - eventos.csv    : bitácora (fallas de aireación, brotes de enfermedad)

IMPORTANTE: los datos son SIMULADOS, no reales. Incluyen faltantes y
outliers de sensor a propósito, para practicar limpieza e integración.

Uso:
    python src/simulate_data.py
    python src/simulate_data.py --estanques 8 --dias 365 --seed 7
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CICLO_DIAS = 200  # duración de un lote en el estanque
ETAPAS = [(0, 60, "alevin"), (60, 150, "parr"), (150, CICLO_DIAS + 1, "smolt")]
MORT_BASE = {"alevin": 3e-4, "parr": 1.5e-4, "smolt": 8e-5}  # tasa diaria
TGC = 2.2  # coeficiente de crecimiento térmico (agua dulce)


def etapa(edad: int) -> str:
    for lo, hi, nombre in ETAPAS:
        if lo <= edad < hi:
            return nombre
    return "smolt"


def simular(n_estanques=12, dias=540, inicio="2025-01-01", seed=42):
    rng = np.random.default_rng(seed)
    fechas = pd.date_range(inicio, periods=dias, freq="D")
    amb, prod, eventos = [], [], []

    for e in range(1, n_estanques + 1):
        estanque = f"E{e:02d}"
        volumen = rng.uniform(250, 400)        # m3
        sesgo_t = rng.normal(0, 0.3)           # sesgo térmico del estanque
        dia_inicio = int(rng.integers(0, 60))  # estanques desfasados
        n_lote = 0
        peces, raiz = 0, 1.0                   # raiz = peso^(1/3), modelo TGC
        oxi_prev = 10.0
        brote, falla = 0, 0                    # días restantes de cada evento

        for d, fecha in enumerate(fechas):
            if d < dia_inicio:
                continue
            edad = (d - dia_inicio) % CICLO_DIAS
            if edad == 0:  # nuevo lote
                n_lote += 1
                peces = int(rng.integers(40_000, 60_000))
                raiz = rng.normal(1.0, 0.05)
            lote = f"{estanque}-L{n_lote}"

            # --- ambiente ---
            doy = fecha.dayofyear
            temp = (10.5 + 3.0 * np.cos(2 * np.pi * (doy - 40) / 365)
                    + sesgo_t + rng.normal(0, 0.3))
            peso = raiz ** 3
            biomasa = peces * peso / 1000
            densidad = biomasa / volumen

            # --- eventos aleatorios ---
            if falla == 0 and rng.random() < 0.004:
                falla = int(rng.integers(1, 4))
                eventos.append((fecha, estanque, "falla_aireacion", falla))
            if brote == 0 and rng.random() < 0.002:
                brote = int(rng.integers(5, 11))
                eventos.append((fecha, estanque, "brote_enfermedad", brote))

            oxi = 11.0 - 0.3 * (temp - 10) - 0.05 * densidad + rng.normal(0, 0.3)
            if falla > 0:
                oxi -= rng.uniform(2.5, 4.0)
                falla -= 1
            oxi = max(oxi, 1.0)
            caudal = max(volumen * 0.6 + rng.normal(0, 10), 50)  # L/s
            ph = rng.normal(6.9, 0.15)

            # --- alimentación ---
            pct = np.clip(4.4 * peso ** -0.25 * (temp / 11), 0.5, 5.0) / 100
            alimento = biomasa * pct
            if oxi < 6.5:  # se reduce la ración ante estrés
                alimento *= 0.6
            amonio = 0.03 + 0.0012 * alimento * (200 / caudal) + rng.normal(0, 0.01)

            # --- mortalidad (depende del estrés del día anterior) ---
            estres = (max(0, 7.5 - oxi_prev) * 4e-4
                      + max(0, temp - 14) * 1e-4
                      + max(0, densidad - 40) * 2e-5)
            tasa = MORT_BASE[etapa(edad)] + estres
            if brote > 0:
                tasa *= rng.uniform(5, 15)
            muertos = int(rng.binomial(peces, min(tasa, 0.2)))

            amb.append(dict(
                fecha=fecha, estanque=estanque,
                temperatura_c=temp, oxigeno_mgl=oxi, ph=ph,
                amonio_mgl=max(amonio, 0), caudal_ls=caudal))
            prod.append(dict(
                fecha=fecha, estanque=estanque, lote=lote, etapa=etapa(edad),
                edad_dias=edad, peces=peces,
                peso_prom_g=peso * rng.normal(1, 0.01),
                biomasa_kg=biomasa, densidad_kgm3=densidad,
                alimento_kg=alimento, mortalidad_n=muertos))

            # --- actualizar estado ---
            factor = 1.0
            if oxi < 6.5:
                factor *= 0.7
            if brote > 0:
                factor *= 0.6
                brote -= 1
            raiz += TGC * max(temp, 0) * factor / 1000
            peces -= muertos
            oxi_prev = oxi

    ambiental = pd.DataFrame(amb)
    productivo = pd.DataFrame(prod)
    ev = pd.DataFrame(eventos, columns=["fecha", "estanque", "tipo", "duracion_dias"])
    return ambiental, productivo, ev


def ensuciar(ambiental: pd.DataFrame, rng) -> pd.DataFrame:
    """Agrega faltantes y outliers de sensor (como pasa en la realidad)."""
    df = ambiental.copy()
    n = len(df)
    for col in ["temperatura_c", "oxigeno_mgl", "ph"]:
        idx = rng.choice(n, int(n * 0.015), replace=False)
        df.loc[df.index[idx], col] = np.nan
    idx = rng.choice(n, int(n * 0.003), replace=False)
    df.loc[df.index[idx], "oxigeno_mgl"] = rng.choice([0.0, 25.0, -1.0], size=len(idx))
    idx = rng.choice(n, int(n * 0.002), replace=False)
    df.loc[df.index[idx], "temperatura_c"] = rng.choice([-5.0, 45.0], size=len(idx))
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--estanques", type=int, default=12)
    p.add_argument("--dias", type=int, default=540)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--salida", default="data/raw")
    a = p.parse_args()

    rng = np.random.default_rng(a.seed + 1)
    ambiental, productivo, eventos = simular(a.estanques, a.dias, seed=a.seed)
    ambiental = ensuciar(ambiental, rng)
    cols_a = ambiental.select_dtypes("float").columns
    cols_p = productivo.select_dtypes("float").columns
    ambiental[cols_a] = ambiental[cols_a].round(3)
    productivo[cols_p] = productivo[cols_p].round(3)

    out = Path(a.salida)
    out.mkdir(parents=True, exist_ok=True)
    ambiental.to_csv(out / "ambiental.csv", index=False)
    productivo.to_csv(out / "productivo.csv", index=False)
    eventos.to_csv(out / "eventos.csv", index=False)

    print(f"ambiental.csv  : {len(ambiental):>6} filas")
    print(f"productivo.csv : {len(productivo):>6} filas")
    print(f"eventos.csv    : {len(eventos):>6} filas")
    print(f"Guardado en {out.resolve()}")


if __name__ == "__main__":
    main()
