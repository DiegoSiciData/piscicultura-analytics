"""
Ejecuta todo el pipeline de punta a punta con un solo comando.

Pasos (en orden): simulación → limpieza → indicadores → alertas → modelo.
Cada paso se corre como un proceso aparte; si uno falla, el pipeline se
detiene y muestra el error. Todo el detalle queda en reports/pipeline.log.

Uso:
    python src/run_pipeline.py                    # corre todo
    python src/run_pipeline.py --desde limpieza   # reutiliza data/raw existente
    python src/run_pipeline.py --seed 7 --estanques 8 --dias 365
    python src/run_pipeline.py --verbose          # muestra la salida de cada paso
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# (nombre, script, archivos que debe producir)
PASOS = [
    ("simulacion", "simulate_data.py",
     ["data/raw/ambiental.csv", "data/raw/productivo.csv", "data/raw/eventos.csv"]),
    ("limpieza", "clean.py",
     ["data/processed/dataset_integrado.csv"]),
    ("indicadores", "indicators.py",
     ["data/processed/indicadores_diarios.csv", "data/processed/resumen_lotes.csv"]),
    ("alertas", "anomalies.py",
     ["data/processed/alertas_diarias.csv", "data/processed/episodios_alerta.csv"]),
    ("modelo", "model.py",
     ["data/processed/validacion_modelos.csv", "models/modelo_mortalidad.joblib",
      "reports/figuras/importancia_variables.png"]),
]
NOMBRES = [p[0] for p in PASOS]


def ejecutar_paso(nombre, script, esperados, args_extra, log, verbose):
    """Corre un script; devuelve (ok, segundos, mensaje_de_error)."""
    t0 = time.time()
    cmd = [sys.executable, str(RAIZ / "src" / script), *args_extra]
    res = subprocess.run(cmd, cwd=RAIZ, capture_output=True, text=True)
    seg = time.time() - t0

    log.write(f"\n===== {nombre} ({script}) =====\n$ {' '.join(cmd)}\n")
    log.write(res.stdout + res.stderr)
    if verbose:
        print(res.stdout, end="")

    if res.returncode != 0:
        cola = "\n".join((res.stderr or res.stdout).strip().splitlines()[-8:])
        return False, seg, f"el script terminó con error:\n{cola}"

    faltan = [f for f in esperados if not (RAIZ / f).exists()]
    if faltan:
        return False, seg, "no se generaron: " + ", ".join(faltan)
    return True, seg, ""


def main():
    p = argparse.ArgumentParser(description="Pipeline completo de piscicultura")
    p.add_argument("--desde", choices=NOMBRES, default=NOMBRES[0],
                   help="paso desde el cual ejecutar (por defecto, desde el inicio)")
    p.add_argument("--seed", type=int, default=None, help="semilla del simulador")
    p.add_argument("--estanques", type=int, default=None, help="n° de estanques simulados")
    p.add_argument("--dias", type=int, default=None, help="días simulados")
    p.add_argument("--verbose", action="store_true", help="mostrar salida de cada paso")
    a = p.parse_args()

    args_sim = []
    for flag, valor in (("--seed", a.seed), ("--estanques", a.estanques), ("--dias", a.dias)):
        if valor is not None:
            args_sim += [flag, str(valor)]

    log_path = RAIZ / "reports" / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    pasos = PASOS[NOMBRES.index(a.desde):]
    resultados = []
    inicio = time.time()
    print(f"Pipeline iniciado: {datetime.now():%Y-%m-%d %H:%M:%S}")

    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"Pipeline iniciado: {datetime.now().isoformat(timespec='seconds')}\n")
        for i, (nombre, script, esperados) in enumerate(pasos, start=1):
            print(f"[{i}/{len(pasos)}] {nombre:<12}", end=" ", flush=True)
            extra = args_sim if nombre == "simulacion" else []
            ok, seg, error = ejecutar_paso(nombre, script, esperados, extra, log, a.verbose)
            resultados.append((nombre, ok, seg))
            print(f"{'OK' if ok else 'ERROR'}  ({seg:.1f} s)")
            if not ok:
                print(f"\nFalló el paso '{nombre}': {error}")
                print(f"Detalle completo en {log_path}")
                sys.exit(1)

    total = time.time() - inicio
    print(f"\nPipeline completado en {total:.1f} s. Log: {log_path.relative_to(RAIZ)}")
    print("Para ver el dashboard:  streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
