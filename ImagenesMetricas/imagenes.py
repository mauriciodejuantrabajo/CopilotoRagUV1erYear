import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------
# Config general
# ---------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
})
np.random.seed(42)

# Ruta al Excel (ajusta si lo tienes en otra carpeta)
XLS_PATH = "RAG_eval_Gemma3-27B_vs_Qwen3Coder-30B_20251205_231251.xlsx"

# Crear carpeta de salida
out_dir = Path("imagenes")
out_dir.mkdir(exist_ok=True)

# Leer todas las hojas
xls = pd.read_excel(XLS_PATH, sheet_name=None)

# ---------------------------------------------------------------------
# Definición de condiciones (coinciden con los nombres en el Excel)
# ---------------------------------------------------------------------
conditions = [
    {
        "id": "Gemma3-27B@0.0",
        "label": "gemma3 27.4B\n$t=0.0$",
        "sheet": "items_Gemma3-27B_0.0",
    },
    {
        "id": "Qwen3Coder-30B@0.0",
        "label": "qwen2.5-moe 30.5B\n$t=0.0$",
        "sheet": "items_Qwen3Coder-30B_0.0",
    },
    {
        "id": "Gemma3-27B@0.2",
        "label": "gemma3 27.4B\n$t=0.2$",
        "sheet": "items_Gemma3-27B_0.2",
    },
    {
        "id": "Qwen3Coder-30B@0.2",
        "label": "qwen2.5-moe 30.5B\n$t=0.2$",
        "sheet": "items_Qwen3Coder-30B_0.2",
    },
]

labels = [c["label"] for c in conditions]
x_pos = np.arange(len(conditions))

# ---------------------------------------------------------------------
# FIGURA 1: Barras F1 medio por condición (+ desviación estándar)
# ---------------------------------------------------------------------
means_f1 = []
stds_f1 = []

for c in conditions:
    df_items = xls[c["sheet"]]
    vals = df_items["F1"].to_numpy() * 100.0  # a porcentaje
    means_f1.append(vals.mean())
    stds_f1.append(vals.std(ddof=1))

plt.figure(figsize=(6.0, 4.0))
plt.bar(x_pos, means_f1, yerr=stds_f1, capsize=4)
plt.xticks(x_pos, labels, rotation=25, ha="right")
plt.ylabel("F1 (%)")
plt.tight_layout()
plt.savefig(out_dir / "f1_por_condicion.png", dpi=300)
plt.close()

# ---------------------------------------------------------------------
# FIGURA 2: Latencias P50 y P95 por condición (barras agrupadas)
# ---------------------------------------------------------------------
summary = xls["summary_all"].set_index("Unnamed: 0")

p50 = []
p95 = []

for c in conditions:
    row = summary.loc[c["id"]]
    p50.append(row["P50_ms"] / 1000.0)  # a segundos
    p95.append(row["P95_ms"] / 1000.0)

width = 0.35
plt.figure(figsize=(6.0, 4.0))
plt.bar(x_pos - width / 2, p50, width, label="P50")
plt.bar(x_pos + width / 2, p95, width, label="P95")
plt.xticks(x_pos, labels, rotation=25, ha="right")
plt.ylabel("Latencia end-to-end (s)")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "latencia_por_condicion.png", dpi=300)
plt.close()

# ---------------------------------------------------------------------
# FIGURA 3: Boxplot F1 por condición (item-level)
# ---------------------------------------------------------------------
data_f1 = []
for c in conditions:
    df_items = xls[c["sheet"]]
    vals = df_items["F1"].to_numpy() * 100.0  # %
    data_f1.append(vals)

plt.figure(figsize=(6.0, 4.0))
plt.boxplot(data_f1, labels=labels, showmeans=True)
plt.ylabel("F1 (%)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(out_dir / "boxplot_f1.png", dpi=300)
plt.close()

# ---------------------------------------------------------------------
# FIGURA 4: Boxplot latencia por condición (item-level)
# ---------------------------------------------------------------------
data_lat = []
for c in conditions:
    df_items = xls[c["sheet"]]
    vals = df_items["lat_ms"].to_numpy() / 1000.0  # s
    data_lat.append(vals)

plt.figure(figsize=(6.0, 4.0))
plt.boxplot(data_lat, labels=labels, showmeans=True)
plt.ylabel("Latencia end-to-end (s)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(out_dir / "boxplot_latencia.png", dpi=300)
plt.close()

# ---------------------------------------------------------------------
# FIGURA 5: Groundedness por condición
#  - Barra = media ± DE
#  - Puntos = cada respuesta (jitter), en %
# ---------------------------------------------------------------------
data_gnd = []
means_gnd = []
stds_gnd = []

for c in conditions:
    df_items = xls[c["sheet"]]
    vals = df_items["Groundedness"].to_numpy() * 100.0  # %
    data_gnd.append(vals)
    means_gnd.append(vals.mean())
    stds_gnd.append(vals.std(ddof=1))

plt.figure(figsize=(6.0, 4.0))

# Barras con error
bars = plt.bar(
    x_pos,
    means_gnd,
    yerr=stds_gnd,
    capsize=4,
    alpha=0.7,
    label="Media ± DE",
)

# Puntos jitter por respuesta
for idx, vals in enumerate(data_gnd):
    jitter = (np.random.rand(len(vals)) - 0.5) * 0.18
    plt.scatter(
        np.full_like(vals, x_pos[idx], dtype=float) + jitter,
        vals,
        s=14,
        alpha=0.55,
    )

plt.xticks(x_pos, labels, rotation=25, ha="right")
plt.ylabel("Groundedness por respuesta (%)")
plt.ylim(95, 101.5)  # zoom para ver bien 97–100 %
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(out_dir / "groundedness_por_condicion.png", dpi=300)
plt.close()

print("Listo. Figuras guardadas en la carpeta 'imagenes'.")
