import re
import matplotlib.pyplot as plt

# Archivos por modelo / temperatura
files = {
    "Gemma3-27B (T = 0.0)": "all_usage-gemma3_27b-0.0.txt",
    "Gemma3-27B (T = 0.2)": "all_usage-gemma3_27b-0.2.txt",
    "Qwen3Coder-30B (T = 0.0)": "all_usage-qwen3-coder_30b-0.0.txt",
    "Qwen3Coder-30B (T = 0.2)": "all_usage-qwen3-coder_30b-0.2.txt",
}

# patrón: 1er número = CPU (o carga inicial), luego el resto de columnas
pattern = re.compile(
    r"^\s*([\d.]+)\s*\|\s*([\d.]+)\s+(\d+)\s*\|\s*(\d+)\s+(\d+)\s+(\d+)"
)

def read_cpu_values(path):
    cpu_vals = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                cpu = float(m.group(1))  # primer dato = CPU
                cpu_vals.append(cpu)
    return cpu_vals

# Figura 2x2 con los cuatro modelos/temperaturas
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
axes = axes.flatten()

for ax, (label, path) in zip(axes, files.items()):
    cpu_vals = read_cpu_values(path)
    ax.plot(cpu_vals, linewidth=1)
    ax.set_title(label)
    ax.set_xlabel("Muestra (tiempo relativo)")
    ax.set_ylabel("Uso CPU (%)")
    ax.grid(True)

fig.suptitle("Evolución del uso de CPU por modelo y temperatura", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])

# Guardar imagen en PNG
plt.savefig("uso_cpu_por_modelo_y_temperatura.png", dpi=300)
plt.close()
print("Imagen guardada en: uso_cpu_por_modelo_y_temperatura.png")