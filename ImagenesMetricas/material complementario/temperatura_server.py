import re
import matplotlib.pyplot as plt

# Archivos de telemetría
files = {
    "Gemma3-27B (T = 0.0)": "all_usage-gemma3_27b-0.0.txt",
    "Gemma3-27B (T = 0.2)": "all_usage-gemma3_27b-0.2.txt",
    "Qwen3Coder-30B (T = 0.0)": "all_usage-qwen3-coder_30b-0.0.txt",
    "Qwen3Coder-30B (T = 0.2)": "all_usage-qwen3-coder_30b-0.2.txt",
}

# patrón: CPU | algo  memoria | dato3 dato4 temperatura
pattern = re.compile(
    r"^\s*([\d.]+)\s*\|\s*([\d.]+)\s+(\d+)\s*\|\s*(\d+)\s+(\d+)\s+(\d+)"
)

def read_temperature(path):
    temps = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                temp = int(m.group(6))   # último dato = temperatura
                temps.append(temp)
    return temps

# Figura 2x2
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False, sharey=True)
axes = axes.flatten()

for ax, (label, path) in zip(axes, files.items()):
    temps = read_temperature(path)
    ax.plot(temps, linewidth=1)
    ax.set_title(label)
    ax.set_xlabel("Muestra (tiempo relativo)")
    ax.set_ylabel("Temperatura (°C)")
    ax.grid(True)

fig.suptitle("Evolución de la temperatura de la GPU por modelo y temperatura",
             fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Guardar imagen
output_path = "temperatura_gpu_por_modelo_y_temperatura.png"
plt.savefig(output_path, dpi=300)
plt.close()

print("Imagen guardada en:", output_path)
