import re
import matplotlib.pyplot as plt

# Archivos de logs
files = {
    "gemma3_27b T=0.0": "all_usage-gemma3_27b-0.0.txt",
    "gemma3_27b T=0.2": "all_usage-gemma3_27b-0.2.txt",
    "qwen3-coder_30b T=0.0": "all_usage-qwen3-coder_30b-0.0.txt",
    "qwen3-coder_30b T=0.2": "all_usage-qwen3-coder_30b-0.2.txt",
}

# Línea tipo:
# 100 | 2.2 724956 | 4 0 57
pattern = re.compile(
    r"^\s*([\d.]+)\s*\|\s*([\d.]+)\s+(\d+)\s*\|\s*(\d+)\s+(\d+)\s+(\d+)"
)

def read_third_value(path):
    third_values = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                value = int(m.group(3))  # tercer dato (memoria en KB)
                third_values.append(value)

    if not third_values:
        return []

    # Normalizar a porcentaje (0–100%)
    max_val = max(third_values)
    third_pct = [v / max_val * 100 for v in third_values]

    return third_pct


# ---- UNA SOLA FIGURA CON TODO ----
plt.figure(figsize=(14, 6))

for label, path in files.items():
    third_pct = read_third_value(path)
    if not third_pct:
        continue

    # Dibujamos la serie de este archivo
    plt.plot(third_pct, linewidth=1, label=label)

plt.title("Uso de memoria GPU (3er dato normalizado en porcentaje)")
plt.xlabel("Muestra")
plt.ylabel("Memoria utilizada (%)")
plt.grid(True, alpha=0.3)
plt.legend(title="Modelo y temperatura", loc="lower right")
plt.tight_layout()

# Guardar TODO en una sola imagen
output_path = "uso_gpu_todos_modelos.png"
plt.savefig(output_path, dpi=300)

plt.show()
