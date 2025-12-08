import json
import matplotlib.pyplot as plt

# Archivos JSON entregados
files = {
    "Gemma3-27B (t = 0.0)": "gemma327b-0.0.json",
    "Gemma3-27B (t = 0.2)": "gemma327b-0.2.json",
    "Qwen3Coder-30B (t = 0.0)": "qwen3coder30b-0.0.json",
    "Qwen3Coder-30B (t = 0.2)": "qwen3coder30b-0.2.json"
}

def load_latencies(path):
    """Carga lat_ms desde el JSON y lo transforma a segundos."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lat_ms = [item["lat_ms"] for item in data]
    lat_sec = [v / 1000 for v in lat_ms]  # convertir a segundos
    return lat_sec

# Crear figura grande con 4 subplots
plt.figure(figsize=(14, 10))

for i, (label, path) in enumerate(files.items(), start=1):
    lat_sec = load_latencies(path)

    plt.subplot(2, 2, i)
    plt.plot(lat_sec, marker="o", markersize=3, linewidth=1)
    plt.title(label)
    plt.xlabel("Pregunta")
    plt.ylabel("Latencia (segundos)")
    plt.grid(True)

plt.tight_layout()

# Guardar imagen
output_path = "latencias_por_pregunta_por_modelo.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Imagen guardada en: {output_path}")
