import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# --------------------------------------------------------------------
# Configuración de salida
# --------------------------------------------------------------------
output_dir = "imagenes"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "arbol_tematico_resultados.png")

# --------------------------------------------------------------------
# Funciones auxiliares
# --------------------------------------------------------------------
def draw_box(ax, center_x, center_y, width, height, text,
             fontsize=8, weight="normal"):
    """
    Dibuja una caja redondeada centrada en (center_x, center_y)
    usando coordenadas normalizadas (0–1).
    """
    x = center_x - width / 2
    y = center_y - height / 2

    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.03",
        linewidth=1.0
    )
    ax.add_patch(box)

    ax.text(
        center_x,
        center_y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        wrap=True
    )

def draw_arrow(ax, x0, y0, x1, y1):
    """
    Flecha simple desde (x0, y0) hasta (x1, y1).
    """
    arrow = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="->",
        linewidth=1.0,
        mutation_scale=10
    )
    ax.add_patch(arrow)

# --------------------------------------------------------------------
# Figura y eje
# --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# --------------------------------------------------------------------
# Nodo raíz
# --------------------------------------------------------------------
root_x, root_y = 0.5, 0.86
root_width, root_height = 0.82, 0.14

root_text = (
    "Desempeño del copiloto RAG\n"
    "(consultas de primer año UV)"
)

draw_box(
    ax,
    root_x,
    root_y,
    root_width,
    root_height,
    root_text,
    fontsize=10,
    weight="bold"
)

# --------------------------------------------------------------------
# Nodos intermedios (más separados)
# --------------------------------------------------------------------
mid_y = 0.44          # un poco más abajo
mid_width = 0.24      # más angostas -> más separación
mid_height = 0.28

# 1) Precisión y completitud
x1 = 0.16
text1 = (
    "Precisión y completitud\n"
    "de la respuesta (H1)\n\n"
    "- Respuesta directa vs. perifrástica\n"
    "- Detalle procedimental\n"
    "- Cobertura de casos especiales"
)

draw_box(
    ax,
    x1,
    mid_y,
    mid_width,
    mid_height,
    text1,
    fontsize=7.5
)
draw_arrow(ax, root_x, root_y - root_height / 2, x1, mid_y + mid_height / 2)

# 2) Uso de citas normativas y groundedness
x2 = 0.5
text2 = (
    "Uso de citas normativas\n"
    "y groundedness (H2)\n\n"
    "- Citas mínimas suficientes\n"
    "- Citas múltiples integradas\n"
    "- Citas redundantes"
)

draw_box(
    ax,
    x2,
    mid_y,
    mid_width,
    mid_height,
    text2,
    fontsize=7.5
)
draw_arrow(ax, root_x, root_y - root_height / 2, x2, mid_y + mid_height / 2)

# 3) Experiencia percibida de uso y estilo
x3 = 0.84
text3 = (
    "Experiencia percibida\n"
    "de uso y estilo (H3–H4)\n\n"
    "- Tono explicativo vs. telegráfico\n"
    "- Organización de la información\n"
    "- Sensación de espera vs. inmediatez"
)

draw_box(
    ax,
    x3,
    mid_y,
    mid_width,
    mid_height,
    text3,
    fontsize=7.5
)
draw_arrow(ax, root_x, root_y - root_height / 2, x3, mid_y + mid_height / 2)

# --------------------------------------------------------------------
# Etiqueta inferior opcional
# --------------------------------------------------------------------
ax.text(
    0.5,
    0.06,
    "Árbol temático del desempeño cualitativo del copiloto RAG",
    ha="center",
    va="center",
    fontsize=8
)

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Árbol temático guardado en: {output_path}")
