---
marp: true
paginate: true
theme: default
class: lead
---

# Introducción a la Inteligencia Artificial
## Clase muestra - Introducción
🧑🏻‍🏫 Jesus Emmanuel Martínez García

---

# 📚 Conceptos básicos en tecnología y programación
- ¿Qué es programar?
- ¿Qué es un lenguaje de programación?
- ¿Qué es Python?
- ¿Qué es un script?
- ¿Qué es una librería?
- ¿Qué es ejecutar código?
- ¿Qué es un dataset?

---

# 📚 Objetivos
- Entender fundamentos basicos de:
    - Programación
    - Inteligencía Artificial
    - Ciencia de datos
- Crear un ejemplo funcional básico aplicado a la realidad

---

# 📚 ¿Qué es la IA?
- Sistemas que realizan tareas que normalmente requieren inteligencia humana.
- No es magia: es **matemática + datos + reglas**.
- Hoy veremos una IA sencilla y visual: **un sistema de recomendación**.

---

# 📚 ¿Qué es un sistema de recomendación?
- Tecnologías que sugieren elementos basados en similitudes.
- Ejemplos:
  - Netflix → películas
  - Spotify → canciones
  - Amazon → productos
- Hoy construiremos uno que recomienda **películas**.

---

# 📚 ¿Cómo funcionan?
### Enfoque simple: Similitud
1. Cada película se representa como un vector:
[acción, romance, comedia]
2. Cada película es un punto en un espacio 3D.  
3. Las películas **cercanas** entre sí son **similares**.  
4. Recomendamos la más cercana.

---

# 📚 ¿Por qué funciona?
- La IA compara características numéricas.  
- Si dos películas se parecen, sus vectores son similares.  
- Matemáticamente medimos esto con **distancia euclidiana**.

---

# 📚 Visualización en 3D
- Usamos **Plotly** para crear un mapa 3D interactivo.  
- Cada punto = una película.  
- El usuario elige una → la IA marca la más parecida.

---

# 💻 Código (1/3)
### Dataset de películas
```python
movies = {
    "Fast & Furious": [9,1,2],
    "Titanic": [2,9,1],
    "Deadpool": [8,3,8],
    "Toy Story": [3,2,9],
    "The Notebook": [1,10,2],
    "Avengers": [10,2,5],
    "Finding Nemo": [2,3,9],
    "John Wick": [10,1,2],
    "La La Land": [2,8,3]
} 
```

---

# 💻 Código (2/3)
### Dataset de películas
```python
movies = {
    "Fast & Furious": [9,1,2],
    "Titanic": [2,9,1],
    "Deadpool": [8,3,8],
    "Toy Story": [3,2,9],
    "The Notebook": [1,10,2],
    "Avengers": [10,2,5],
    "Finding Nemo": [2,3,9],
    "John Wick": [10,1,2],
    "La La Land": [2,8,3]
} 
```

---

# 💻 Código (3/3)
### Función de recomendación
```python
from math import dist
def recommend(movie_name, titles, coords):
    idx = titles.index(movie_name)
    target = coords[idx]
    best = None
    best_d = float("inf")

    for i, other in enumerate(coords):
        if i == idx:
            continue
        d = dist(target, other)
        if d < best_d:
            best_d = d
            best = titles[i]
    return best
```


---

# Código (2/3)
### Gráfico 3D interactivo con Plotly
```python
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers+text',
    marker=dict(size=7, color=colors),
    text=titles,
    textposition="top center"
))

fig.show()
```

---

# ¿Qué aprendimos?
- La IA puede ser sencilla y visual.
- Los sistemas de recomendación usan similitud.
- Podemos representar datos como puntos en 3D.
- Podemos hacer IA sin redes neuronales complejas.

---

# Materíales recomendados:
- Curso de Fundamentos de ingeniería de Software - Platzi:
![bg height:350px right:50%](./Fundamentos_QR.png)

---

# Gracias !
Sección de preguntas y respuestas...