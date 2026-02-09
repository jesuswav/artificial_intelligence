---
marp: true
theme: default
paginate: true
title: "Optimización Combinatoria"
---

# 🔷 Optimización Combinatoria

**Autor:**  
*Jesús Emmanuel Mmartínez García*  
**Tema:** Introducción, técnicas y áreas de aplicación  

---

# 🧩 ¿Qué es la Optimización Combinatoria?

> Rama de la **optimización matemática** que busca la **mejor combinación o selección** posible dentro de un conjunto finito pero grande de soluciones.

**Características:**
- Las soluciones son **discretas o enteras**.  
- Los problemas suelen ser **NP-duros**.  
- El número de soluciones crece **factorial o exponencialmente**.

---

# 🔹 Ejemplos de Problemas Combinatorios

| Problema | Descripción | Tipo |
|-----------|--------------|------|
| Viajante (TSP) | Ruta mínima que visita todas las ciudades una vez | Ruta / Permutación |
| Mochila (Knapsack) | Seleccionar objetos maximizando valor y sin exceder peso | Selección |
| Asignación | Asignar tareas a trabajadores minimizando coste | Emparejamiento |
| Coloración de grafos | Asignar colores a nodos sin que los adyacentes coincidan | Grafos |
| Ruteo de vehículos | Optimizar rutas múltiples con capacidad limitada | Logística |

---

# ⚙️ Elementos clave

1. **Conjunto de soluciones posibles**  
   - Finito pero enorme.  
2. **Función objetivo**  
   - Minimizar o maximizar un valor (coste, tiempo, distancia…).  
3. **Restricciones**  
   - Condiciones que deben cumplirse (capacidad, nodos visitados, etc.).  

---

# 🧮 Métodos para resolver problemas combinatorios

### 🔸 1. Métodos Exactos
Buscan **el óptimo garantizado**, pero son lentos para problemas grandes.

**Ejemplos:**
- Programación Lineal Entera (PLE)  
- Branch and Bound  
- Branch and Cut  
- Programación Dinámica (Held–Karp)

---

# ⚡ Métodos Aproximados

Cuando el tamaño del problema es grande, se usan métodos que **no garantizan el óptimo**, pero **proporcionan soluciones muy buenas en menos tiempo**.

Se dividen en:
- 🔹 **Heurísticas**
- 🔹 **Metaheurísticas**

---

# 🔹 Heurísticas

> Estrategias simples que construyen soluciones **buenas** (aunque no perfectas).

**Ejemplos:**
- **Vecino más cercano** (TSP)  
- **Greedy / Voraz** (mochila)  
- **Inserción más barata**  
- **Búsqueda local**

---

# 🔹 Metaheurísticas

> Algoritmos **más generales y potentes** que guían la búsqueda global, evitan óptimos locales y mejoran soluciones heurísticas.

---

**Ejemplos principales:**

| Metaheurística | Inspiración | Tipo |
|-----------------|--------------|------|
| 🔥 Recocido simulado (Simulated Annealing) | Física (enfriamiento del metal) | Búsqueda local probabilística |
| 🧬 Algoritmos genéticos | Evolución biológica | Poblacional |
| 🐜 Colonia de hormigas | Comportamiento social | Poblacional |
| 🕵️‍♂️ Búsqueda Tabú | Memoria adaptativa | Búsqueda local |
| 🕊️ PSO (Optimización por enjambre de partículas) | Movimiento de partículas | Poblacional |

---

# 🌍 Áreas de aplicación

✅ Logística y transporte  
✅ Planificación y asignación de recursos  
✅ Telecomunicaciones  
✅ Diseño de redes  
✅ Bioinformática  
✅ Producción y manufactura  
✅ Finanzas (carteras óptimas)

---

# 💡 Claves para dominar la Optimización Combinatoria

1. Comprender la **naturaleza discreta** de los problemas.  
2. Aprender a **formular matemáticamente** el problema (variables, restricciones, función objetivo).  
3. Conocer los **métodos exactos** y sus límites.  
4. Experimentar con **metaheurísticas** y ajustes de parámetros.  
5. Evaluar soluciones con **métricas de calidad y tiempo**.  

---

# 🔚 Conclusión

> La optimización combinatoria es el corazón de muchos problemas reales de decisión y planificación.

- Permite **ahorrar recursos y mejorar eficiencia**.  
- Une **matemática, computación e inteligencia artificial**.  
- Las **metaheurísticas** hacen posible resolver problemas imposibles para los métodos exactos.

---

# 📘 Referencias sugeridas

- Papadimitriou, C., & Steiglitz, K. (1998). *Combinatorial Optimization: Algorithms and Complexity.*  
- Glover, F., & Kochenberger, G. (2003). *Handbook of Metaheuristics.*  
- Aarts, E., & Lenstra, J. K. (2003). *Local Search in Combinatorial Optimization.*