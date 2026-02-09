---
marp: true
theme: default
paginate: true
class: lead
---

# 🧠 Metaheurísticas

**Desde su origen, concepto y principales algoritmos**
**Jesús Emmanuel Martínez García**

---

## 📘 Introducción

- Las **metaheurísticas** son estrategias de optimización de alto nivel.  
- Diseñadas para **guiar heurísticas** hacia soluciones óptimas o casi óptimas.  
- Propuesto el término por **Fred Glover (1986)**.  
- Usadas para resolver problemas **complejos y no lineales**.

<!-- Imagen sugerida: comparación entre búsqueda exacta y heurística (exploratoria vs determinista) -->

---

## 🎯 Motivación

- Resolver problemas donde los métodos exactos son inviables por tiempo o costo.  
- Aplicadas en:
  - Logística  
  - Ingeniería  
  - Inteligencia Artificial  
  - Planificación y diseño

---

## ⚙️ ¿Qué son las metaheurísticas?

- Procedimientos **generales de optimización** aplicables a muchos tipos de problemas.  
- No garantizan la solución óptima, pero buscan **buenas soluciones en tiempo razonable**.  
- Inspiradas en:
  - Procesos naturales (evolución, física, comportamiento animal).  
  - Estrategias probabilísticas y matemáticas.

---

## 📈 Ejemplo visual

<!-- Imagen sugerida: superficie con múltiples mínimos locales y un algoritmo buscando el óptimo global -->

- Las metaheurísticas permiten **escapar de mínimos locales**.
- Combinan **exploración** (buscar nuevas zonas) y **explotación** (mejorar soluciones existentes).

---

## 🧩 Definición formal

Sea un problema de optimización:

- Minimizar o maximizar una función objetivo `f(x)`  
- Sujeto a restricciones `x ∈ S`  

**Metaheurística:** busca `x*` tal que `f(x*) ≈ f(opt)` en **tiempo computacional reducido**.

---

## 🧱 Características principales

- Independientes del tipo de problema.  
- No requieren derivadas ni gradientes.  
- Incorporan aleatoriedad.  
- Manejan espacios no convexos o discontinuos.  
- Permiten escapar de óptimos locales.  
- Requieren una **función objetivo clara**.

---

## 🧮 Clasificación general

**1️⃣ Basadas en trayectorias (Trajectory-based):**
- Exploran una sola solución y la modifican iterativamente.  
- Ejemplo: **Temple Simulado**, **Búsqueda Tabú**.

**2️⃣ Basadas en poblaciones (Population-based):**
- Mantienen un conjunto de soluciones.  
- Ejemplo: **Algoritmos Genéticos**, **PSO**, **ACO**.

---

# 🔥 Temple Simulado  
### (*Simulated Annealing*)

---

## 🌍 Origen

- Inspirado en el **proceso físico del recocido del metal**.  
- Propuesto por **Kirkpatrick, Gelatt y Vecchi (1983)**.  
- Imitación del enfriamiento lento para alcanzar una estructura estable.

<!-- Imagen sugerida: representación del enfriamiento de un metal -->

---

## ⚙️ Principio básico

- Parte de una **solución inicial**.  
- En cada iteración:
  - Genera una solución **vecina**.  
  - Si mejora → se acepta.  
  - Si empeora → puede aceptarse con **cierta probabilidad** que depende de la **temperatura**.

---

## 🧮 Fórmula de aceptación

Una solución peor `x'` se acepta con probabilidad:

`P(aceptar) = exp(-(ΔE) / T)`

Donde:  
- `ΔE = f(x') - f(x)`  
- `T` = temperatura actual.  

A menor temperatura → menor probabilidad de aceptar soluciones peores.

---

## 🔁 Pseudocódigo

1. Inicializar `T = T0`  
2. Elegir solución inicial `x`  
3. Mientras `T > Tmin`:
   - Generar solución vecina `x'`  
   - Calcular `ΔE = f(x') - f(x)`  
   - Si `ΔE < 0`, aceptar `x'`  
   - Si `ΔE > 0`, aceptar con `exp(-ΔE / T)`  
   - Reducir `T` gradualmente  
4. Devolver la mejor solución encontrada

---

## ⚖️ Ventajas y desventajas

**Ventajas:**
- Escapa de óptimos locales.  
- Fácil de implementar.  
- Requiere pocos parámetros.  

**Desventajas:**
- Alta dependencia del **plan de enfriamiento**.  
- Puede ser **lento** en espacios grandes.

<!-- Imagen sugerida: gráfico mostrando descenso de temperatura y convergencia de energía -->

---

# 🔒 Búsqueda Tabú  
### (*Tabu Search*)

---

## 📘 Contexto histórico

- Propuesta por **Fred Glover (1986)**.  
- Extiende la búsqueda local usando **memoria adaptativa**.  
- Evita volver a explorar soluciones ya visitadas.

<!-- Imagen sugerida: diagrama de búsqueda con zonas prohibidas (tabú) -->

---

## ⚙️ Principio básico

1. Comenzar con una solución inicial.  
2. Explorar el vecindario.  
3. Seleccionar el **mejor movimiento no tabú**.  
4. Actualizar la **lista tabú**.  
5. Repetir hasta cumplir un criterio de parada.

---

## 🧩 Componentes clave

- **Lista Tabú:** movimientos prohibidos temporalmente.  
- **Criterio de aspiración:** permite ignorar tabú si la solución es excelente.  
- **Intensificación:** profundiza en buenas zonas.  
- **Diversificación:** explora nuevas áreas del espacio.

---

## 📈 Ventajas y desventajas

**Ventajas:**
- Evita ciclos.  
- Excelente exploración global.  
- Combinable con otras metaheurísticas.  

**Desventajas:**
- Sensible al tamaño de la lista tabú.  
- Costosa si el vecindario es muy grande.

---

# ⚖️ Comparación  
### Temple Simulado vs Búsqueda Tabú

| Característica | Temple Simulado | Búsqueda Tabú |
|-----------------|------------------|----------------|
| Inspiración | Física (recocido) | Memoria adaptativa |
| Aceptación | Probabilística | Determinística (según lista tabú) |
| Memoria | No usa | Usa memoria explícita |
| Control | Temperatura | Lista tabú + estrategias |
| Escapa de óptimos | Aleatoriamente | Evitando ciclos |

---

# 🧩 Aplicaciones comunes

- Programación de tareas y horarios.  
- Diseño de rutas de transporte.  
- Optimización de redes.  
- Diseño de circuitos.  
- Machine Learning (tuning de hiperparámetros).

<!-- Imagen sugerida: collage de aplicaciones de optimización -->

---

# 🧭 Conclusión

- Las metaheurísticas equilibran **exploración y explotación**.  
- Permiten resolver problemas complejos sin requerir métodos exactos.  
- **Temple Simulado** y **Búsqueda Tabú** son pilares fundamentales en optimización moderna.  

<!-- Imagen sugerida: mapa conceptual conectando metaheurísticas clásicas con modernas -->

---

## 📚 Referencias

- Glover, F. (1986). *Future paths for integer programming and links to artificial intelligence.*  
- Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). *Optimization by Simulated Annealing.*  
- Talbi, E.-G. (2009). *Metaheuristics: From Design to Implementation.* Wiley.