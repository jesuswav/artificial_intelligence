---
marp: true
theme: default
paginate: true
class: lead
---

# 🧬 Algoritmos Genéticos  
### Inspirados en la evolución, aplicados a la optimización  
**Jesus Emmanuel Martínez García**

---

## 🎯 Objetivos de la presentación

- Comprender **qué es un algoritmo genético (AG)**.  
- Explicar sus **fundamentos biológicos y matemáticos**.  
- Conocer las **etapas del proceso evolutivo computacional**.  
- Analizar **casos de uso reales en la industria**.  
- Reflexionar sobre **ventajas, limitaciones y perspectivas futuras**.

---

## 🌱 ¿Qué es un Algoritmo Genético?

Un **algoritmo genético (AG)** es un método de **búsqueda y optimización inspirada en la evolución natural**, donde una población de posibles soluciones evoluciona hacia mejores soluciones con el tiempo.

> “Los AG imitan la selección natural, donde los individuos más aptos tienen más probabilidades de reproducirse y transmitir sus características.”

---

![bg height:100%](https://www.ighniz.com/wp-content/uploads/2017/06/Proceso.png)

---

## 🧠 Inspiración biológica

Los AG se basan en los principios de la **teoría de la evolución de Darwin**:

- **Selección natural** 🧬  
  Los individuos con mejor desempeño tienen más posibilidades de reproducirse.
- **Cruzamiento (Crossover)** 🔄  
  Combina características de dos “padres”.
- **Mutación** ⚡  
  Introduce variaciones aleatorias para mantener diversidad.

---

## ⚙️ Estructura general de un Algoritmo Genético

1. **Inicialización**: generar población inicial aleatoria.  
2. **Evaluación**: medir desempeño con una *función de aptitud*.  
3. **Selección**: elegir los mejores individuos.  
4. **Cruzamiento**: combinar características de los seleccionados.  
5. **Mutación**: aplicar pequeños cambios aleatorios.  
6. **Reemplazo**: formar nueva población.  
7. **Criterio de parada**: alcanzar solución óptima o límite de generaciones.

---
## Diagrama de flujo: Algoritmo Genético

![height:70%](https://blogs.imf-formacion.com/blog/tecnologia/wp-content/uploads/2020/10/alg_geneticos_horizontal-1024x477.png)

---

## 🔍 Representación de soluciones

Cada individuo (solución) se representa como un **cromosoma**, formado por **genes** que pueden ser:

- Binarios (0, 1)
- Reales (valores continuos)
- Enteros (índices o posiciones)
- Estructuras personalizadas

Ejemplo (binario):  
`Cromosoma = [1, 0, 1, 1, 0, 0, 1]`

---

## 🧩 Ejemplo académico clásico

### Problema del Viajante (TSP)
Encontrar la ruta más corta que visita todas las ciudades una vez.

🔹 Representación: cada cromosoma es un orden de ciudades.  
🔹 Función de aptitud: distancia total del recorrido.  
🔹 Operadores genéticos:  
- Cruzamiento → mezcla de rutas.  
- Mutación → intercambio de dos ciudades.

![bg right:40% height:90%](https://blogs.sas.com/content/sasla/files/2019/09/modelo1.jpg)

---

## 🏭 Aplicaciones industriales reales

### 1. Manufactura y Producción  
- **Optimización de rutas de robots industriales.**  
- **Planificación de producción y distribución.**  
- **Diseño de líneas de montaje eficientes.**

### 2. Energía y Recursos  
- **Optimización del despacho eléctrico.**  
- **Configuración óptima de turbinas eólicas o paneles solares.**

---

## 🏭 Aplicaciones industriales reales

### 3. Transporte y Logística  
- **Ruteo de vehículos (VRP).**  
- **Gestión de flotas o drones autónomos.**

### 4. Ingeniería y Diseño  
- **Diseño estructural asistido por IA.**  
- **Parámetros óptimos en simulaciones CAD.**

---

## 💡 Ejemplo aplicado a la industria logística

**Problema:** Optimizar rutas de entrega para minimizar distancia y tiempo.

**Solución con AG:**
1. Cada individuo representa una posible ruta.  
2. Función de aptitud: tiempo total de entrega.  
3. Cruce y mutación → generan nuevas combinaciones de rutas.  
4. Selección → mantiene las mejores soluciones.

Resultado:  
🚚 Reducción del 15–25% en costos logísticos.  
📉 Menor consumo de combustible y emisiones.

---

## 📊 Ventajas de los Algoritmos Genéticos

✅ Capaces de manejar problemas **no lineales** y **no diferenciables**.  
✅ No necesitan gradientes ni modelos matemáticos exactos.  
✅ Flexibles ante restricciones y múltiples objetivos.  
✅ Excelentes para **espacios de búsqueda grandes**.

---

## ⚠️ Limitaciones

⚙️ **Alto costo computacional** (muchas generaciones).  
🎯 **No garantizan el óptimo global**, solo soluciones cercanas.  
🔍 **Requieren ajuste fino de parámetros** (tasa de mutación, tamaño de población).  
📉 **Pueden converger prematuramente** si la diversidad se pierde.

---

## 🔬 Optimización híbrida

En la práctica industrial, los AG suelen **combinarse con otros métodos**:

- AG + **Redes Neuronales** → ajuste de hiperparámetros.  
- AG + **Sistemas Difusos** → optimización de reglas fuzzy.  
- AG + **Simulated Annealing** → refinamiento de soluciones.


---

## 🧮 Ejemplo en optimización de parámetros

Un algoritmo genético puede ajustar los parámetros de una red neuronal:
- Tasa de aprendizaje  
- Número de neuronas por capa  
- Función de activación  

🎯 Objetivo: minimizar el error de validación.  
Resultado: modelos **más precisos y estables**.

---

## 🚀 Futuro de los Algoritmos Genéticos

- Integración con **aprendizaje profundo (Neuroevolution)**.  
- Aplicaciones en **sistemas autónomos inteligentes**.  
- Uso en **optimización cuántica y bioinformática**.  
- **Automatización de diseño industrial** mediante evolución dirigida.

---

## 🧩 Conclusiones

- Los AG son una **herramienta poderosa de optimización inspirada en la biología**.  
- Tienen aplicaciones **multidisciplinarias** en ingeniería, energía, transporte y manufactura.  
- Su eficacia depende del **diseño de la función de aptitud** y **operadores genéticos**.  
- Son una base importante para la **inteligencia artificial evolutiva moderna**.

---

## 📚 Referencias

1. Goldberg, D. E. *Genetic Algorithms in Search, Optimization, and Machine Learning.* Addison-Wesley, 1989.  
2. Holland, J. H. *Adaptation in Natural and Artificial Systems.* University of Michigan Press, 1975.  
3. Mitchell, M. *An Introduction to Genetic Algorithms.* MIT Press, 1998.  
4. IEEE Transactions on Evolutionary Computation (revista).  
5. Casos industriales en Siemens, Toyota, y General Electric.