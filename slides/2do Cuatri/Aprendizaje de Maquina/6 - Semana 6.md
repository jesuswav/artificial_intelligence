---
marp: true
title: Aprendizaje Máquina – Semana 6
theme: default
paginate: true
---

# Aprendizaje Máquina  
## Semana 6  
### Aprendizaje Supervisado: Clasificación

---

## Contexto de la Unidad

En esta semana se estudian técnicas de **aprendizaje supervisado** orientadas a la **clasificación**, como parte del análisis de datos para la toma de decisiones.

Se enfatiza:
- El uso correcto de los datos
- La selección del modelo adecuado
- La evaluación del desempeño del clasificador

---

## ¿Qué es la Clasificación?

La **clasificación** es una técnica de aprendizaje supervisado que asigna una observación a una **categoría o clase discreta**, con base en ejemplos previamente etiquetados.

Ejemplos:
- Aprobar / No aprobar
- Spam / No spam
- Riesgo alto / Riesgo bajo

---

## Clasificación vs Regresión

| Característica | Clasificación | Regresión |
|---------------|---------------|-----------|
| Tipo de salida | Discreta | Continua |
| Objetivo | Asignar clases | Estimar valores |
| Ejemplo | Diagnóstico médico | Predicción de precios |

---

## Flujo General de un Modelo de Clasificación

1. Recolección de datos
2. Análisis exploratorio
3. Preprocesamiento
4. Selección del algoritmo
5. Entrenamiento
6. Evaluación del modelo
7. Interpretación de resultados

---

## Importancia del Preprocesamiento

Antes de entrenar un clasificador es necesario:
- Identificar valores nulos
- Detectar valores atípicos
- Normalizar o escalar variables
- Seleccionar atributos relevantes

Un mal preprocesamiento impacta directamente en el desempeño del modelo.

---

## Algoritmos Comunes de Clasificación

Algunos algoritmos ampliamente utilizados son:
- Regresión logística
- K-Nearest Neighbors (KNN)
- Árboles de decisión
- Máquinas de soporte vectorial (SVM)

Cada uno tiene ventajas y limitaciones según el tipo de datos.

---

## Métricas de Evaluación en Clasificación

Para evaluar un clasificador se utilizan métricas como:
- Exactitud (Accuracy)
- Precisión (Precision)
- Recuperación (Recall)
- F1-Score

Estas métricas permiten analizar distintos aspectos del desempeño del modelo.

---

## Importancia de las Métricas

No siempre la exactitud es suficiente.

Por ejemplo:
- En detección de fraude
- En diagnóstico médico
- En sistemas de alerta

Es necesario evaluar **errores y aciertos por clase**.

---

## Aplicaciones Prácticas

La clasificación se utiliza en:
- Sistemas de recomendación
- Diagnóstico automatizado
- Detección de fraudes
- Análisis de riesgo
- Clasificación de clientes

---

## Cierre de la Semana

Al finalizar esta semana, el estudiante será capaz de:
- Identificar problemas de clasificación
- Seleccionar un algoritmo adecuado
- Evaluar un modelo usando métricas apropiadas
- Interpretar resultados para apoyar la toma de decisiones