---
marp: true
title: Programación para Inteligencia Artificial
subtitle: Semana 5 – Análisis y procesamiento de datos con histogramas
author: Ingeniería en Tecnologías de la Información
theme: default
paginate: true
---

# Análisis y procesamiento de datos con histogramas
## Semana 5  

---

## Contexto en el Aprendizaje Supervisado

- El rendimiento de un modelo depende directamente de la calidad de los datos
- Antes de entrenar un algoritmo es necesario:
  - Analizar la distribución de los datos
  - Identificar sesgos, escalas y valores atípicos
- El histograma es una herramienta clave en esta etapa

---

## ¿Qué es un histograma?

- Representación gráfica de la distribución de una variable numérica
- Agrupa los datos en intervalos llamados *bins*
- Permite observar:
  - Concentración de valores
  - Dispersión
  - Asimetría
  - Presencia de outliers

---

## Componentes de un histograma

- Eje X: intervalos de valores
- Eje Y: frecuencia de aparición
- Número de bins:
  - Influye directamente en la interpretación
  - Un mal ajuste puede ocultar o distorsionar información

---

## Importancia de los histogramas en IA

- Apoyan el análisis exploratorio de datos (EDA)
- Permiten detectar problemas antes del modelado
- Ayudan a decidir técnicas de preprocesamiento
- Reducen errores en etapas posteriores del desarrollo

---

## Histograma normalizado

- La frecuencia se expresa como proporción
- El área total del histograma es igual a 1
- Facilita:
  - Comparación entre datasets
  - Análisis estadístico
- Es común en flujos de trabajo de IA

---

## Histograma acumulado

- Muestra la suma progresiva de las frecuencias
- Permite responder preguntas como:
  - ¿Qué porcentaje de los datos es menor a cierto valor?
- Es útil para:
  - Análisis de percentiles
  - Definición de umbrales
  - Identificación de concentraciones extremas

---

## Histogramas y normalización de datos

- Histogramas permiten detectar diferencias de escala
- Datos mal escalados afectan algoritmos supervisados
- El análisis previo ayuda a decidir:
  - Normalización
  - Estandarización
  - Transformaciones adicionales

---

## Histogramas y detección de outliers

- Colas largas o barras aisladas indican valores atípicos
- Los outliers pueden:
  - Distorsionar métricas
  - Afectar el entrenamiento del modelo
- Su detección es clave antes del modelado

---

## Aplicaciones prácticas en IA

- Exploración de variables de entrada
- Comparación entre conjuntos de datos
- Validación visual del preprocesamiento
- Soporte para la toma de decisiones técnicas

---

## Cierre de la semana

- Los histogramas son una herramienta fundamental en IA
- Permiten comprender los datos antes del entrenamiento
- Un buen análisis mejora la calidad del modelo final
- El preprocesamiento inicia con la exploración visual