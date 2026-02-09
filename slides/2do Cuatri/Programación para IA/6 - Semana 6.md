---
marp: true
title: Programación para Inteligencia Artificial
subtitle: Semana 6 – Aprendizaje No Supervisado
author: Lic. en Ingeniería en Tecnologías de la Información
theme: default
paginate: true
---

# Semana 6  
## Aprendizaje No Supervisado
### Funciones básicas en imágenes digitales

---

## Objetivo de la semana

Comprender los fundamentos del aprendizaje no supervisado y su aplicación en el análisis de datos sin etiquetas, mediante la implementación y evaluación de algoritmos de agrupamiento.

---

## ¿Qué es el aprendizaje no supervisado?

- Tipo de aprendizaje automático sin datos etiquetados  
- Identifica patrones ocultos en los datos  
- Se basa en similitud, distancia y densidad  
- Uso común en exploración y análisis de datos

---

## Casos de uso comunes

- Segmentación de clientes  
- Análisis exploratorio de datos  
- Detección de patrones y grupos  
- Compresión y reducción de información  

---

## Algoritmos de aprendizaje no supervisado

- Algoritmos de agrupamiento (clustering)  
- Algoritmos basados en centroides  
- Algoritmos basados en mapas topológicos  

---

## K-Means (K-medias)

- Algoritmo de agrupamiento basado en centroides  
- Requiere definir el número de clusters (k)  
- Minimiza la distancia intra-cluster  
- Sensible a la inicialización y a outliers  

---

## Funcionamiento general de K-Means

1. Inicialización de centroides  
2. Asignación de puntos al centroide más cercano  
3. Recalculo de centroides  
4. Iteración hasta convergencia  

---

## Self-Organizing Maps (SOM)

- Red neuronal no supervisada  
- Proyecta datos de alta dimensión a un mapa 2D  
- Preserva relaciones topológicas  
- Útil para visualización y exploración de datos  

---

## Comparación: K-Means vs SOM

- K-Means: simple, rápido, basado en distancia  
- SOM: más complejo, basado en redes neuronales  
- Ambos permiten descubrir estructura en los datos  

---

## Evaluación de modelos no supervisados

- No existe una “etiqueta real”  
- Se utilizan métricas internas  
- Se evalúa la calidad del agrupamiento  

---

## Métricas de desempeño

- **Silhouette Score**
  - Mide cohesión y separación
- **Davies-Bouldin Index**
  - Mide similitud entre clusters
  - Menor valor indica mejor agrupamiento  

---

## Importancia de las métricas

- Comparar diferentes configuraciones  
- Seleccionar el mejor modelo  
- Justificar decisiones técnicas  

---

## Conclusiones

- El aprendizaje no supervisado permite descubrir patrones ocultos  
- K-Means y SOM son técnicas fundamentales  
- Las métricas son clave para validar resultados  

---

## Próximos pasos

- Implementación práctica de algoritmos  
- Análisis de resultados  
- Interpretación y comunicación de hallazgos  