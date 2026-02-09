---
marp: true
title: Fundamentos de Visión por Computadora
theme: default
paginate: true
---

# Fundamentos de Visión por Computadora  
## Semana 5  
### Transformaciones de Intensidad y Espaciales

---

## Objetivo de la Semana

Aplicar transformaciones de intensidad sobre imágenes digitales mediante:
- Funciones lineales
- Procesamiento por histogramas

Con el fin de mejorar la información visual contenida en una imagen.

---

## Transformaciones de Intensidad

Las transformaciones de intensidad:
- Operan directamente sobre los valores de los píxeles
- No modifican la geometría de la imagen
- Permiten resaltar detalles, contrastes o regiones específicas

---

## Funciones Lineales de Intensidad

Una función lineal se define como:

$
s = a \cdot r + b
$

Donde:
- \( r \): intensidad original
- \( s \): intensidad transformada
- \( a \): ganancia
- \( b \): desplazamiento

---

## Transformación de Identidad

- Mantiene la imagen sin cambios
- Sirve como referencia para otras transformaciones

$
s = r
$

Aplicación común:
- Validación de pipelines de procesamiento

---

## Transformación Negativa

Invierte los niveles de intensidad:

$
s = L - 1 - r
$

Usos:
- Realce de detalles en imágenes oscuras
- Análisis médico e industrial

---

## Introducción al Histograma

El histograma representa:
- La distribución de intensidades de una imagen
- La frecuencia de aparición de cada nivel de gris

Permite:
- Analizar contraste
- Detectar saturaciones
- Tomar decisiones de mejora visual

---

## Procesamiento por Histogramas

El procesamiento por histogramas permite:
- Ajustar el contraste
- Redistribuir los niveles de gris
- Preparar imágenes para segmentación

Ejemplo:
- Histogramas no normalizados
- Histogramas acumulativos

---

## Aplicaciones Prácticas

- Preprocesamiento para visión artificial
- Mejora visual en imágenes médicas
- Inspección industrial
- Sistemas de reconocimiento de patrones

---

## Conclusiones

- Las transformaciones de intensidad son la base del procesamiento digital
- Los histogramas permiten analizar y mejorar imágenes
- Estas técnicas son fundamentales para etapas posteriores como segmentación

---

## Fin de la Semana 5