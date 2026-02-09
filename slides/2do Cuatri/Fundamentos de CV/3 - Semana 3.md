---
marp: true
paginate: true
theme: default
footer: "Fundamentos de Visión por Computadora | Unidad I – Semana 3"
---

# Fundamentos de Visión por Computadora  
## Semana 3  
### Transformaciones de Intensidad e Histogramas

---

## Objetivo de la sesión

Al finalizar esta semana, el estudiante será capaz de:

- Comprender las **transformaciones básicas de intensidad**
- Interpretar y analizar **histogramas de imágenes**
- Relacionar las transformaciones de intensidad con la mejora visual de imágenes
- Identificar aplicaciones prácticas del procesamiento por histograma

---

## Transformaciones de Intensidad

Las transformaciones de intensidad operan directamente sobre los valores de los píxeles.

- Se aplican en el **dominio espacial**
- Cada píxel se transforma de manera independiente
- Se expresan como funciones del tipo:

\[
s = T(r)
\]

Donde:
- \( r \) es la intensidad de entrada
- \( s \) es la intensidad de salida

---

## Transformación de Identidad

La transformación más simple:

- El valor del píxel **permanece igual**
- No altera la imagen
- Se utiliza como referencia o punto de comparación

\[
s = r
\]

**Aplicación principal:**  
Verificación de pipelines de procesamiento y análisis comparativo.

---

## Transformación Negativa

Invierte los niveles de intensidad:

- Los valores altos se vuelven bajos
- Los valores bajos se vuelven altos

\[
s = L - 1 - r
\]

Donde:
- \( L \) es el número de niveles de intensidad

**Uso común:**  
Resaltar detalles en imágenes con fondos claros u oscuros.

---

## ¿Qué es un histograma?

Un histograma representa:

- La **distribución de frecuencias** de los niveles de intensidad
- Cuántos píxeles existen para cada valor de intensidad

Ejes:
- Eje X: niveles de intensidad
- Eje Y: frecuencia (cantidad de píxeles)

---

## Interpretación del histograma

A partir del histograma podemos inferir:

- Brillo general de la imagen
- Contraste
- Presencia de saturación
- Distribución desigual de intensidades

El histograma **no depende de la posición espacial** de los píxeles.

---

## Procesamiento por histograma

El procesamiento por histograma permite:

- Analizar la calidad de una imagen
- Mejorar contraste
- Normalizar distribuciones de intensidad

Ejemplos comunes:
- Histogramas no normalizados
- Ecualización de histograma

---

## Aplicaciones prácticas

Las transformaciones de intensidad y el análisis de histogramas se usan en:

- Mejora de imágenes médicas
- Preprocesamiento para visión artificial
- Análisis de imágenes satelitales
- Sistemas de inspección visual

---

## Conclusiones

- Las transformaciones de intensidad son la base del procesamiento digital de imágenes
- El histograma es una herramienta clave para el análisis global de imágenes
- Estos conceptos son fundamentales para técnicas más avanzadas de visión por computadora

---

## Próxima semana

- Introducción al **filtrado espacial**
- Concepto de vecindad y máscaras
- Relación entre ruido y filtrado