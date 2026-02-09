---
marp: true
theme: default
class: lead
paginate: true
---

# Fundamentos de Visión por Computadora  
## Semana 4: Transformaciones de Intensidad y Espaciales

---

## Objetivo de la semana

Al finalizar esta semana, el estudiante será capaz de procesar imágenes digitales mediante **transformaciones de intensidad** y **operaciones espaciales**, comprendiendo su fundamento matemático y su impacto en la calidad y análisis de la imagen.

---

## ¿Qué son las transformaciones de intensidad?

- Operaciones aplicadas directamente a los valores de los píxeles
- Modifican brillo y contraste de la imagen
- Se aplican píxel a píxel (dominio puntual)

---

## Funciones básicas de transformación

- Transformación identidad  
- Transformación negativa  
- Transformaciones lineales  
- Ajustes de contraste  

---

## Transformación identidad

- No altera la imagen
- Se utiliza como referencia
- Permite validar implementaciones

$
s = r
$

---

## Transformación negativa

- Invierte los niveles de intensidad
- Resalta detalles claros en fondos oscuros

$
s = L - 1 - r
$

---

## Transformaciones lineales

- Ajuste de brillo
- Ajuste de contraste
- Escalamiento de intensidades

$
s = a \cdot r + b
$

---

## Procesamiento en el dominio espacial

- Operaciones considerando vecindarios
- Uso de máscaras o kernels
- Base del filtrado espacial

---

## Filtrado espacial

- Suavizado (reducción de ruido)
- Realce de bordes
- Detección de detalles

---

## Máscaras y convolución

- Matriz pequeña aplicada sobre la imagen
- Operación de convolución
- Influencia directa en cada píxel

---

## Ejemplos de filtros espaciales

- Filtro promedio
- Filtro gaussiano
- Filtros pasa-altas

---

## Procesamiento de histogramas

- Representación estadística de intensidades
- Base para mejora de contraste
- Análisis de calidad de imagen

---

## Histograma de una imagen

- Eje X: niveles de intensidad
- Eje Y: frecuencia de píxeles
- Describe la distribución tonal

---

## Aplicaciones prácticas

- Mejora de imágenes médicas
- Preprocesamiento para visión artificial
- Sistemas de inspección visual

---

## Conclusiones

- Las transformaciones de intensidad son fundamentales
- El filtrado espacial permite extraer información relevante
- Son la base de procesos más avanzados en visión por computadora