---
marp: true
title: Fundamentos de Visión por Computadora
subtitle: Semana 6 – Transformaciones de Intensidad (Funciones básicas)
author: Licenciatura en Ingeniería en Tecnologías de la Información
theme: default
class: lead
paginate: true
---

# Semana 6  
## Transformaciones de Intensidad  
### Funciones básicas en imágenes digitales

---

## Objetivo de la semana

Comprender y aplicar funciones básicas de transformación de intensidad
en imágenes digitales para mejorar, analizar o modificar su información visual.

---

## Transformaciones de Intensidad

Las transformaciones de intensidad modifican los valores de los píxeles
sin alterar su posición espacial.

Se aplican directamente sobre:
- Imágenes en escala de grises
- Canales individuales de imágenes a color

---

## Dominio espacial

- Las transformaciones se realizan **píxel a píxel**
- Cada valor de entrada produce un valor de salida
- No depende de los píxeles vecinos

Relación general:

$
s = T(r)
$

Donde:
- \( r \): intensidad original
- \( s \): intensidad transformada

---

## Funciones lineales

Las funciones lineales permiten:
- Ajustar brillo
- Ajustar contraste
- Invertir intensidades

Son simples, eficientes y ampliamente utilizadas.

---

## Transformación identidad

La imagen no sufre cambios:

$
s = r
$

Uso principal:
- Verificación de sistemas
- Comparación con otras transformaciones
- Punto de referencia

---

## Transformación negativa

Invierte los niveles de intensidad:

$
s = L - 1 - r
$

Donde:
- \( L \) es el número de niveles de gris

Aplicaciones:
- Realce de detalles
- Análisis médico
- Imágenes con fondos claros

---

## Ajuste de brillo

Se realiza sumando o restando un valor constante:

$
s = r + b
$

- \( b > 0 \): imagen más clara
- \( b < 0 \): imagen más oscura

Debe controlarse el rango válido de intensidades.

---

## Ajuste de contraste

Escala los valores de intensidad:

$
s = a \cdot r
$

- \( a > 1 \): mayor contraste
- \( 0 < a < 1 \): menor contraste

Puede combinarse con ajuste de brillo.

---

## Importancia práctica

Las transformaciones de intensidad son la base para:
- Preprocesamiento de imágenes
- Mejora visual
- Segmentación
- Análisis automático

Son esenciales antes de aplicar algoritmos más complejos.

---

## Cierre de la semana

- Las transformaciones de intensidad modifican valores, no posiciones
- Las funciones lineales son simples pero poderosas
- Constituyen la base del procesamiento digital de imágenes