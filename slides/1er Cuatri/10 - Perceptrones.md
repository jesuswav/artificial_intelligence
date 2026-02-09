---
marp: true
theme: default
paginate: true
class: lead
backgroundColor: #f5f5f5
math: mathjax
---

# 🧠 El Perceptrón y el Perceptrón Multicapa  
**Fundamentos Teóricos y Funcionamiento Interno**
**Autor:** Jesús Emmanuel Martínez García  

---

## 🏁 Introducción

- El perceptrón es la **unidad básica de una red neuronal artificial**.  
- Fue propuesto por **Frank Rosenblatt (1958)**.  
- Su objetivo: **imitar el comportamiento de una neurona biológica** para realizar tareas de clasificación.  

📊 Aplicaciones: reconocimiento de patrones, clasificación lineal, procesamiento de señales, etc.

---

## ⚙️ Estructura de una Neurona Artificial

- Entrada: vector de características \( x = [x_1, x_2, ..., x_n] \)  
- Pesos: \( w = [w_1, w_2, ..., w_n] \)  
- Umbral o sesgo: \( b \)  
- Salida: \( y = f(z) \), donde \( z = w \cdot x + b \)

![height:300](https://upload.wikimedia.org/wikipedia/commons/4/4f/Perceptron_example.svg)

---

## 🧮 Funcionamiento Matemático

1. **Suma ponderada**:  
   $$
   z = \sum_{i=1}^{n} w_i x_i + b
   $$

2. **Función de activación**:  
   Decide si la neurona se activa o no.  
   Ejemplo (función escalón):  
   $$
   f(z) = 
   \begin{cases}
   1 & \text{si } z > 0 \\
   0 & \text{si } z \leq 0
   \end{cases}
   $$

3. **Salida final**:  
   $$
   y = f(z)
   $$

---

## 🧩 Ejemplo Visual del Perceptrón Simple

![height:400](https://94fa3c88.delivery.rocketcdn.me/es/files/2021/04/illu_perceptron_blog-138.png)

- Clasifica puntos en dos clases (líneas separables linealmente).  
- Si los datos **no son linealmente separables**, el perceptrón **falla**.

---

## 🧭 Limitaciones del Perceptrón Simple

- Solo puede resolver **problemas linealmente separables**.  
- No puede aprender funciones como **XOR**.  
- No maneja relaciones complejas entre variables.

👉 Para solucionar esto surge el **Perceptrón Multicapa (MLP)**.

---

## 🏗️ Perceptrón Multicapa (MLP)

- Extiende la idea del perceptrón simple añadiendo **capas ocultas**.
- Cada capa aprende **representaciones intermedias**.
- Se compone de:
  - Capa de entrada
  - Una o más capas ocultas
  - Capa de salida

![bg right:40% height:80%](https://upload.wikimedia.org/wikipedia/commons/4/46/Colored_neural_network.svg)

---

## 🧮 Propagación hacia Adelante (Forward Propagation)

1. Cada neurona calcula su salida:  
   $$
   a_j^{(l)} = f\left(\sum_i w_{ji}^{(l)} a_i^{(l-1)} + b_j^{(l)}\right)
   $$
2. Las salidas se propagan capa por capa hasta la salida final.

💡 Funciones de activación comunes:
- Sigmoide: \( f(x) = \frac{1}{1 + e^{-x}} \)
- ReLU: \( f(x) = \max(0, x) \)
- Tanh: \( f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \)

---

## 🔁 Retropropagación del Error (Backpropagation)

- Mecanismo de **aprendizaje supervisado**.  
- Calcula el error entre la salida predicha y la deseada.  
- Propaga ese error hacia atrás para ajustar los pesos:

$$
w_{ij}^{(l)} \leftarrow w_{ij}^{(l)} - \eta \frac{\partial E}{\partial w_{ij}^{(l)}}
$$

Donde:
- \( E \): error global  
- \( \eta \): tasa de aprendizaje  

---

## 🧠 Aprendizaje del MLP

1. Inicializa pesos aleatoriamente  
2. Calcula salida (forward)  
3. Mide el error (función de pérdida)  
4. Propaga el error (backpropagation)  
5. Actualiza pesos  
6. Repite hasta convergencia

---

## 📊 Ejemplo de Clasificación con MLP

![height:400](https://antonio-richaud.com/blog/imagenes/archivo/41-redes-perceptron-multicapa/diagrama.png)

- El MLP puede **aprender fronteras no lineales**.  
- Permite resolver problemas como **XOR** o reconocimiento de imágenes simples.

---

## 📚 Comparación: Perceptrón Simple vs. Multicapa

| Característica | Perceptrón Simple | MLP |
|----------------|------------------|-----|
| Nº de capas ocultas | 0 | ≥ 1 |
| Función de activación | Escalón | No lineal (ReLU, Sigmoide, etc.) |
| Tipo de problemas | Lineales | No lineales |
| Algoritmo de entrenamiento | Regla del perceptrón | Backpropagation |
| Capacidad de aprendizaje | Limitada | Alta |

---

## 🧩 Ejemplo: Solución del Problema XOR

- Perceptrón simple: ❌ no puede separar clases  
- MLP con una capa oculta: ✅ logra separación no lineal  

![bg right:50% height:65%](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiXRiYcUybQ5_g5UuOyB0KLx7Y-UZYr-KCxQBSDZUiVEMGoQOo49souE8HFH-BVpTpInJFXelkl9Hi4pRcF5gRXAUl8mnimGmRAVJuR2qz5T_k5S_ysOU1iEQj3NMZN_1ilXNmYobIz4VeR/s1600/PerceptronMultiCapaXOR.png)

---

## 🧠 Interpretación Conceptual

- Cada capa aprende una **representación más abstracta** de los datos.  
- En capas profundas, las neuronas aprenden **características complejas**.  
- Este principio es la base del **Deep Learning**.

---

## 🧾 Conclusiones

✅ El perceptrón fue el primer modelo de red neuronal.  
✅ El perceptrón multicapa superó sus limitaciones mediante el uso de capas ocultas.  
✅ El entrenamiento mediante **backpropagation** es la base del aprendizaje profundo.  
✅ Comprender su funcionamiento es esencial para entender las **redes neuronales modernas**.

---

## 🔗 Referencias

- Rosenblatt, F. (1958). *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain.*  
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press.  
- Nielsen, M. (2015). *Neural Networks and Deep Learning.*