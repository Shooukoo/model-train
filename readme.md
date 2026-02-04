# Guía de Desarrollo: Worker de IA (YOLO + Python)

Este documento resume los comandos esenciales para el flujo de trabajo diario y explica los resultados de la primera prueba de concepto (PoC) realizada para el sistema de detección de frutas.

## 1. Comandos para usar

### Gestión del Entorno

**Revisar** Siempre verifica que aparezca `(venv)` al inicio de tu terminal antes de trabajar.

* **Activar el entorno:**

```bash
source venv/Scripts/activate
```

* **Desactivar (al terminar):**

```bash
deactivate
```

### Gestión de Dependencias

Mantenimiento de librerías para replicar el entorno en otros servidores o contenedores.

* **Instalar nueva librería:**

```bash
pip install nombre_paquete
```

* **Congelar dependencias (Guardar cambios):**

*Ejecutar siempre antes de un commit si instalaste algo nuevo.*

```bash
pip freeze > requirements.txt
```

* **Instalar todo (Setup inicial):**

```bash
pip install -r requirements.txt
```

### Comandos YOLO (Ultralytics)

Operaciones principales para el "Worker de IA".

* **Entrenar Modelo (Training):**
  * `epochs`: Aumentar a 50-100 para producción.
  * `imgsz`: 640 es el estándar balanceado velocidad/precisión.

```bash
yolo task=detect mode=train model=yolo11n.pt data=frutas_config.yaml epochs=50 imgsz=640
```

* **Probar/Inferir (Prediction):**
  * Detecta objetos en nuevas imágenes usando tu modelo entrenado (`best.pt`).
  * *Nota:* Verifica la ruta `runs/detect/trainX` (el número incrementa con cada entrenamiento).

```bash
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=datasets/frutas/images/val/ save=True
```

* **Reanudar Entrenamiento:**
  * Útil si se canceló el proceso o hubo un error.

```bash
yolo task=detect mode=train resume=True model=runs/detect/train/weights/last.pt
```

## 2. Análisis del Ejercicio: "Hello World" de Detección

### Contexto del Proyecto

[cite_start]Según la arquitectura de investigación, este componente corresponde al **"Worker de IA en Python"**[cite: 188]. [cite_start]Su función es operar de forma asíncrona: descargar imágenes crudas (RAW) desde S3[cite: 192], procesarlas y generar el conteo de frutas.

### Resumen de la Prueba

Se realizó un entrenamiento supervisado ("Fine-tuning") utilizando la arquitectura **YOLOv11 Nano**.

* **Dataset:** Muestra pequeña de frutas (imágenes propias).
* **Configuración:** 10 épocas, imagen de 640px.

### Interpretación de Resultados (Métricas Clave)

Los resultados obtenidos en la época 10 mostraron el siguiente comportamiento:

| Métrica | Valor Obtenido | Interpretación Técnica |
| :--- | :--- | :--- |
| **Recall (Recuperación)** | `1.0` (100%) | **Excelente.** El modelo no ignoró ninguna fruta. De todas las frutas presentes en las fotos de validación, las encontró absolutamente todas. Esto es crítico para el conteo preciso. |
| **Precision** | `~0.028` (2.8%) | **Bajo (Esperado).** El modelo tiene "falsos positivos". Al tener pocos ejemplos, está dibujando recuadros en exceso o confundiendo el fondo con frutas para asegurarse de no fallar el Recall. |
| **Box Loss** | `1.72` 📉 `1.50` | **Aprendizaje Exitoso.** La "pérdida de caja" disminuyó constantemente. Esto indica que la red neuronal está entendiendo matemáticamente cómo ajustar los bordes (bounding boxes) alrededor de los objetos. |

### Conclusión

La prueba fue un **éxito técnico**. Se validó que:

1. El entorno virtual y las librerías (`ultralytics`, `torch`) están correctamente configurados.
2. La estructura de carpetas `datasets/frutas` es legible por el motor de IA.
3. El hardware es capaz de completar ciclos de entrenamiento.

**Siguientes pasos:** Para subir la precisión, se requiere aumentar el tamaño del dataset (más fotos variadas) y el número de épocas de entrenamiento (mínimo 50).
