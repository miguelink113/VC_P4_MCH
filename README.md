# Detección de Vehículos y Lectura de Matrículas con YOLO y OCR

Este proyecto realiza la **detección de vehículos y personas en vídeo** usando modelos YOLO y el reconocimiento de **matrículas** mediante OCR. Se implementan dos enfoques de OCR: **EasyOCR** y **SmolVLM**, permitiendo comparar velocidad y precisión.

---

## Requisitos

- Python >= 3.9
- CUDA compatible si se desea usar GPU
- Librerías principales:
  ```bash
  pip install torch torchvision torchaudio
  pip install ultralytics opencv-python pandas easyocr transformers pillow matplotlib
    ```

Modelos YOLO:

`yolo11n.pt` → Modelo preentrenado para detección de personas y vehículos.

`runs/detect/train/weights/best.pt` → Modelo entrenado para detección de matrículas.

dataset: https://drive.google.com/file/d/10VFeoGYP6D9QiDtg-PM9o0pBY9xw2sED/view

## Preparación

Comprobar GPU y CUDA:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

### Entrenamiento del modelo de matrículas (opcional si ya tienes best.pt):

```python
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.train(
    data='dataset/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0
)
```

## Uso
1. Detección con YOLO y EasyOCR

- Detecta **personas** y **vehículos** en el vídeo.
- Recorta la zona de la **matrícula** detectada.
- Aplica **OCR** con **EasyOCR** para leer el texto.

### Salida generada:
- **Vídeo anotado**: `output_tracking_easyocr.mp4`
- **CSV de detecciones**: `detecciones_tracking_easyocr.csv`
- **Ouput consola**: EASYOCR.txt

## Configuración de modelos y OCR
```python
# Configuración de los modelos YOLO
model_general = YOLO('yolo11n.pt')  # Detección general (personas, vehículos)
model_plates = YOLO(r'runs\detect\train\weights\best.pt')  # Detección de matrículas
reader = easyocr.Reader(['es'], gpu=True)
```
2. Detección con YOLO y SmolVLM


- Detecta **personas** y **vehículos** en el vídeo.
- Recorta la zona de la **matrícula** detectada.
- Aplica **OCR** con **EasyOCR** para leer el texto.

### Salida generada:
- **Vídeo anotado**: `output_tracking_smolvlm.mp4`
- **CSV de detecciones**: `detecciones_tracking_smolvlm.csv`
- **Ouput consola**: SMOLVLM.txt

## Configuración de modelos y SmolVLM
```python
from transformers import AutoProcessor, AutoModelForImageTextToText
# Configuración de los modelos YOLO
model_general = YOLO('yolo11n.pt')  # Detección general (personas, vehículos)
model_plates = YOLO(r'runs\detect\train\weights\best.pt')  # Detección de matrículas
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM-Instruct").to(Device)
```

## Funcionalidades

- Detección de objetos: personas, bicicletas, coches, motos, autobuses y camiones.

- OCR de matrículas: dos enfoques para comparar precisión y velocidad.

- CSV de resultados: información detallada por frame, incluyendo coordenadas de caja y texto detectado.

- Visualización de resultados: vídeos anotados con bounding boxes y texto de matrícula.

- Comparativa de rendimiento: genera gráficas de tiempos de inferencia por imagen y estadísticas (mínimo, promedio, máximo).

## Estructura de CSV
| frame | tipo_objeto | confianza | id_tracking | x1   | y1  | x2   | y2  | mx1    | my1  | mx2    | my2  | confianza_matricula | texto_matricula |
|-------|-------------|-----------|-------------|------|-----|------|-----|--------|------|--------|------|---------------------|-----------------|
| 1697  | car         | 0.9      | 88          | 1257 | 283 | 1516 | 508 | 1393.0 | 448.0| 1461.0 | 466.0| 0.84                | 3685KWM          |

## Comparativa de OCR

### Comparación estadística de tiempo por imagen(mínimo, promedio, máximo)

Se pueden analizar tiempos de inferencia y precisión entre EasyOCR y SmolVLM usando los logs `EASYOCR.txt` y `SMOLVLM.txt`

<img src="images/output.png" alt="Tiempos de inferencia por imagen" width="600"/>

<img src="images/output2.png" alt="Comparación tiempo EASYOCR y SMOLVLM" width="600"/>

- EasyOCR: promedio=0.77ms, min=0.40ms, max=16.50ms

- SmolVLM: promedio=0.66ms, min=0.40ms, max=22.40ms

### Comparación estadística en lectura de matrículas

<img src="images/output3.png" alt="Comparación lecturas EASYOCR y SMOLVLM" width="600"/>

- EasyOCR no consiguió leer correctamente ninguna matrícula completa, aunque logró algunas detecciones parciales.
- En cambio, SmolVLM obtuvo resultados notablemente mejores: logró lecturas parciales más precisas en muchas matrículas y, además, consiguió identificar varias de ellas de forma completa.

### Conclusión
A pesar de que ambos modelos tardan casi lo mismo en procesar cada imagen, SmolVLM logra resultados mucho mejores. Mientras EasyOCR solo reconoce partes de las matrículas, SmolVLM consigue leer varias por completo, manteniendo además una buena velocidad. SmolVLM demuestra ser la opción más eficaz.

## Autor

Proyecto desarrollado por Miguel Castellano Hernández

