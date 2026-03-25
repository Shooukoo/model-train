import cv2
import os
from pathlib import Path

def resize_images_in_folder(folder_path, output_folder=None, width=5120, height=5120):
    """
    Redimensiona todas las imágenes en una carpeta.
    
    Args:
        folder_path: Ruta de la carpeta con imágenes
        output_folder: Carpeta destino (opcional)
        width: Ancho deseado
        height: Alto deseado
    """
    folder = Path(folder_path)
    
    if output_folder:
        Path(output_folder).mkdir(exist_ok=True)
    
    for image_file in folder.glob('*'):
        if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            img = cv2.imread(str(image_file))
            if img is not None:
                resized = cv2.resize(img, (width, height))
                output_path = Path(output_folder) / image_file.name if output_folder else image_file
                cv2.imwrite(str(output_path), resized)
                print(f"✓ {image_file.name}")

# Uso:
resize_images_in_folder('D:/magenes proyecto ia', output_folder='D:/imagenes proyecto ia', width=5170, height=5181)