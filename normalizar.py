import os
from PIL import Image, UnidentifiedImageError

def paso_2_normalizar():
    input_folder = os.path.join('datasets', 'frutas', 'temp_raw')
    output_folder = os.path.join('datasets', 'frutas', 'temp_procesadas')
    size = (224, 224) # Tamaño cuadrado estándar
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"--- PASO 2: Procesando imágenes hacia '{output_folder}' ---")
    
    count = 0
    # Recorremos todas las subcarpetas de descarga
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            try:
                img_path = os.path.join(root, file)
                with Image.open(img_path) as img:
                    # Convertir a RGB y Redimensionar
                    img = img.convert('RGB').resize(size, Image.Resampling.LANCZOS)
                    
                    # Guardar con nombre limpio secuencial
                    new_name = f"zarzamora_dañada_{count}.jpg"
                    img.save(os.path.join(output_folder, new_name), 'JPEG', quality=90)
                    count += 1
            except:
                continue # Ignorar archivos corruptos silenciosamente

    print(f"[OK] Se han procesado y limpiado {count} imágenes.")

if __name__ == "__main__":
    paso_2_normalizar()