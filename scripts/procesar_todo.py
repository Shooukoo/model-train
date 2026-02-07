import os
import random
import shutil
from PIL import Image, UnidentifiedImageError
from collections import defaultdict

def organizar_para_clasificacion():
    # --- RUTAS ---
    # 1. Donde están las descargas sucias (del script anterior)
    # Nota: El script de descarga usaba 'dataset' (singular), tu foto 'datasets' (plural).
    # Buscamos en 'dataset' primero por si acaso.
    input_dir = os.path.join('datasets', 'frutas', 'temp_raw')
    
    # 2. Ruta FINAL según tu captura de pantalla
    base_output = os.path.join('datasets', 'frutas', 'images')
    
    # Configuración
    img_size = (224, 224)
    train_ratio = 0.8

    # Verificar que existen fotos para procesar
    if not os.path.exists(input_dir):
        print(f"[ERROR] No encuentro la carpeta de descargas: {input_dir}")
        print("Asegúrate de haber ejecutado el paso 1 (descargar) o corrige el nombre de la carpeta.")
        return

    print(f"--- PROCESANDO DESDE: {input_dir} ---")
    print(f"--- HACIA: {base_output} ---\n")

    # 1. Agrupar archivos por categoría según su nombre
    files_by_category = defaultdict(list)
    
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    
    for f in files:
        # Detectar categoría: "roya_1.jpg" -> "roya"
        if '_' in f:
            category = f.split('_')[0]
            files_by_category[category].append(f)
        else:
            files_by_category['otros'].append(f)

    total_processed = 0

    # 2. Procesar cada categoría
    for category, file_list in files_by_category.items():
        # Mezclar para que el split sea aleatorio
        random.shuffle(file_list)
        
        # Calcular división
        split_idx = int(len(file_list) * train_ratio)
        train_files = file_list[:split_idx]
        val_files = file_list[split_idx:]
        
        # Función interna para procesar y guardar
        def procesar_lista(lista_archivos, subfolder):
            count = 0
            # Ruta destino: datasets/frutas/images/train/roya
            dest_dir = os.path.join(base_output, subfolder, category)
            os.makedirs(dest_dir, exist_ok=True) # Crea la carpeta si no existe

            for filename in lista_archivos:
                src = os.path.join(input_dir, filename)
                dst = os.path.join(dest_dir, filename)
                
                try:
                    with Image.open(src) as img:
                        # Convertir y redimensionar
                        img = img.convert('RGB').resize(img_size, Image.Resampling.LANCZOS)
                        img.save(dst, 'JPEG', quality=90)
                        count += 1
                except:
                    print(f"  [x] Error con: {filename}")
            return count

        print(f"Procesando '{category}'...")
        n_train = procesar_lista(train_files, 'train')
        n_val = procesar_lista(val_files, 'val')
        
        print(f"  -> Guardados en Train: {n_train} | Val: {n_val}")
        total_processed += (n_train + n_val)

    print("\n" + "="*40)
    print(f"¡LISTO! Total imágenes procesadas: {total_processed}")
    print(f"Estructura creada en: {base_output}")
    print("  /train/roya/...")
    print("  /val/roya/...")
    print("="*40)

if __name__ == "__main__":
    organizar_para_clasificacion()