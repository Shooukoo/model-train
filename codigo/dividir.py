import os
import shutil
import random

def paso_3_dividir():
    # Origen: las fotos ya procesadas
    source_dir = os.path.join('datasets', 'frutas', 'temp_procesadas')
    
    # Destinos finales según tu estructura
    base_dir = os.path.join('datasets', 'frutas', 'images')
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')

    # Crear directorios finales
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Listar y mezclar
    images = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
    random.shuffle(images) # ¡Importante mezclar para que sea aleatorio!

    # Calcular corte 80/20
    split_idx = int(len(images) * 0.8)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    print(f"--- PASO 3: Dividiendo en '{base_dir}' ---")
    print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)}")

    # Mover archivos
    for img in train_imgs:
        shutil.move(os.path.join(source_dir, img), os.path.join(train_dir, img))
        
    for img in val_imgs:
        shutil.move(os.path.join(source_dir, img), os.path.join(val_dir, img))

    # Limpieza opcional: borrar las carpetas temporales
    shutil.rmtree(os.path.join('datasets', 'frutas', 'temp_raw'), ignore_errors=True)
    shutil.rmtree(os.path.join('datasets', 'frutas', 'temp_procesadas'), ignore_errors=True)

    print("\n[OK] ¡Dataset listo! Revisa la carpeta: datasets/frutas/images")

if __name__ == "__main__":
    paso_3_dividir()