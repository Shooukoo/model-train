from ultralytics import YOLO
import os
import random

def probar_modelo():
    ruta_modelo = 'runs/classify/runs_zarzamora/modelo_v8/weights/best.pt'
    
    if not os.path.exists(ruta_modelo):
        print(f"No encuentro el modelo en: {ruta_modelo}")
        return

    model = YOLO(ruta_modelo)

    ruta_val = os.path.join('datasets', 'frutas', 'images', 'val')

    categorias = os.listdir(ruta_val)
    cat_random = random.choice(categorias)
    folder_random = os.path.join(ruta_val, cat_random)
    imgs = os.listdir(folder_random)
    
    if not imgs:
        print("Carpeta vacía.")
        return
        
    img_random = os.path.join(folder_random, random.choice(imgs))

    print(f"\n--- Probando con imagen real de: {cat_random} ---")
    print(f"Archivo: {img_random}")

    # Predecir
    results = model(img_random)

    # Mostrar resultados
    for r in results:
        top1_idx = r.probs.top1
        confianza = r.probs.top1conf.item()
        nombre_predicho = r.names[top1_idx]

        print(f"\nRESULTADO DE LA IA:")
        print(f"Predicción: {nombre_predicho.upper()}")
        print(f"Confianza:  {confianza * 100:.1f}%")
        
        if nombre_predicho == cat_random:
            print("✅ ¡ACERTÓ!")
        else:
            print(f"❌ FALLÓ (Era {cat_random})")

if __name__ == "__main__":
    probar_modelo()