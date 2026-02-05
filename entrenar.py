from ultralytics import YOLO

def entrenar_profesional():
    # Usamos el modelo nano (n) para velocidad, o small (s) para más inteligencia
    model = YOLO('yolov8n-cls.pt') 

    print("--- INICIANDO ENTRENAMIENTO CON AUMENTO DE DATOS ---")
    print("El sistema generará variaciones de tus fotos automáticamente.")

    results = model.train(
        data='datasets/frutas/images', # Tu carpeta confirmada
        epochs=30,                     # Aumentamos épocas porque ahora es más difícil memorizar
        imgsz=224,
        project='runs_zarzamora',
        name='modelo_v5',
        device='cpu',                  # Cambia a 0 si tienes GPU
        
        # --- SECCIÓN MÁGICA: DATA AUGMENTATION ---
        # Estos valores generan nuevas fotos en cada vuelta (epoch)
        degrees=15.0,      # Rotar ligeramente la foto (+/- 15°)
        translate=0.1,     # Mover la foto a los lados (simula mal encuadre)
        scale=0.5,         # Zoom in/out (simula estar cerca o lejos)
        fliplr=0.5,        # Espejo horizontal (clave para hojas)
        mosaic=1.0,        # Mezclar 4 fotos en 1 (ayuda muchísimo a generalizar)
        hsv_h=0.015,       # Cambiar tono de color (simula diferentes soles/climas)
        hsv_s=0.4,         # Cambiar saturación
        hsv_v=0.4,         # Cambiar brillo
        erasing=0.4,       # Borrar pequeños pedazos (obliga a la IA a no fijarse en detalles tontos)
    )
    
    print("\n¡Entrenamiento listo!")
    print("El modelo final estará en: runs_zarzamora/modelo_final_augment/weights/best.pt")

if __name__ == "__main__":
    entrenar_profesional()