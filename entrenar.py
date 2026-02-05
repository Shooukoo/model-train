from ultralytics import YOLO

def entrenar_profesional():
    # Usamos el modelo nano (n) para velocidad, o small (s) para más inteligencia
    model = YOLO('yolov8n-cls.pt') 

    print("--- INICIANDO ENTRENAMIENTO CON AUMENTO DE DATOS ---")
    print("El sistema generará variaciones de tus fotos automáticamente.")

    results = model.train(
        data='datasets/frutas/images', 
        epochs=50,                     # Aumentamos épocas porque ahora es más difícil memorizar
        imgsz=224,
        project='runs_zarzamora',
        name='modelo_v7',
        device='cpu',                  # Cambia a 0 si tienes GPU
        
        augment=True,                  # Activar aumento de datos
        dropout=0.5,                   # Desactivar algunas neuronas al azar
        patience=10,                   # Detener si no hay mejora en 10 épocas
        degrees=15.0,      # Rotar ligeramente la foto 
        translate=0.1,     # Mover la foto a los lados 
        scale=0.5,         # Zoom in/out 
        fliplr=0.5,        # Espejo horizontal 
        mosaic=1.0,        # Mezclar 4 fotos en 1 
        hsv_h=0.015,       # Cambiar tono de color 
        hsv_s=0.4,         # Cambiar saturación
        hsv_v=0.4,         # Cambiar brillo
        erasing=0.4,       # Borrar pequeños pedazos 
    )
    
    print("\n¡Entrenamiento listo!")
    print("El modelo final estará en: runs_zarzamora/modelo_final_augment/weights/best.pt")

if __name__ == "__main__":
    entrenar_profesional()