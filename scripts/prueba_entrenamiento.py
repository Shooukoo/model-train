from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Cargamos un modelo base. 
    # Usamos 'yolo11n.pt' (Nano), que es el más ligero y rápido para pruebas.
    print("--- Cargando modelo Nano ---")
    model = YOLO('yolo11n.pt') 

    # data='coco8.yaml' dataset de 8 fotos 
    # epochs=5: 5 pasadas de entrenamiento 
    # imgsz=640: Tamaño de la imagen ).
    print("--- Iniciando entrenamiento de prueba ---")
    results = model.train(data='frutas_config.yaml', epochs=50, imgsz=640)

    print("--- ¡Prueba finalizada con éxito! ---")