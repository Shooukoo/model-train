from ultralytics import YOLO

if __name__ == '__main__':
    print("--- Cargando modelo Nano ---")
    model = YOLO('yolov8m-seg.pt') 

    print("--- Iniciando entrenamiento de prueba ---")
    results = model.train(data='dataset_seg/data.yaml', epochs=50, imgsz=640,device=0)

    print("--- ¡Prueba finalizada con éxito! ---")