from ultralytics import YOLO

if __name__ == '__main__':
    print("--- Cargando modelo Nano ---")
    model = YOLO('yolo26s-seg.pt') 

    print("--- Iniciando entrenamiento de prueba ---")
    results = model.train(data='dataset_combinado_ago/data.yaml', epochs=100, imgsz=640,batch=16,device=0)

    print("--- ¡Prueba finalizada con éxito! ---")