from ultralytics import YOLO

if __name__ == '__main__':
    print("--- Cargando modelo Nano ---")
    model = YOLO('yolo26s-seg.pt') 

    print("--- Iniciando entrenamiento de prueba ---")
    results = model.train(data='dataset_nvo/data.yaml', epochs=100, imgsz=640,batch=8,device=0)

    print("--- ¡Prueba finalizada con éxito! ---")