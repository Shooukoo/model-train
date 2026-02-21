from ultralytics import YOLO

# Load a model

model = YOLO("runs\\segment\\train3\\weights\\best.pt")  # load a custom model

# Predict with the model
results = model.predict(
    source=r"dataset_seg\\boton.jpeg",
    save=True,      # GUARDA la imagen
    show=False,      # no la abre, solo guarda
    conf=0.10,       # confidence threshold (default 0.25)
)

print("Listo. Revisa: runs/segment/predict/")
# Access the results
for result in results:
    xy = result.masks.xy  # mask in polygon format
    xyn = result.masks.xyn  # normalized
    masks = result.masks.data  # mask in matrix format (num_objects x H x W)