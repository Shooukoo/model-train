from ultralytics import YOLO

# Load a model

model = YOLO("runs\\segment\\train2\\weights\\best.pt")  # load a custom model

# Predict with the model
results = model.predict(
    source=r"dataset_seg\\126.jpeg",
    save=True,      # GUARDA la imagen
    show=False      # no la abre, solo guarda
)

print("Listo. Revisa: runs/segment/predict/")
# Access the results
for result in results:
    xy = result.masks.xy  # mask in polygon format
    xyn = result.masks.xyn  # normalized
    masks = result.masks.data  # mask in matrix format (num_objects x H x W)