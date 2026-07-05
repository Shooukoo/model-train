from ultralytics import YOLO
import cv2
# Load a model

model = YOLO("runs\\segment\\train2\\weights\\best.pt")  # load a custom model
img = cv2.imread("dataset_seg\\C1_C1.jpeg")
img = cv2.resize(img, (5120, 5120))

# Predict with the model
results = model.predict(
    source=img,     
    save=True,      # GUARDA la imagen
    show=False,      # no la abre, solo guarda
    conf=0.25,       # confidence threshold (default 0.25)
    iou=0.95,        # NMS IoU threshold (default 0.45)
    retina_masks=False,
    show_labels=False,  # no muestra etiquetas
    show_conf=False,     # no muestra confianza
    device='cpu',
    save_dir='runs/segment/predicts'
)

print("Listo. Revisa: runs/segment/predict/")

# Access the results
for result in results:
    xy = result.masks.xy  # mask in polygon format
    xyn = result.masks.xyn  # normalized
    masks = result.masks.data  # mask in matrix format (num_objects x H x W)

