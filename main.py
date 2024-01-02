# %%
from roboflow import Roboflow
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image

# %%
## Dataset
rf = Roboflow(api_key="API")
project = rf.workspace("acqmal").project("bisindo-classication")
dataset = project.version(1).download("yolov8")

# %%
## Model YoloV8
model = YOLO("yolov8s-seg.pt")
results = model.train(data='Bisindo-classication-1/data.yaml', epochs=100, imgsz=640)

# %%
## Visual result
fig, axs = plt.subplots(3, 1, figsize=(15, 15))

cm = Image.open("runs/segment/train/confusion_matrix.png")
result = Image.open("runs/segment/train/results.png")
predict = Image.open("runs/segment/train/val_batch0_pred.jpg")

axs[0].imshow(cm)
axs[0].set_title("confusion matrix")
axs[1].imshow(result)
axs[1].set_title("result")
axs[2].imshow(predict)
axs[2].set_title("predict")

fig.suptitle("Result", fontsize=24, fontweight="bold")
fig.show()



# %%
