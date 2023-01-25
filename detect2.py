import cv2
import torch
from models.experimental import attempt_load
from torch.backends import cudnn
from utils.datasets import LoadStreams, LoadImages
from utils.plots import plot_one_box
from utils.torch_utils import select_device, TracedModel
from utils.general import check_img_size, non_max_suppression, scale_coords

def detect():
    weights = "best.pt"
    size = 960

    device = select_device("cpu")

    model = TracedModel(attempt_load(weights, map_location=device), device, size)

    stride = int(model.stride.max())
    size = check_img_size(size, s=stride)

    names = model.names

    cudnn.benchmark = True
    dataset = LoadStreams("0", img_size=size, stride=stride)

    for _, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device).float() / 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = model(img, augment=False)[0]

        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

        for i, det in enumerate(pred):
            im0 = im0s[i].copy()
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    label = f"{names[int(cls)]} {conf:.2f}"
                    plot_one_box(xyxy, im0, label=label, color=(0, 0, 0), line_thickness=2)

            for cls in det[:, -1].unique():
                n = (det[:, -1] == cls).sum()
                print(f"Detected {n} {names[int(cls)]}{'s' * (n > 1)}")

            print()

            cv2.imshow("Image", im0)

    print("Done.")

if __name__ == "__main__":
    with torch.no_grad():
        detect()
