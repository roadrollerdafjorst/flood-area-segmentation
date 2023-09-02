import torch
from archs.unet_arch import UNET, convBlock
from PIL import Image
import numpy as np
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 320
IMAGE_WIDTH = 480
transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ],
)

# Load the model
def load_unet(model_path, device="cuda"):
    model = UNET(in_channels=3, out_channels=1).to(device)
    # PATH = "models/unet-01.pth.tar"
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["state_dict"])
    return model

model = UNET(in_channels=3, out_channels=1).to(device)

# Load the trained weights
PATH = "models/unet-01.pth.tar"
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint["state_dict"])


def get_image(img_path):
    img = np.array(Image.open(img_path).convert("RGB"))
    img = transform(image=np.array(img))["image"].unsqueeze(0)
    return img

def get_image_from_array(img):
    img = Image.fromarray(img)
    img = np.array(img.convert("RGB"))
    img = transform(image=np.array(img))["image"].unsqueeze(0)
    return img

def generate_mask_array(img, model, device="cuda"):
    model.eval()
    img = img.to(device)
    with torch.no_grad():
        preds = torch.sigmoid(model(img))
        preds = (preds>0.5).float()
    # model.train()
    return preds

def generate_mask_img(img, model, device="cuda"):
    # img = get_image(img_path)
    preds = generate_mask_array(img, model, device).cpu()
    mask = Image.fromarray(np.array(preds.cpu().squeeze()*255.0, dtype=np.uint8))
    return mask

# img = get_image("sample.jpg")
# print(img.shape)
# preds = generate_mask_array(img, model, device).cpu()
# print(preds[0])
# print(preds.shape)
# op = Image.fromarray(np.array(preds.cpu().squeeze()*255.0, dtype=np.uint8))
# op.save("op.png")
# print