import gdown
import os

os.makedirs("models", exist_ok=True)

url = "https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812"
output = "models/bisenet_resnet18.pth"

print("Downloading BiSeNet weights (~50MB)...")
gdown.download(url, output, quiet=False)
print("Done — saved to models/bisenet_resnet18.pth")