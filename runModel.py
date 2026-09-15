import torch
import main
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

#Load model (given its here)
model = main.NeuralNetwork().to(main.device)
model.load_state_dict(torch.load("model.pth", weights_only=True))

# classifications for AI
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = main.test_data[0][0], main.test_data[0][1]
with torch.no_grad():
    x = x.to(main.device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')