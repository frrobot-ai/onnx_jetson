#%%
# Importing libs
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

batch_size = 32
num_labels = 131
image_size = 224
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_path = '../onnx/vit_fruits_man-'+str(num_labels)+'.onnx'
dataset_path = '../data/fruits/fruits-360_dataset/fruits-360/Test'
#%%
# Check the onnx model
# import onnx

onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)

#%%
# Getting data
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
example_data, example_targets = example_data.to(device), example_targets.to(device)

#%%
# Run the onnx model
import onnxruntime

ort_session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(example_data)}
ort_outs = ort_session.run(None, ort_inputs)
onnx_out = ort_outs[0]           # onnx model output
_, onnx_pred = torch.max(torch.from_numpy(onnx_out), 1)

#%%
# Visualizing the prediction result of the onnx model
example_data = example_data.detach().cpu()
import matplotlib.pyplot as plt
fig = plt.figure()
# Draw the first 4 of a batch
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.tight_layout()
    plt.imshow(np.transpose(example_data[i].numpy(), [1,2,0]), interpolation='none')
    plt.title("Prediction: {}\nGroundtruth: {}".format(onnx_pred[i], example_targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()
# %%
plt.savefig("images/apple_prediction.png")