# pytorch implementation of Grad-Cam on VGG-19 trained on ImageNet


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import vgg19

import numpy as np
import cv2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

transform = transforms.Compose([transforms.Resize((224,224)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# 1 image dataset
dataset = datasets.ImageFolder(root = 'data/', transform = transform)
dataloader = DataLoader(dataset = dataset, batch_size = 1)


# last activation map from the convolution layer is to be used for 
# getting the heatmap of its gradients wrt to the predicted class logits

class VGG(nn.Module):

	def __init__(self):
		super(VGG, self).__init__()

		self.vgg = vgg19(pretrained = True) # get weights trained on ImageNet
		self.features_conv = self.vgg.features[:36]
		self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False)
		self.classifier = self.vgg.classifier
		self.gradients = None

	def get_hook(self, gradients):
		self.gradients = gradients

	def forward(self, x):
		x = self.features_conv(x)
		h = x.register_hook(self.get_hook)
		x = self.max_pool(x)
		x = x.view((1,-1))
		x = self.classifier(x)
		return x

	def get_gradients(self):
		return self.gradients

	def get_last_conv_features(self, x):
		return self.features_conv(x)


vgg = VGG()
vgg.eval()
img, _ = next(iter(dataloader))
pred_logits = vgg(img)
pred_class = pred_logits.argmax(dim = 1)
print("predicted class : ", pred_class.item())


pred_logits[:,36].backward()
gradients = vgg.get_gradients() # gradients wrt to the last map of conv features
print("size of gradients of last conv feature map : " , gradients.size()) # 1 x 512 x 14 x 14

pooled_gradients = torch.mean(gradients, dim = [0,2,3]) # 512
activations = vgg.get_last_conv_features(img)
print("size of last conv feature map : ", activations.size()) # 1 x 512 x 14 x 14

# weighting the channels by corresponding gradients
for i in range(512): # iter over channels
	activations[:,i,:,:] *= pooled_gradients[i]

heatmap = torch.mean(activations, dim = 1).squeeze().detach()
print(heatmap.size()) # 14 x 14
heatmap = np.maximum(heatmap, 0)
heatmap /= torch.max(heatmap)
plt.matshow(heatmap)
plt.show()


# interpolating heat map and projecting it on the image


img = cv2.imread('data/Elephant/elephant.jpeg')
heatmap = cv2.resize(np.array(heatmap), (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('output.jpeg', superimposed_img)