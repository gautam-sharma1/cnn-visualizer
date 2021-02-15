import streamlit as st
from matplotlib import pyplot as plt

import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torchvision.models as models
import torchvision.utils as utils
import torch


alexnet_layers = {"Convolutional Layer 1":0, "Convolutional Layer 2":3, "Convolutional Layer 3":6,"Convolutional Layer 4":8,\
    "Convolutional Layer 5":10}


@st.cache(allow_output_mutation=True)
def plot_kernels(tensor, num_cols=8):
    if not tensor.ndim == 4:
        raise Exception("assumes a 4D tensor")
    if not tensor.shape[-1] == 3:
        tensor = torch.from_numpy(tensor)
        tensor = torch.sum(tensor, dim=3)
        print(tensor.shape)
        tensor = tensor.numpy()
    num_kernels = tensor.shape[0]
    num_rows = 1 + num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols, num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows, num_cols, i + 1)
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig("cnn.png")


@st.cache(allow_output_mutation=True)
def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1):
    n, c, w, h = tensor.shape
    print(tensor.shape)

    if allkernels:
        tensor = tensor.view(n * c, -1, w, h)
        print(tensor.shape)
    elif c != 3:
        #tensor = torch.sum(tensor, dim=1)
        #tensor = tensor.view(tensor.shape[0], 1, tensor.shape[1],tensor.shape[2])
        print("Before:",tensor.shape)
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)
        print("After: ",tensor.shape)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure(figsize=(nrow, rows))
    plt.imshow(grid.detach().numpy().transpose((1, 2, 0)))
    plt.savefig("cnn.png")



st.title('Convolutions Visualizer')

# @st.cache(allow_output_mutation=True)
# def load(scaler_path, model_path):



choice = st.selectbox('Choose any architecture', ["Alexnet"])

if choice == "Alexnet":
    layer = st.selectbox('Select which CNN layer to visualize?', [key for key in alexnet_layers.keys()])



models = {'Alexnet':models.alexnet(pretrained=True)}


uploaded_file = st.file_uploader("Choose a image file", type="png")


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

# st.write("Choose number of kernels")
n_rows = st.sidebar.slider("Number of CNN's per row?", 0, 64, 4, 2)

clicked = st.button("Visualize..")

if clicked:
    model = models[choice]

    tran = transforms.Compose([
        transforms.Resize([int(224), int(224)]),
        transforms.ToTensor()
    ])

    img = tran(image)
    plt.imshow(img.numpy().transpose((1,2,0)))

    img = img.view(1, img.shape[0], img.shape[1], img.shape[2])

    model(img.float())

    filter = model.features[alexnet_layers[layer]].weight
    # filter = filter.view(filter.shape[0], filter.shape[2], filter.shape[3], filter.shape[1])
    # filter = filter.data.numpy()
    # filter = (1 / (2 * 3.69201088)) * filter + 0.5

    #plot_kernels(filter, n_rows)
    visTensor(filter, ch=0, allkernels=False,nrow=n_rows)
    st.write("Visualizing...")
    c_image = Image.open('/Users/gautamsharma/Desktop/Computer-Vision/Pytorch Practice/cnn.png')
    st.image(c_image,  use_column_width=True)


