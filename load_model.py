"""
main testing code
"""

import numpy as np
from scipy import linalg
import pickle
import torch
from skimage import io
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage.transform import resize
from skimage.metrics import *
from cbam import *
import AKSVD_function

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Overcomplete Discrete Cosinus Transform:
patch_size = 8
m = 16
Dict_init = AKSVD_function.init_dct(patch_size, m)
Dict_init = Dict_init.to(device)

# Squared Spectral norm:
c_init = linalg.norm(Dict_init, ord=2) ** 2
c_init = torch.FloatTensor((c_init,))
c_init = c_init.to(device)

# Average weight:
w_init = torch.normal(mean=1, std=1 / 10 * torch.ones(patch_size ** 2)).float()
w_init = w_init.to(device)

D_in, H_1, H_2, H_3, D_out_lam, T, min_v, max_v = patch_size ** 2, 512,1024,512,256, 5, -1, 1
model = AKSVD_function.DenoisingNet_MLP(
    patch_size,
    D_in,
    H_1,
    H_2,
    H_3,
    D_out_lam,
    T,
    min_v,
    max_v,
    Dict_init,
    c_init,
    w_init,
    #w_2_init,
    #w_3_init,
    device,
)

model.load_state_dict(torch.load("../model_9_10.pth", map_location="cpu"))
model.to(device)
model.eval()

# Test image names:
file_test = open("test_set12.txt", "r")
onlyfiles_test = []
for e in file_test:
    onlyfiles_test.append(e[:-1])

# Rescaling in [-1, 1]:
mean = 255 / 2
std = 255 / 2
data_transform = transforms.Compose(
    [AKSVD_function.Normalize(mean=mean, std=std), AKSVD_function.ToTensor()]
)
# Noise level:
sigma = 25

def InNormalization(image):
    mean=255/2
    std=255/2
    return image*std+mean


# Test Dataset:
my_Data_test = AKSVD_function.FullImagesDataset(
    root_dir="Set12", image_names=onlyfiles_test, sigma=sigma, transform=data_transform
)

dataloader_test = DataLoader(my_Data_test, batch_size=1, shuffle=False, num_workers=0)

# List PSNR:
file_to_print = open("list_test_PSNR_set12_25.csv", "w")
file_to_print.write(str(device) + "\n")
file_to_print.flush()

with open("list_test_PSNR_set12_25.txt", "wb") as fp:
    with torch.no_grad():
        list_PSNR = []
        list_PSNR_init = []
        list_SSIM = []
        PSNR = 0
        for k, (image_true, image_noise) in enumerate(dataloader_test, 0):

            image_true_t = image_true[0, 0, :, :]
            image_true_t = image_true_t.to(device)

            image_noise_0 = image_noise[0, 0, :, :]
            image_noise_0 = image_noise_0.to(device)

            image_noise_t = image_noise.to(device)
            image_restored_t = model(image_noise_t)
            image_restored_t = image_restored_t[0, 0, :, :]


            PSNR_init = 10 * torch.log10(
                4 / torch.mean((image_true_t - image_noise_0) ** 2)
            )
            img1 = InNormalization(image_true_t).numpy().astype('uint8')
            img_noise1 = InNormalization(image_noise_0).numpy().astype('uint8')
            real_denoise = InNormalization(image_restored_t).numpy().astype('uint8')
            
            SSIM_init = structural_similarity(img1, img_noise1, data_range=255) #convert to image

            print(k,onlyfiles_test[k])
            file_to_print.write("name:"+ onlyfiles_test[k]+ "\n")
            file_to_print.write("Init_PSNR:" + " " + str(PSNR_init) + "\n")
            file_to_print.write("Init_SSIM:" + " " + str(SSIM_init) + "\n")
            file_to_print.flush()

            list_PSNR_init.append(PSNR_init)


            PSNR = 10 * torch.log10(
                4 / torch.mean((image_true_t - image_restored_t) ** 2)
            )
            PSNR = PSNR.cpu()
            SSIM_test = structural_similarity(img1,real_denoise, data_range=255)
            file_to_print.write("Test_PSNR:" + " " + str(PSNR) + "\n")
            file_to_print.write("Test_SSIM:" + " " + str(SSIM_test) + "\n")
            file_to_print.flush()

            list_PSNR.append(PSNR)
            list_SSIM.append(SSIM_test)


    mean_PSNR = np.mean(list_PSNR)
    mean_SSIM = np.mean(list_SSIM)
    file_to_print.write("FINAL_PSNR" + " " + str(mean_PSNR) + "\n")
    file_to_print.write("FINAL_SSIM" + " " + str(mean_SSIM) + "\n")
    file_to_print.flush()
    pickle.dump(list_PSNR, fp)
