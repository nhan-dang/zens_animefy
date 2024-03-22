import torch
import torchvision.transforms as transforms, utils
import os
import sys
import inspect
# Add JOJOGan ath
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(current_dir)
jojogan_dir = os.path.join(current_dir, 'JoJoGAN')
sys.path.append(jojogan_dir)
from e4e.models.psp import pSp
from util import *
from argparse import Namespace
from model import *
from copy import deepcopy
import dlib
import numpy as np
from PIL import Image
import scipy.ndimage


@ torch.no_grad()
def modified_projection(img, e4e_encode_path, device='cpu'):
	e4e_ckpt = torch.load(e4e_encode_path, map_location='cpu')
    opts = e4e_ckpt['opts']
    opts['checkpoint_path'] = e4e_encode_path
    opts= Namespace(**opts)
    net = pSp(opts, device).eval().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    img = transform(img).unsqueeze(0).to(device)
    images, w_plus = net(img, randomize_noise=False, return_latents=True)
    return w_plus[0]


def animefy_face_image_based_on_style(img, stylegan2_ckpt_path, e4e_ckpt_path, anime_style_ckpt_path):
	# Define 
	latent_dim = 512
	# Load original generator
	original_generator = Generator(1024, latent_dim, 8, 2).to(device="cpu")
	stylegan2_ckpt = torch.load(stylegan2_ckpt_path, map_location=lambda storage, loc: storage)
	original_generator.load_state_dict(stylegan2_ckpt["g_ema"], strict=False)
	mean_latent = original_generator.mean_latent(10000)
	# to be finetuned generator
	generator = deepcopy(original_generator)
	# cast projection
	img_projection = modified_projection(img, e4e_ckpt_path = e4e_ckpt_path, device="cpu").unsqueeze(0)
	# generate result
	anime_style_ckpt = torch.load(anime_style_ckpt_path, map_location=lambda storage, loc: storage)
	generator.load_state_dict(anime_style_ckpt["g"], strict=False)
	# Set up parameters for generator
	n_sample =  10
	seed = 4000
	torch.manual_seed(seed)
	with torch.no_grad():
	    generator.eval()
	    output = generator(img_projection, input_is_latent=True)
	# Normalize and write output to PIL image
	output = utils.make_grid(output, normalize=True)
	if not isinstance(output, torch.Tensor):
		output = transforms.ToTensor()(output).unsqueeze(0)
	if output.is_cuda:
	    output = output.cpu()
	if output.dim() == 4:
	    output = output[0]
	output = output.permute(1, 2, 0).detach().numpy()
	PIL_image = Image.fromarray((output * 255).astype(np.uint8)).convert('RGB')
	return PIL_image
	PIL_image.save("animefy_results.jpg")

if __name__ == "__main__":
	img_path = os.path.join(os.getcwd(), "face.jpg")
	img = Image.open(img_path)
	stylegan2_ckpt_path = os.path.join(os.getcwd(), "stylegan2-ffhq-config-f.pt")
	e4e_ckpt_path = os.path.join(os.getcwd(), "e4e_ffhq_encode.pt")
	anime_style_ckpth_path = os.path.join(os.getcwd(), "jojo.pt")
	# Generate and save results
	PIL_img = animefy_face_image_based_on_style(img = img, stylegan2_ckpt_path = stylegan2_ckpt_path, e4e_ckpt_path = e4e_ckpt_path, anime_style_ckpt_path = anime_style_ckpt_path)
	print("Saving result as JPG format")
	PIL_img.save("animefy_results.jpg")
	print("Done")
