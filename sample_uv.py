import sys 
sys.path.append('./stylegan3/')

import torch 
import dnnlib
import numpy as np
import PIL.Image
import legacy

network_pkl = './models/hands_texture_model.pkl'
print('Loading networks from "%s"...' % network_pkl)
device = torch.device('cuda')
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
label = torch.zeros([1, G.c_dim], device=device)


def sample_uv(seed):
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    img = G(z, label, truncation_psi=1, noise_mode='const')
    
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
    return img
