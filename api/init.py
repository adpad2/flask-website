from api.projects.athlete_progan.model import Generator
import torch

def init_generator(generator_path):
    netG = Generator(32, 256, 16)
    netG.load_state_dict(torch.load(generator_path, weights_only=True))
    netG.eval()
    return netG
