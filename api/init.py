from api.projects.athlete_progan.model import Generator
import torch
from api.projects.athlete_progan.face_restorer.face_enhancement import FaceEnhancement

def init_generator(model_path):
    netG = Generator(32, 256, 16)
    netG.load_state_dict(torch.load(model_path))
    netG.eval()
    return netG

def init_face_restorer():
    face_restorer = FaceEnhancement(in_size=256, model='GPEN-BFR-256', use_sr=True, sr_scale=2)
    return face_restorer
