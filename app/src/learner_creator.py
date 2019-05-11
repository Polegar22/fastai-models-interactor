from fastai.vision import *


async def setup_learner(path_to_model):
    defaults.device = torch.device('cpu')
    learn = load_learner(path_to_model)
    return learn
