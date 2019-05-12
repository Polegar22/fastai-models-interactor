import base64

from fastai.vision import *

from src import learner_creator


async def reconstruct_image(path, data):
    learn = await learner_creator.setup_learner(path)
    only_base_64 = data['image'].replace('data:image/png;base64,', '')
    decoded = base64.b64decode(only_base_64)
    bytes_io = BytesIO(decoded)
    img = open_image(bytes_io)
    prediction = learn.predict(img)[0]

    return prediction
