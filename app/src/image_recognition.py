from fastai.vision import *
from src import learner_creator


async def predict_image(path, img_bytes):
    learn = await learner_creator.setup_learner(path)
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]

    if prediction.obj == 'beauty':
        response = 'This is my taste ! '
    else:
        response = 'Well ... you can do better.'

    return response
