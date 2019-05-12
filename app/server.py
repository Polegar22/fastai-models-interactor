import base64

import uvicorn
from fastai.vision import *
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.staticfiles import StaticFiles

from src import image_reconstruction, image_recognition, nlp, sound_recognition

path = Path(__file__).parent
app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


@app.route('/')
def index(request):
    html = path / 'view' / 'index.html'
    return HTMLResponse(html.open().read())


@app.route('/image-recognition', methods=['GET'])
def index(request):
    html = path / 'view' / 'index.html'
    return HTMLResponse(html.open().read())


@app.route('/sound-detection', methods=['GET'])
def index(request):
    html = path / 'view' / 'index.html'
    return HTMLResponse(html.open().read())


@app.route('/nlp', methods=['GET'])
def index(request):
    html = path / 'view' / 'index.html'
    return HTMLResponse(html.open().read())


@app.route('/image-reconstruction', methods=['GET'])
def index(request):
    html = path / 'view' / 'index.html'
    return HTMLResponse(html.open().read())


@app.route('/api/image-recognition', methods=['POST'])
async def classify_image(request):
    data = await request.form()
    img_bytes = await (data['file'].read())

    response = await image_recognition.predict_image(path / 'models/image-recognition', img_bytes)

    return JSONResponse({'result': response})


# Forced to put the fake FeatureLoss here, otherwise the reconstruction model can't retrieve it
class FeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return input

    def __del__(self): self.hooks.remove()


@app.route('/api/image-reconstruction', methods=['POST'])
async def reconstruct_image(request):
    filename = 'prediction.jpg'
    path_to_file = path / 'data/image-reconstruction' / filename
    data = await request.form()

    reconstructed = await image_reconstruction.reconstruct_image(path / 'models/image-reconstruction', data)

    reconstructed.save(path_to_file)
    with open(path_to_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return Response(encoded_string, media_type='application/octet-stream')


@app.route('/api/sound-detection', methods=['POST'])
async def classify_sound(request):
    data = await request.form()
    fileitem = data['audio']

    if fileitem.filename:
        audio_path = path / 'data/sound-detection' / fileitem.filename
        open(audio_path, 'wb').write(fileitem.file.read())
        prediction = await sound_recognition.classify_sound(path / 'models/sound-detection', audio_path)
        return JSONResponse({'result': 'This is a : ' + prediction})


@app.route('/api/nlp', methods=['POST'])
async def generate_text(request):
    data = await request.form()
    generated_text = await nlp.nlp_generation(path / 'models/nlp', data)
    return JSONResponse(
        {'result': generated_text + ' ...'})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app, host='0.0.0.0', port=8080)
