import aiohttp
import uvicorn
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

from app.utils import *

path = Path(__file__).parent
app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)


async def setup_learner(pathToModel):
    defaults.device = torch.device('cpu')
    learn = load_learner(pathToModel)
    return learn


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


@app.route('/api/image-recognition', methods=['POST'])
async def classifyImage(request):
    learn = await setup_learner(path / 'models/image-recognition')
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]

    if prediction.obj == 'beauty':
        response = 'This is my taste ! '
    else:
        response = 'Well ... you can do better.'

    return JSONResponse({'result': response})


@app.route('/api/sound-detection', methods=['POST'])
async def classifySound(request):
    learn = await setup_learner(path / 'models/sound-detection')
    data = await request.form()
    fileitem = data['audio']

    if fileitem.filename:
        audio_path = path / 'data/sound-detection' / fileitem.filename
        open(audio_path, 'wb').write(fileitem.file.read())
        filename = create_spectrograms(audio_path)
        spectogram = open_image(filename)
        pred_class, pred_idx, outputs = learn.predict(spectogram)

        if not pred_class.obj:
            pred_class.obj = SOUND_TYPES[(outputs == torch.max(outputs)).nonzero()]
        else:
            pred_class.obj = ', '.join(pred_class.obj)

        return JSONResponse({'result': 'This is a : ' + pred_class.obj})


@app.route('/api/nlp', methods=['POST'])
async def generateText(request):
    learn = await setup_learner(path / 'models/nlp')
    data = await request.form()
    entry_text = data['entry_text']
    if entry_text:
        return JSONResponse({'result': learn.predict(entry_text, 45, temperature=0.75) + ' ...'})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app, host='0.0.0.0', port=8080)
