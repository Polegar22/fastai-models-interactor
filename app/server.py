import librosa
import librosa.display
import uvicorn
from fastai.vision import *
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
import base64

SOUND_TYPES = ['Accelerating_and_revving_and_vroom',
               'Accordion',
               'Acoustic_guitar',
               'Applause',
               'Bark',
               'Bass_drum',
               'Bass_guitar',
               'Bathtub_(filling_or_washing)',
               'Bicycle_bell',
               'Burping_and_eructation',
               'Bus',
               'Buzz',
               'Car_passing_by',
               'Cheering',
               'Chewing_and_mastication',
               'Child_speech_and_kid_speaking',
               'Chink_and_clink',
               'Chirp_and_tweet',
               'Church_bell',
               'Clapping',
               'Computer_keyboard',
               'Crackle',
               'Cricket',
               'Crowd',
               'Cupboard_open_or_close',
               'Cutlery_and_silverware',
               'Dishes_and_pots_and_pans',
               'Drawer_open_or_close',
               'Drip',
               'Electric_guitar',
               'Fart',
               'Female_singing',
               'Female_speech_and_woman_speaking',
               'Fill_(with_liquid)',
               'Finger_snapping',
               'Frying_(food)',
               'Gasp',
               'Glockenspiel',
               'Gong',
               'Gurgling',
               'Harmonica',
               'Hi-hat',
               'Hiss',
               'Keys_jangling',
               'Knock',
               'Male_singing',
               'Male_speech_and_man_speaking',
               'Marimba_and_xylophone',
               'Mechanical_fan',
               'Meow',
               'Microwave_oven',
               'Motorcycle',
               'Printer',
               'Purr',
               'Race_car_and_auto_racing',
               'Raindrop',
               'Run',
               'Scissors',
               'Screaming',
               'Shatter',
               'Sigh',
               'Sink_(filling_or_washing)',
               'Skateboard',
               'Slam',
               'Sneeze',
               'Squeak',
               'Stream',
               'Strum',
               'Tap',
               'Tick-tock',
               'Toilet_flush',
               'Traffic_noise_and_roadway_noise',
               'Trickle_and_dribble',
               'Walk_and_footsteps',
               'Water_tap_and_faucet',
               'Waves_and_surf',
               'Whispering',
               'Writing',
               'Yell',
               'Zipper_(clothing)']

path = Path(__file__).parent
app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


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


@app.route('/image-reconstruction', methods=['GET'])
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


class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()

    def forward(self, input, target):
        return input

    def __del__(self): self.hooks.remove()


@app.route('/api/image-reconstruction', methods=['POST'])
async def reconstructImage(request):
    learn = await setup_learner(path / 'models/image-reconstruction')
    data = await request.form()
    onlyBase64 = data['image'].replace('data:image/png;base64,', '')
    decoded = base64.b64decode(onlyBase64)
    bytes = BytesIO(decoded)
    img = open_image(bytes)
    prediction = learn.predict(img)[0]
    prediction.show(figsize=(10, 5), title='Restored')
    if prediction:
        return JSONResponse({'result': 'cc'})


def create_spectrograms(audio_path):
    print(f'Processing spectogram')
    filename = audio_path.with_suffix('.png')
    samples, sample_rate = librosa.load(audio_path, duration=4.0)
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=samples, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close('all')
    return filename


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
    nb_words = data['nb_words']
    randomness = data['randomness']
    if entry_text:
        return JSONResponse(
            {'result': learn.predict(entry_text, int(nb_words), temperature=float(randomness)) + ' ...'})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app, host='0.0.0.0', port=8080)
