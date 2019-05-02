import librosa
import librosa.display

from fastai.vision import *

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
