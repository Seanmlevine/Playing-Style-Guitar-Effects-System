
from music_dealer import MusicDealer
from models import CnnModel, CrnnLongModel, CrnnModel
import json
from Paras_nb import Para
import os
import time
import numpy as np

import pyaudio
import wave

from pydub import AudioSegment

import librosa
import librosa.display as display
import matplotlib.pyplot as plt

from pythonosc.udp_client import SimpleUDPClient
import collections

# %% [markdown]
# Device Setup

# %%
# Check for Audio Interface
p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

for i in range(0, numdevices):
    if (p.get_device_info_by_host_api_device_index(0, i).get('name')) == 'Scarlett 18i8 USB':
        device_index = p.get_device_info_by_host_api_device_index(0, i).get('index')
        device = p.get_device_info_by_host_api_device_index(0, i)
    else:
        device_index = p.get_device_info_by_host_api_device_index(
            0, 1).get('index')
        device = p.get_device_info_by_host_api_device_index(0, 1)
    if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print("Input Device id ", i, " - ",
              p.get_device_info_by_host_api_device_index(0, i).get('name'))


# %% [markdown]
# Check which Device is in Use

# %%
device

# %% [markdown]
# Initialize Interpolation Dictionary with Stored Effects JSON File

# %%
path_to_json_file = "effects_params.json"

with open(path_to_json_file) as f:
    effects_params = json.load(f)

def interpDictionary(dictionaries, percentages):
    interp_dict = collections.defaultdict(dict)
    temp_val = []
    for idx, val in enumerate(dictionaries):
        elem = effects_params['pattrstorage']['slots'][str(val+1)]['data']

        for key, value in elem.items():
            if bool(interp_dict[key]):
                temp_val = interp_dict[key]
                temp_val += (value[0] * percentages[idx]/100.0)
            else:
                temp_val = (value[0] * percentages[idx]/100.0)
            interp_dict.update({key: temp_val})
    return interp_dict


# %% [markdown]
# Continuous Recording!

# %%


# Initialize Model used for Prediction
WEIGHT_PATH = "../model/"
cnn_dealer = MusicDealer(
    WEIGHT_PATH + "CnnModel_guitar_JohnAll_128.pt", CnnModel())

# Begin OSC client for communication to Max MSP
ip = "127.0.0.1"
port = 1338
client = SimpleUDPClient(ip, port)  # Create client

# Initialize Pyaudio stream parameters
CHUNKSIZE = 2**11
SAMPLERATE = 44100
channels = 1
seconds = 9
filename = "guitar_recording.mp3" # a demo filename 
sample_format = pyaudio.paFloat32 # might have to change this to int16
rec_num = 1

# first open preset value
preset_num = 9


p = pyaudio.PyAudio()

# Set up the PyAudio stream
stream = p.open(format=sample_format,
                channels=1,
                rate=SAMPLERATE,
                input_device_index=device_index,
                input=True,
                frames_per_buffer=CHUNKSIZE)

try:
# Loop to continuously capture audio and make predictions
    while True:

        frames = []  # Initialize array to store frames

        # Store data in chunks for 10 seconds
        for i in range(0, int(SAMPLERATE / CHUNKSIZE * seconds)):
            data = stream.read(CHUNKSIZE, exception_on_overflow=False)
            frames.append(data)

        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(SAMPLERATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        print(f"Prediction for Chunk {rec_num}: \n")
        rec_num += 1

        # Make a prediction using the genre predictor model    
        res1, res2, res3, score = cnn_dealer.get_genre(filename)
        print(res1, res2, res3, score,
              "\n********************************************************\n\n")
        
        # Grab the list of genre indices and percentages
        genres = list(score.keys())
        percents = list(score.values())

        # Interpolate the predicted genres by their percentages to create a new effect in between 
        interp_dict = interpDictionary(genres, percents)

        # send to OSC
        for key, value in interp_dict.items():
            client.send_message(key, [preset_num, value])
            # print(key)
            # print(value)

        # Tell Max MSP the storing of new parameters is done
        client.send_message("/end", preset_num-1)
        preset_num += 1

        

except KeyboardInterrupt as e:
    stream.stop_stream()
    stream.close()
    p.terminate()
    print(e)

except Exception as e:
    stream.stop_stream()
    stream.close()
    p.terminate()
    print(e)


