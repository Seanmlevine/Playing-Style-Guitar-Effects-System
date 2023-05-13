from music_dealer import MusicDealer
from models import CnnModel
from Paras_nb import Para
import time

import pyaudio
import wave
from pythonosc.udp_client import SimpleUDPClient


# Detect Audio Interface
p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

for i in range(0, numdevices):
    if (p.get_device_info_by_host_api_device_index(0, i).get('name')) == 'Scarlett 18i8 USB':
        device_index = p.get_device_info_by_host_api_device_index(0, i).get('index') # save device index
        device = p.get_device_info_by_host_api_device_index(0, i) # save device
    # if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
    #     print("Input Device id ", i, " - ",
    #         p.get_device_info_by_host_api_device_index(0, i).get('name'))

chunk = 2048  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 32 bits per sample
channels = 1
fs = 44100  # Record at 44100 samples per second
seconds = 10
filename = "guitar_recording.mp3"
p = pyaudio.PyAudio()  # Create an interface to PortAudio
    

print('Recording in...\n')
time.sleep(1)
print('3\n')
time.sleep(1)
print('2\n')
time.sleep(1)
print('1\n')
time.sleep(1)
print('Recording...', '\n\n')

stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                input_device_index=device_index,
                frames_per_buffer=chunk,
                input=True)

frames = []  # Initialize array to store frames

# Store data in chunks for 10 seconds
for i in range(0, int(fs / chunk * seconds)):
    data = stream.read(chunk)
    # data = pcm2float(data)
    # decoded = np.frombuffer(data, dtype='float32')
    frames.append(data)

# Stop and close the stream
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
p.terminate()

print('Finished recording')


# Save the recorded data as a WAV file
wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()


# Initialize Model Prediction
WEIGHT_PATH = "./model/"
cnn_dealer = MusicDealer(WEIGHT_PATH + "CnnModel_guitar_JohnAll.pt", CnnModel())

# output prediction class index and score
res1, res2, res3, score = cnn_dealer.get_genre(filename)

ip = "127.0.0.1"
port = 1338

client = SimpleUDPClient(ip, port)  # Create client
client.send_message("/setpreset", res1)
