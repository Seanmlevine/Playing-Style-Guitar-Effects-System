from genericpath import isfile
from pydub import AudioSegment
import os

from sqlalchemy import false

data_path = './electric_guitar'
for (dirpath, dirnames, filenames) in os.walk(data_path):
    for au_file in filenames:
        current_path = dirpath + '/' + au_file  
        reference_audio = AudioSegment.from_file(current_path)

        # Measure the RMS amplitude of the reference audio file
        reference_rms = reference_audio.rms
        print("RMS: ",reference_rms, "\n")
        print("dBFS: ", reference_audio.dBFS, "\n")
        # if os.path.isfile(name_to_mp3) == False:
        #     AudioSegment.from_wav(current_path).export(
        #         name_to_mp3, format="mp3")
