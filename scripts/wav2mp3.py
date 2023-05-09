# This code is used to convert all audio files that are wav into mp3 - since the IDMT SMT Dataset is all in wav files

from genericpath import isfile
from pydub import AudioSegment
import os

from sqlalchemy import false

data_path = './electric_guitar' 

for (dirpath, dirnames, filenames) in os.walk(data_path):
    for au_file in filenames:
        current_path = dirpath + '/' + au_file
        name_to_mp3 = current_path.replace("wav", "mp3")
        if os.path.isfile(name_to_mp3) == False:
            AudioSegment.from_wav(current_path).export(name_to_mp3, format="mp3")
            