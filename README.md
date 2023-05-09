# Playing Style-Based Guitar Effects System
 Automatic guitar effects system using Pytorch Deep Learning and Max MSP for Guitar Effects Handling

<a href="https://github.com/Seanmlevine/Playing-Style-Guitar-Effects-System/blob/main/docs/Master_s_Final_Paper.pdf">Project report here.</a>


## Getting Started:
### Audio Files
- The guitar files used are unprocessed guitar tracks from the IDMT-SMT Dataset (subset 4) or were made for this project following similar protocols

- There are 8 genres, they are:
```
{0: 'pop',
 1: 'metal',
 2: 'reggae/ska',
 3: 'classical',
 4: 'country/folk',
 5: 'latin',
 6: 'rock/blues',
 7: 'jazz',
}
```
### Log Mel-Spectrogram Datasets
Datasets can be created using **Build Dataset Handmade.ipynb** or **Build Dataset.ipynb** or by downloading the dataset made during this project at this link. The dataset uses 128^2 chunks or frame sizes

<a href='https://www.dropbox.com/scl/fo/ql4p7q2l363wbq8e73nwv/h?dl=0&rlkey=57ae789wx57gzrs6us85t725g'>Download Here</a>

- electric_guitar = raw guitar tracks from IDMT and made for this project
- handmadeDBL_JohnAll_128 = database file split for training and test with 128^2 spectrogram size and includes all test audio
- handmadeDBL_JohnAll = database file split for training and test with 256^2 spectrogram size and includes all test audio

### Training
- Define Parameters in Paras_nb.py
- Use train.py for training or train_models.ipynb
- Training Logs saved in log fold (loss/accuracy vs epoch on train set and validation set)

### Test
- Use music_dealer.py to predict the genre components of full song, see **genre_predictor.ipynb** and **music_dealer.py** for details
- Use **realtime_predict-recording.ipynb** to have use the continuous testing model in conjunction with the Max MSP model found in the **Max-Patch-Effects-System** folder
- Test result saved in log fold



## Results

- Accuracy
<table>
  <tr>
    <th></th>
    <th>CNN Model</th>
  </tr>
  <tr>
    <td>Test Set</td>
    <td>91.3%</td>
  </tr>
  <tr>
    <td>Validation Set</td>
    <td>85.6%</td>
  </tr>
</table>


<img src='https://github.com/Seanmlevine/Playing-Style-Guitar-Effects-System/blob/main/docs/Levine_7100_Poster-compressed.pdf' width=500>





# Thanks:
- Nat Condit-Schultz - Associate Professor at Georgia Institute of Technology
- https://github.com/XiplusChenyu/Musical-Genre-Classification#readme - for the CNN architecture example
- https://cycling74.com/tutorials/max-8-guitar-processor-part-1 - for the guitar effects system example
- https://www.idmt.fraunhofer.de/en/publications/datasets/audio_effects.html - for the IDMT-SMT dataset (subset 4)