
import numpy as np
import librosa
import librosa.display as display
import matplotlib.pyplot as plt
import os
import h5py
import time
import wav2mp3
import random

class DatasetBuilder:
    def __init__(self, data_path='electric_guitar', data_set_path='handmadeDBL_JohnAll_128/', size=128):  # mel-size 128
        #self.data_path = '../electric_guitar' # location of dataset
        #self.data_set_path = '../handmadeDBL_JohnAll_128/' #change for each dataset change

        self.data_path = data_path
        self.data_set_path = data_set_path
        self.index = size

    def run(self):
        self.genre_path_dict, self.label_dict, self.total_file = self.file_analysis()
        self.genreDictionary_init()

        self.signal, self.chunk_num = self.test_frame_feature()
        self.mel_shape, self.count = self.separateDatasets()
        self.dataset, self.fin_count = self.combineDataset(self.mel_shape)
        self.files, self.sets, self.train_set, self.valid_set, self.test_set = self.train_test_split(self.fin_count)
        self.shuffle_and_create()

        

    def file_analysis(self):
        self.genre_path_dict = dict()
        self.label_dict = dict()

        for (dirpath, dirnames, filenames) in os.walk(self.data_path):
            if dirnames:
                for idx, genre in enumerate(dirnames):
                    self.genre_path_dict[genre] = list()
                    self.label_dict[genre] = idx
                continue
            else:
                genre = dirpath.split('/')[-1]
                for au_file in filenames:
                    if 'mp3' in au_file:
                        current_path = dirpath + '/' + au_file
                        self.genre_path_dict[genre].append(current_path)
                    if 'wav' in au_file:
                        current_path = dirpath + '/' + au_file
                        self.genre_path_dict[genre].append(current_path)


        total_file = sum([len(self.genre_path_dict[genre]) for genre in self.genre_path_dict])
        return self.genre_path_dict, self.label_dict, total_file

    def genreDictionary_init(self):
        genre_dict = dict()
        for key in self.label_dict:
            genre_dict[self.label_dict.get(key)] = key


# %% [markdown]
# Create log-Mel spectrogram


    # This is the newer one
    # changed to allow, spectrogram size as a input parameter 
    def frame_feature_extractor(self, file_path, size):
        # global varibles:
        sr = 16000
        cut_time = 0  # throw first and last 10s
        cut_wave_length = cut_time * sr
        n_ftt = 256
        hop = n_ftt // 8

        signal, _ = librosa.load(file_path, sr=sr)
        # signal = signal[cut_wave_length: -1 * cut_wave_length] # cut wave length
        
        S = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=size).T
        S = librosa.power_to_db(S)
        if not S.shape[0] % size == 0:
            S = S[:-1 * (S.shape[0] % size)] # divide the mel spectrogram
        chunk_num = int(S.shape[0] / size) # was 64 in previous training
        mel_chunks = np.split(S, chunk_num) # create 128 * 128 data frames
        return mel_chunks, chunk_num

    def test_frame_feature(self, display = False):
        self.signal, chunk_num = self.frame_feature_extractor(self.genre_path_dict['rock_blues'][2], 128)

        if display:
            plt.title('Rock_Blues2')
            display.specshow(self.signal[1], y_axis='time')
            plt.show()
        return self.signal, chunk_num

# %% [markdown]
# ## Build datasets

    def build_tag(self, genre):
        target = np.zeros(len(self.label_dict), dtype=int)
        pos = self.label_dict.get(genre)
        target[pos] = 1
        return target

# %% [markdown]
# ## Create separate dataset for each file

# %%
    def separateDatasets(self):
        count = 0
        file_index = 0
        mel_shape = self.signal[0].shape

        #if path doesn’t exist we create a new path
        try:
            os.makedirs(self.data_set_path)
        except FileExistsError:
        # directory already exists
            print('Folder needs to be empty to create new dataset')
            pass

        for key in self.label_dict:
            print('deal with {0}'.format(key))
            for file_i, file in enumerate(self.genre_path_dict.get(key)):
                mel_list, chunk_number = self.frame_feature_extractor(file,128)
                dataset_name = self.data_set_path + str(file_index) + '.h5'
                current_dataset = h5py.File(dataset_name, 'a')
                
                current_dataset.create_dataset('mel', shape=(chunk_number, mel_shape[0], mel_shape[1]), dtype=np.float32)
                current_dataset.create_dataset('tag', shape=(chunk_number, 8), dtype=int)   
                
                for i, mel_signal_chunk in enumerate(mel_list):
                    current_dataset['tag'][i] = self.build_tag(key)
                    current_dataset['mel'][i] = mel_signal_chunk
                    count += 1
                
                print('->{0}'.format(file_i), end='')
                current_dataset.close()
                file_index += 1
                
            print(' ')

        print('Total spectrograms: ' + str(count))
        print('Total files: ' + str(file_index))
        return mel_shape, count


    # %% [markdown]
    # ### load data in one set

    def combineDataset(self, mel_shape, count):
        all_setpath = self.data_set_path + 'hand_all.h5'
        all_dataset = h5py.File(all_setpath, 'a')

        all_dataset.create_dataset('mel', shape=(
            count, mel_shape[0], mel_shape[1]), dtype=np.float32)
        all_dataset.create_dataset('tag', shape=(count, 8), dtype=int)
                

        count = 0
        for (dirpath, dirnames, filenames) in os.walk(self.data_set_path):
            if filenames:
                for set_name in filenames:
                    if 'h5' not in set_name or 'all' in set_name:
                        continue
                    set_path = self.data_set_path + set_name
                    tmp_dataset = h5py.File(set_path, 'r')
                    tmp_count = tmp_dataset['mel'].shape[0]
                    for i in range(tmp_count):
                        all_dataset['mel'][count] = tmp_dataset['mel'][i]
                        all_dataset['tag'][count] = tmp_dataset['tag'][i]
                        count += 1
                    tmp_dataset.close()
        all_dataset.close()
        print('Total Spectrograms in Combined Dataset: ' + str(count))

        setpath1 = self.data_set_path + 'hand_all.h5'
        setpath2 = self.data_set_path + 'all_long.h5'
        db1 = h5py.File(setpath1, 'r')
        # db2 = h5py.File(setpath2, 'r')
        all_chunk = count + 3000

        fin_setpath = self.data_set_path + 'fin_all.h5'
        dataset = h5py.File(fin_setpath, 'a')
        dataset.create_dataset('mel', shape=(all_chunk, mel_shape[0], mel_shape[1]), dtype=np.float32)
        dataset.create_dataset('tag', shape=(all_chunk, 8), dtype=int) 

        fin_count = 0
        for j in range(count):
            dataset['mel'][fin_count] = db1['mel'][j]
            dataset['tag'][fin_count] = db1['tag'][j]
            fin_count += 1

        db1.close()

        # for j in range(3000):
        #     dataset['mel'][fin_count] = db2['mel'][j]
        #     dataset['tag'][fin_count] = db2['tag'][j]
        #     fin_count += 1

        # db2.close()
        dataset.close()
        # print(fin_count)

        dataset = h5py.File(setpath1, 'r')
        return dataset, fin_count

        # %%
    def train_test_split(self, fin_count):
        train_file = fin_count * 0.7
        valid_file = fin_count * 0.2
        test_file = fin_count - int(train_file) - int(valid_file)

        # %%
        # train_set = '../handmadeDBL2/l_train.h5'
        # valid_set = '../handmadeDBL2/l_valid.h5'
        # test_set = '../handmadeDBL2/l_test.h5'
        train_set = self.data_set_path + 'l_train.h5'
        valid_set =  self.data_set_path + 'l_valid.h5'
        test_set =  self.data_set_path + 'l_test.h5'

        # %%
        files = [int(a) for a in [train_file, valid_file, test_file]]
        sets = [train_set, valid_set, test_set]
        print('Number of files for train, validation, and test, respectively: ' + str(files))
        return files, sets, train_set, valid_set, test_set

# %% [markdown]
# ### Shuffle and create

    def shuffle_and_create(self):
        idx = [i for i in range(self.fin_count)]
        random.seed(516)
        random.shuffle(idx)

        train_idx = idx[:self.files[0]]
        valid_idx = idx[self.files[0]: self.files[0]+self.files[1]]
        test_idx = idx[-self.files[2]:]
        indices = [train_idx, valid_idx, test_idx]

        for i, dset in enumerate(self.sets):
            s_set = h5py.File(dset, 'a')
            indice = indices[i]
            file_num = self.files[i]
            
            s_set.create_dataset('mel', shape=(file_num, self.mel_shape[0], self.mel_shape[1]), dtype=np.float32)
        #     s_set.create_dataset('mfcc', shape=(file_num, mfcc_shape[0], mfcc_shape[1]), dtype=np.float32)
            s_set.create_dataset('tag', shape=(file_num, 8), dtype=int)
            
            count = 0
            for i in indice:
                s_set['mel'][count] = self.dataset['mel'][i]
        #         s_set['mfcc'][count] = dataset['mfcc'][i]
                s_set['tag'][count] = self.dataset['tag'][i]
                count += 1
                
                if count % 10 == 0:
                    print('*', end="")
            
            s_set.close()
            print()  
            print('Create Separate Datasets {0}'.format(dset))
    

# %% [markdown]
# ## Test Here
    def testDataset(self):
        t_set = h5py.File(self.train_set, 'r')
        self.mel_shape

        print(t_set['mel'].shape)
        print(t_set['tag'][:5])

# %%
if __name__ == '__main__':
    datasetBuilder = DatasetBuilder(data_path='electric_guitar', data_set_path='handmadeDBL_JohnAll_256/', size=256)
    datasetBuilder.run()
    datasetBuilder.testDataset()
