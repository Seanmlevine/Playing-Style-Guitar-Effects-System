import matplotlib.pyplot as plt
from models import CnnModel, CrnnLongModel, CrnnModel, RnnModel
from train import main_train, validate_test, record_matrix
from Paras import Para
from data_loader import torch_dataset_loader
import torch


Para.learning_rate = 1e-5
Para.batch_size = 20
Para.epoch_num = 5

train_loader = torch_dataset_loader(
    Para.A_TRAIN_DATA_PATH, Para.batch_size, True, Para.kwargs)
validation_loader = torch_dataset_loader(
    Para.A_VAL_DATA_PATH, Para.batch_size, False, Para.kwargs)
test_loader = torch_dataset_loader(
    Para.A_TEST_DATA_PATH, Para.batch_size, False, Para.kwargs)


CNN = CnnModel()

res = main_train(model=CNN,
                 train_loader=train_loader,
                 valid_loader=validation_loader,
                 log_name='CnnModel.json',
                 save_name='CnnModel.pt',
                 lr=Para.learning_rate,
                 epoch_num=Para.epoch_num)

        

plt.figure()
plt.plot(res['train_accu'], c='b', label='Training set accuracy')
plt.plot(res['valid_accu'], c='r', label='Validation set accuracy')
plt.title('Accuracy vs Epochs / CNN Model')
plt.legend()
plt.show()
