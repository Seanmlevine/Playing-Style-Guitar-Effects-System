import torch.nn as nn
import torch
torch.manual_seed(1)


class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        cov1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov1.weight)
        self.convBlock1 = nn.Sequential(cov1,
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2))

        cov2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov2.weight)
        self.convBlock2 = nn.Sequential(cov2,
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2))

        cov3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov3.weight)
        self.convBlock3 = nn.Sequential(cov3,
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=4))

        cov4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov4.weight)
        self.convBlock4 = nn.Sequential(cov4,
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=4)) # try kernel size 8 but can reduce clarity
        
        # try 1 to 2 layers with 2 linear layers to reduce parameters and can help with over fitting
        
        # reduce number of convolutional layers, number of neurons, reduce learning rate, filter size of convolution (stride size)

        # Less efficient method of linearization
        # self.fcBlock1 = nn.Sequential(nn.Linear(in_features=8192, out_features=2048),
        #                               nn.ReLU(),
        #                               nn.Dropout(0.5)) # in features was 2048 to 1024, changed to 8192 because of tensor size
        
        # self.fcBlock2 = nn.Sequential(nn.Linear(in_features=2048, out_features=1024),
        #                               nn.ReLU(),
        #                               nn.Dropout(0.5)) # in features was 2048 to 1024, changed to 8192 because of tensor size

        # self.fcBlock3 = nn.Sequential(nn.Linear(in_features=1024, out_features=256),
        #                               nn.ReLU(),
        #                               nn.Dropout(0.5))

        # self.output = nn.Sequential(nn.Linear(in_features=256, out_features=8), 
        #                             nn.Softmax(dim=1)) # out_features changed to 8 for new number of features
        


        # Previous Linear Blocks
        self.fcBlock1 = nn.Sequential(nn.Linear(in_features=2048, out_features=1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))  # in features was 2048 to 1024, changed to 8192 because of tensor size

        self.fcBlock2 = nn.Sequential(nn.Linear(in_features=1024, out_features=256),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.output = nn.Sequential(nn.Linear(in_features=256, out_features=8),
                                    nn.Softmax(dim=1))  # out_features changed to 8 for new number of features

    def forward(self, inp):
        # print(inp.shape)

        out = self.convBlock1(inp)
        out = self.convBlock2(out)
        out = self.convBlock3(out)
        out = self.convBlock4(out)

        out = out.view(out.size()[0], -1)
        out = self.fcBlock1(out)
        out = self.fcBlock2(out)
        # out = self.fcBlock3(out) # for 3 block method
        out = self.output(out)
        # print(out)
        return out


class CrnnModel(nn.Module):
    def __init__(self):
        super(CrnnModel, self).__init__()
        cov1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov1.weight)
        self.convBlock1 = nn.Sequential(cov1,
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2))

        cov2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov2.weight)
        self.convBlock2 = nn.Sequential(cov2,
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2))

        cov3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov3.weight)
        self.convBlock3 = nn.Sequential(cov3,
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=4))

        self.GruLayer = nn.GRU(input_size=2048,
                               hidden_size=256,
                               num_layers=1,
                               batch_first=True,
                               bidirectional=False)

        self.GruLayerF = nn.Sequential(nn.BatchNorm1d(2048),
                                       nn.Dropout(0.6))

        self.fcBlock1 = nn.Sequential(nn.Linear(in_features=2048, out_features=512),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.fcBlock2 = nn.Sequential(nn.Linear(in_features=512, out_features=256),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.output = nn.Sequential(nn.Linear(in_features=256, out_features=10),
                                    nn.Softmax(dim=1))

    def forward(self, inp):
        # _input (batch_size, time, freq)

        out = self.convBlock1(inp)
        out = self.convBlock2(out)
        out = self.convBlock3(out)
        # [16, 256, 8, 8]

        out = out.contiguous().view(out.size()[0], out.size()[2], -1)
        out, _ = self.GruLayer(out)
        out = out.contiguous().view(out.size()[0],  -1)
        # out_features=4096

        out = self.GruLayerF(out)
        out = self.fcBlock1(out)
        out = self.fcBlock2(out)
        out = self.output(out)
        return out


class CrnnLongModel(nn.Module):
    def __init__(self):
        super(CrnnLongModel, self).__init__()
        cov1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov1.weight)
        self.convBlock1 = nn.Sequential(cov1,
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2))

        cov2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov2.weight)
        self.convBlock2 = nn.Sequential(cov2,
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2))

        cov3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov3.weight)
        self.convBlock3 = nn.Sequential(cov3,
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=4))

        cov4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov4.weight)
        self.convBlock4 = nn.Sequential(cov4,
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=4))

        self.GruLayer = nn.GRU(input_size=2048,
                               hidden_size=256,
                               num_layers=1,
                               batch_first=True,
                               bidirectional=False)

        self.GruLayerF = nn.Sequential(nn.BatchNorm1d(1024),
                                       nn.Dropout(0.5))

        self.fcBlock1 = nn.Sequential(nn.Linear(in_features=1024, out_features=512),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.fcBlock2 = nn.Sequential(nn.Linear(in_features=512, out_features=256),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.output = nn.Sequential(nn.Linear(in_features=256, out_features=10),
                                    nn.Softmax(dim=1))

    def forward(self, inp):
        # _input (batch_size, time, freq)

        out = self.convBlock1(inp)
        out = self.convBlock2(out)
        out = self.convBlock3(out)
        out = self.convBlock4(out)
        # [16, 256, 16, 16]

        out = out.contiguous().view(out.size()[0], out.size()[2], -1)
        out, _ = self.GruLayer(out)
        out = out.contiguous().view(out.size()[0], -1)

        out = self.GruLayerF(out)
        out = self.fcBlock1(out)
        out = self.fcBlock2(out)

        out = self.output(out)

        return out


class RnnModel(nn.Module):
    def __init__(self):
        super(RnnModel, self).__init__()

        self.GruLayer = nn.GRU(input_size=256,
                               hidden_size=1024,
                               num_layers=2,
                               batch_first=True,
                               bidirectional=False)

        cov1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov1.weight)
        self.convBlock1 = nn.Sequential(cov1,
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2))

        cov2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov2.weight)
        self.convBlock2 = nn.Sequential(cov2,
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2))

        cov3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov3.weight)
        self.convBlock3 = nn.Sequential(cov3,
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=4))

        cov4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov4.weight)
        self.convBlock4 = nn.Sequential(cov4,
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=4))

        self.fcBlock1 = nn.Sequential(nn.Linear(in_features=11520, out_features=4096),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.fcBlock2 = nn.Sequential(nn.Linear(in_features=4096, out_features=2048),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.fcBlock3 = nn.Sequential(nn.Linear(in_features=2048, out_features=1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.fcBlock4 = nn.Sequential(nn.Linear(in_features=1024, out_features=256),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.output = nn.Sequential(nn.Linear(in_features=256, out_features=10),
                                    nn.Softmax(dim=1))

    def forward(self, inp):
        # _input (batch_size, 1, time, freq)
        inp = inp.contiguous().view(inp.size()[0], inp.size()[2], -1)
        out, _ = self.GruLayer(inp)
        out = out.contiguous().view(out.size()[0], 1, out.size()[1], out.size()[2])
        out = self.convBlock1(out)
        out = self.convBlock2(out)
        out = self.convBlock3(out)
        out = self.convBlock4(out)
        out = out.contiguous().view(out.size()[0], -1)
        out = self.fcBlock1(out)
        out = self.fcBlock2(out)
        out = self.fcBlock3(out)
        out = self.fcBlock4(out)
        out = self.output(out)
        return out


if __name__ == '__main__':
    TestModel = CnnModel()
    from Paras_nb import Para
    Para.batch_size = 32
    from data_loader import torch_dataset_loader
    test_loader = torch_dataset_loader(Para.LA_TEST_DATA_PATH, Para.batch_size, False, Para.kwargs)

    for index, data in enumerate(test_loader):
        spec_input, target = data['mel'], data['tag']

        TestModel.eval()
        predicted = TestModel(spec_input)
        break



