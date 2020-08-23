import torch
from torch import nn

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, dropout=0.5, num_layers=5)
        self.embedding = nn.Linear(nHidden * 2, nOut)


    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec) # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class CRNN(nn.Module):

    def __init__(self, cnnOutSize, nc, nclass, nh, n_rnn=2, leakyRelu=True):
        super(CRNN, self).__init__()

        ks = [3, 3, 3, 3, 3, 3]
        ps = [1, 1, 1, 1, 1, 1]
        ss = [1, 1, 1, 1, 1, 1]
        # nm = [64, 128, 256, 256, 512, 512, 512]
        nm = [16, 32, 48, 64, 80]
        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0, True)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1, True)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        cnn.add_module('dropout{0}', nn.Dropout2d(p=0.2))
        convRelu(3, True)
        cnn.add_module('dropout{0}', nn.Dropout2d(p=0.2))
        # cnn.add_module('pooling{0}'.format(2),
        #                nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        cnn.add_module('dropout{0}', nn.Dropout2d(p=0.2))
        # cnn.add_module('pooling{0}'.format(3),
        #                nn.MaxPool2d((2, 2), (2, 1), (0, 0)))  # 512x2x16
        # convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = BidirectionalLSTM(cnnOutSize, nh, nclass)
        self.softmaxLayer = nn.Linear(nclass,nclass)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        # batch, classes, height, width?
        b, c, h, w = conv.size()
        conv = torch.flatten(conv, start_dim=1, end_dim=2)
        conv = conv.view(b, -1, w)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        rnn = self.rnn(conv)
        output = self.softmax(rnn)

        return output, rnn

def create_model(config):
    # nh = ((config['num_of_outputs']/32)+1)*512*(config['input_height']/4)
    nh = 80 * (config['input_height']/4)
    crnn = CRNN(int(nh), config['num_of_channels'], config['num_of_outputs'], 256)
    return crnn
