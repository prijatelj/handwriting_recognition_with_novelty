import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchnets.unet.unet2d import UNet
from .pixel_rnn import DiagonalPixelLSTM


class MDLSTM_hwr(nn.Module):
    # //        "input_height": 128,
    # //        "input_width": 1792,
    def __init__(self, char_set_len):
        super(MDLSTM_hwr, self).__init__()
        # "conv0": {"class": "conv2", "n_features": 15, "filter": [3, 3], "pool_size": [2, 2], "from": ["1Dto2D"]},
        # "mdlstm0": {"class": "mdlstm", "n_out": 30, "dropout": 0.25, "from": ["conv0"]},


        # "conv1": {"class": "conv2", "n_features": 45, "dropout": 0.25, "filter": [3, 3], "pool_size": [2, 2],
        #           "from": ["mdlstm0"]},
        # "mdlstm1": {"class": "mdlstm", "n_out": 60, "dropout": 0.25, "from": ["conv1"]},


        # "conv2": {"class": "conv2", "n_features": 75, "dropout": 0.25, "filter": [3, 3], "pool_size": [2, 2],
        #           "from": ["mdlstm1"]},
        # "mdlstm2": {"class": "mdlstm", "n_out": 90, "dropout": 0.25, "from": ["conv2"]},


        # "conv3": {"class": "conv2", "n_features": 105, "dropout": 0.25, "filter": [3, 3], "pool_size": [1, 1],
        #           "from": ["mdlstm2"]},
        # "mdlstm3": {"class": "mdlstm", "n_out": 120, "dropout": 0.25, "from": ["conv3"]},


        # "conv4": {"class": "conv2", "n_features": 105, "dropout": 0.25, "filter": [3, 3], "pool_size": [1, 1],
        #           "from": ["mdlstm3"]},
        # "mdlstm4": {"class": "mdlstm", "n_out": 120, "dropout": 0.25, "from": ["conv4"]},


        # "output": {"class": "softmax", "loss": "ctc", "dropout": 0.25, "from": ["mdlstm4"]}\

        # After exensive research this appears to be what our good friends in Aachen are doing when calling conv2.
        self.conv0 = nn.Conv2d(3, 15, (3,3))
        # self.drop0 = nn.Dropout2d(.25)
        # self.convPool0 = nn.MaxPool2d((2,2))
        # self.tanh0 = nn.Hardtanh()
        self.mdlstm0 = DiagonalPixelLSTM(15, 30)

        self.conv1 = nn.Conv2d(30, 45, (3,3))
        # self.drop1 = nn.Dropout2d(.25)
        # self.convPool1 = nn.MaxPool2d((2,2))
        # self.tanh1 = nn.Hardtanh()
        self.mdlstm1 = DiagonalPixelLSTM(45, 60)

        self.conv2 = nn.Conv2d(60, 75, (3,3))
        # self.drop2 = nn.Dropout2d(.25)
        # self.convPool2 = nn.MaxPool2d((2,2))
        # self.tanh2 = nn.Hardtanh()
        self.mdlstm2 = DiagonalPixelLSTM(75, 90)

        self.conv3 = nn.Conv2d(90, 105, (3,3))
        # self.drop3 = nn.Dropout2d(.25)
        # self.convPool3 = nn.MaxPool2d((2,2))
        # self.convPool3 = nn.MaxPool2d((4, 2))
        # self.tanh3 = nn.Hardtanh()
        self.mdlstm3 = DiagonalPixelLSTM(105, 120)

        # self.conv4 = nn.Conv2d(120, 135, (3,3))
        # self.drop4 = nn.Dropout2d(.25)
        # self.convPool4 = nn.MaxPool2d((4,2))
        # self.tanh4 = nn.Hardtanh()
        # self.mdlstm4 = DiagonalPixelLSTM(135, 150)
        # self.fc = nn.Linear(150,char_set_len)
        self.fc = nn.Linear(120, char_set_len)
        # self.output = nn.LogSoftmax(dim=1)

        def init_weights(m):
            net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
            net.apply(init_weights)






                # if bridge_width is None:
        #     bridge_width = imgH
        # super(URNN, self).__init__()
        # # assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        # cnn = nn.Sequential()
        # if pretrainedUnet is not None:
        #     self.UNet = pretrainedUnet
        # else:
        #     self.UNet = UNet(in_channels=nc, min_filters=min_filters, n_classes=nclass)
        # self.downConv = nn.Conv2d(nclass, unet_outs, (imgH, bridge_width), stride=1, padding=0)
        # self.rnn = nn.Sequential(
        #     BidirectionalLSTM(unet_outs, nh, nh),
        #     BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        # print input.shape
        x = self.conv0(input)
        # x = self.drop0(x)
        x = torch.tanh(x)
        x = F.max_pool2d(x,(2,2))
        x = self.mdlstm0(x)

        x = self.conv1(x)
        x = torch.tanh(x)
        x = F.max_pool2d(x, (2, 2))
        x = F.dropout2d(x, p=0.25)
        x = self.mdlstm1(x)
        x = F.dropout2d(x, p=0.25)

        x = self.conv2(x)
        x = torch.tanh(x)
        x = F.max_pool2d(x, (2, 2))
        x = F.dropout2d(x, p=0.25)
        x = self.mdlstm2(x)
        x = F.dropout2d(x, p=0.25)

        x = self.conv3(x)
        x = torch.tanh(x)
        x = F.max_pool2d(x, (4, 2))
        x = F.dropout2d(x,p=0.25)
        x = self.mdlstm3(x)
        x = F.dropout2d(x,p=0.25)

        # x = self.conv4(x)
        # x = self.drop4(x)
        # x = self.tanh0(x)
        # x = self.convPool4(x)
        # x = self.mdlstm4(x)

        # print(x.shape)

        x = torch.squeeze(x,2)
        x = x.permute(2, 0, 1)
        x = self.fc(x)
        x = F.log_softmax(x,dim=1)

        return x


def init_weights(m):
    if type(m) != DiagonalPixelLSTM and type(m) != MDLSTM_hwr:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def create_model(char_set_len):
    urnn = MDLSTM_hwr(char_set_len+1)
    urnn.apply(init_weights)
    return urnn

