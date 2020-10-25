import torch
from torch import nn


class BidirectionalLSTM(nn.Module):
    """Wraps Pytorch's bidirectional LSTM to allow ease of extracting the final
    hidden layer of the LSTM as an embedding.
    """
    def __init__(
        self,
        input_dim,
        hidden_size,
        output_size,
        dropout=0.5,
        num_layers=5,
    ):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(
            input_dim,
            hidden_size,
            bidirectional=True,
            dropout=dropout,
            num_layers=num_layers,
        )

        self.embedding = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, return_pen_ult=False):
        recurrent, _ = self.rnn(input)
        timesteps, batches, height = recurrent.size()
        t_rec = recurrent.view(timesteps * batches, height)

        output = self.embedding(t_rec) # [timesteps * batches, output_size]
        output = output.view(timesteps, batches, -1)

        if return_pen_ult:
            return output, t_rec
        return output


class CRNN(nn.Module):
    """Implementation of the Convolutional Recurrent Neural Network (CRNN),
    which is sequentially a CNN with a RNN and a CTC loss for image-based
    sequence recognition tasks, e.g. scene text recognition and OCR.

    Notes
    -----
        This is the CRNN as specified in the paper at
            https://arxiv.org/abs/1507.05717
    """
    def __init__(
        self,
        num_channels,
        num_classes,
        hidden_size,
        num_hidden=5,
        cnn_output_size=None,
        input_height=None,
        leakyRelu=True,
        batch_norm=True,
    ):
        super(CRNN, self).__init__()

        # TODO have these be adjustable params or CRNN_Network_Param class
        # Conv Relu Architecture Parameters
        kernel_sizes = [3] * 5
        strides = [1] * 5
        paddings = [1] * 5
        cnn_input_dims = [16, 32, 48, 64, 80]
        # [64, 128, 256, 256, 512, 512, 512]

        # Set the cnn_output_size if not given explicitly
        if cnn_output_size is None and input_height is None:
            raise ValueError(' '.join([
            'Must provide either cnn_output_size or input_height. Both were',
            'unchanged from None.',
            ]))
        elif cnn_output_size is None and input_height is not None:
            # TODO figure out constant multiple value and why.
            #cnn_output_size = int(cnn_input_dims[-1] * input_height / 8)
            cnn_output_size = int(cnn_input_dims[-1] * input_height / 4)

        cnn = nn.Sequential()

        # CNN sequential architecture parameters
        # TODO consider replacing dicts w/ another ** expandable object that
        # can only take the values of some set of arguments. Perhaps a config
        # object for the CRNN.
        maxpool2d_args = [
            {'kernel_size': 2, 'stride': 2},
            {'kernel_size': 2, 'stride': 2},
            {'kernel_size': 2, 'stride': 2},
            None, # *[(2, 2), (2, 1), (0, 1)]
            None, # *[(2, 2), (2, 1), (0, 0)]
        ]
        maxpool_idx = [0, 1, 1, None, None] # to better match original's name scheme...
        dropout_probs = [0, 0, 0.2, 0.2, 0.2]

        # Construct the CRNN given architecture specification
        for i in range(len(maxpool2d_args)):
            input_dim = num_channels if i == 0 else cnn_input_dims[i - 1]
            output_size = cnn_input_dims[i]

            cnn.add_module(
                f'conv{i}',
                nn.Conv2d(
                    input_dim,
                    output_size,
                    kernel_sizes[i],
                    strides[i],
                    paddings[i],
                ),
            )

            if batch_norm:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(output_size))

            if leakyRelu:
                cnn.add_module(f'relu{i}', nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module(f'relu{i}', nn.ReLU(True))

            if isinstance(maxpool2d_args[i], dict):
                cnn.add_module(
                    f'pooling{maxpool_idx[i]}',
                    nn.MaxPool2d(**maxpool2d_args[i]),
                )

            if dropout_probs[i] > 0:
                cnn.add_module('dropout{0}', nn.Dropout2d(p=dropout_probs[i]))

        # Original conv shapes (from pytorch version by meijieru)
        # 64x16x64 -> 128x8x32 -> 256x4x16 -> 512x2x16
        # Current conv shapes post max pool:
        # 64x16x64 -> 128x8x32 -> 128x8x32

        self.cnn = cnn
        self.rnn = BidirectionalLSTM(
            cnn_output_size,
            hidden_size,
            num_classes,
            num_layers=num_hidden,
        )
        # TODO determine if softmaxLayer is to be removed or renamed
        self.softmaxLayer = nn.Linear(num_classes, num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, return_conv=False, return_rnn=False):
        """Forward pass of CRNN.

        Parameters
        ----------
        input :
            Input to the CRNN
        return_conv : bool
            Returns the output of the final convolutional layer of the CRNN if
            True.
        return_rnn : bool
            Returns the output of the final recurrent layer of the CRNN if
            True.
        """
        # conv features
        conv = self.cnn(input)

        # batch, classes, height, width?
        batches, classes, height, width = conv.size()
        #batches, width = conv.size()[[0, 3]]  # TODO consider a slice
        conv = torch.flatten(conv, start_dim=1, end_dim=2)
        conv = conv.view(batches, -1, width)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        if return_rnn:
            rnn, rnn_emb = self.rnn(conv, return_rnn)
        else:
            rnn = self.rnn(conv)

        output = self.softmax(rnn)

        if not (return_conv or return_rnn):
            return output

        return_list = [output]
        if return_conv:
            return_list.append(conv)

        if return_rnn:
            return_list.append(rnn_emb)

        # TODO correct this return
        #return tuple(return_list)
        return return_list
