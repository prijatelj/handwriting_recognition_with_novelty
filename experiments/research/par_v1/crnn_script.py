import json
import character_set
import sys
import hw_dataset
from hw_dataset import HwDataset
import model.mdlstm_hwr as model
import os
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.modules.loss import CTCLoss
import error_rates
import string_utils
import time
from tqdm import tqdm


def train_crnn():
    """Streamline the training of the CRNN."""
    # TODO

    return


def eval_crnn(
    hw_crnn,
    dataloader,
    idx_to_char,
    dtype,
    output_crnn_eval=True,
    layer=None,
    return_logits=True,
):
    """Evaluates CRNN and returns the CRNN output. Optionally, this is also
    used to obtain certain layer's outputs such as the penultimate RNN or CNN
    layers of the CRNN.

    Parameters
    ----------
    hw_crnn :
        The
    dataloader :
        Pytorch dataloader for input to CRNN
    dtype : type
        Pytorch type of a the handwritten line image data. Used to handle CPU
        or GPU use.
    output_crnn_eval : bool, optional
        Outputs the CRNNs performance without the EVM.
    layer : str, optional
        If 'rnn', uses the CRNN's final RNN output as input to the MultipleEVM.
        If 'conv', uses the final convolutional layer's output. If 'concat',
        then returns both concatenated together (Concat is to be implemented).
        Defaults to None, and thus only evaluates the CRNN itself.

    Returns
    -------
    list(np.ndarray)
        Returns a list of the selected layer's output for each input sample.
        `layer` determines which layer of the CRNN is used. The shape of each
        np.ndarray is [glyph_window, classes]. This assumes batch size is
        always 1.
    """
    # Initialize metrics
    if output_crnn_eval:
        tot_ce = 0.0
        tot_we = 0.0
        sum_loss = 0.0
        sum_wer = 0.0
        steps = 0.0

    hw_crnn.eval()

    layer_outs = []

    if return_logits:
        logits_list = []

    for x in dataloader:
        if x is None:
            continue
        with torch.no_grad():
            line_imgs = Variable(
                x['line_imgs'].type(dtype),
                requires_grad=False,
            )

            if layer.lower() == 'rnn':
                preds, layer_out = hw_crnn(line_imgs, return_rnn=True)

                # Shape is then [timesteps, hidden layer width]
                layer_outs.append(layer_out.data.cpu().numpy())

            elif layer.lower() in {'conv', 'cnn'}:
                # Last Convolution Layer
                preds, layer_out = hw_crnn(line_imgs, return_conv=True)

                # Shape is then [timesteps, conv layer flat: height * width]
                layer_outs.append(layer_out.data.cpu().numpy())
                layer_outs.append(np.squeeze(
                    layer_out.permute(1, 0, 2).data.cpu().numpy()
                ))
            else:
                raise NotImplementedError('Concat/both RNN and Conv of CRNN.')

            """
            print(f'line_imgs.shape = {line_imgs.shape}')
            print(f'line_imgs = {line_imgs}')
            print(f'Preds shape: {preds.shape}')
            print(f'RNN Out: {layer_out}')
            print(f'RNN Out: {layer_out.shape}')
            print(f'x["gt"] = {x["gt"]}')
            print(f'x["gt"] len = {len(x["gt"][0])}')
            """
            # Swap 0 and 1 indices to have:
            #   batch sample, "character window", classes
            # Except, since batch sample is always 1 here, that dim is removed:
            #   "character windows", classes

            if layer is None or output_crnn_eval:
                # TODO save CRNN output for ease of eval and comparison
                output_batch = preds.permute(1, 0, 2)
                out = output_batch.data.cpu().numpy()

                # Consider MEVM input here after enough obtained to do batch
                # training Or save the layer_outs to be used in training the
                # MEVM

                for i, gt_line in enumerate(x['gt']):
                    logits = out[i, ...]

                    pred, raw_pred = string_utils.naive_decode(logits)
                    pred_str = string_utils.label2str(pred, idx_to_char, False)

                    wer = error_rates.wer(gt_line, pred_str)
                    sum_wer += wer

                    cer = error_rates.cer(gt_line, pred_str)

                    tot_we += wer * len(gt_line.split())
                    tot_ce += cer * len(u' '.join(gt_line.split()))

                    sum_loss += cer

                    steps += 1

                if return_logits:
                    logits_list.append(out)


    if layer is None or output_crnn_eval:
        message = ''
        message = message + "\nTest CER: " + str(sum_loss / steps)
        message = message + "\nTest WER: " + str(sum_wer / steps)

        print('CRNN results:')
        print("Validation CER", sum_loss / steps)
        print("Validation WER", sum_wer / steps)

        print("Total character Errors:", tot_ce)
        print("Total word errors", tot_we)

        tot_ce = 0.0
        tot_we = 0.0
        sum_loss = 0.0
        sum_wer = 0.0
        steps = 0.0

    if return_logits and layer is not None:
        return logits_list, layer_outs
    if return_logits and layer is None:
        return logits_list
    if not return_logits and layer is not None:
        return layer_outs


def main():
    config_path = sys.argv[1]
    try:
        jobID = sys.argv[2]
    except:
        jobID = ""
    print(jobID)


    with open(config_path) as f:
        config = json.load(f)

    try:
        model_save_path = sys.argv[3]
        if model_save_path[-1] != os.path.sep:
            model_save_path = model_save_path + os.path.sep
    except:
        model_save_path = config['model_save_path']
    dirname = os.path.dirname(model_save_path)
    print(dirname)
    if len(dirname) > 0 and not os.path.exists(dirname):
        os.makedirs(dirname)


    with open(config_path) as f:
        paramList = f.readlines()

    for x in paramList:
        print(x[:-1])

    baseMessage = ""

    for line in paramList:
        baseMessage = baseMessage + line


    # print(baseMessage)

    idx_to_char, char_to_idx = character_set.load_char_set(config['character_set_path'])

    train_dataset = HwDataset(
        config['training_set_path'],
        char_to_idx,
        img_height=config['network']['input_height'],
        root_path=config['image_root_directory'],
        augmentation=config['augmentation'],
    )

    try:
        test_dataset = HwDataset(config['validation_set_path'], char_to_idx, img_height=config['network']['input_height'],  root_path=config['image_root_directory'])
    except KeyError as e:
        print("No validation set found, generating one")
        master = train_dataset
        print("Total of " +str(len(master)) +" Training Examples")
        n = len(master)  # how many total elements you have
        n_test = int(n * .1)
        n_train = n - n_test
        idx = list(range(n))  # indices to all elements
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]
        test_dataset = data.Subset(master, test_idx)
        train_dataset = data.Subset(master, train_idx)

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=1,
                                  collate_fn=hw_dataset.collate)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=1,
                                 collate_fn=hw_dataset.collate)
    print("Train Dataset Length: " + str(len(train_dataset)))
    print("Test Dataset Length: " + str(len(test_dataset)))


    hw = model.create_model(len(idx_to_char))
    # hw = model.create_model({
    #     'input_height': config['network']['input_height'],
    #     'cnn_out_size': config['network']['cnn_out_size'],
    #     'num_of_channels': 3,
    #     'num_of_outputs': len(idx_to_char) + 1,
    #     'bridge_width': config['network']['bridge_width']
    # })

    if torch.cuda.is_available():
        hw.cuda()
        dtype = torch.cuda.FloatTensor
        print("Using GPU")
    else:
        dtype = torch.FloatTensor
        print("No GPU detected")

    optimizer = torch.optim.Adadelta(hw.parameters(), lr=config['network']['learning_rate'])
    criterion = CTCLoss(reduction='sum',zero_infinity=True)
    lowest_loss = float('inf')
    best_distance = 0
    for epoch in range(1000):
        torch.enable_grad()
        startTime = time.time()
        message = baseMessage
        sum_loss = 0.0
        sum_wer_loss = 0.0
        steps = 0.0
        hw.train()
        disp_ctc_loss = 0.0
        disp_loss = 0.0
        gt = ""
        ot = ""
        loss = 0.0
        print("Train Set Size = " + str(len(train_dataloader)))
        prog_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, x in prog_bar:
            # message = str("CER: " + str(disp_loss) +"\nGT: " +gt +"\nex: "+out+"\nProgress")
            prog_bar.set_description(f'CER: {disp_loss} CTC: {loss} Ground Truth: |{gt}| Network Output: |{ot}|')
            line_imgs = x['line_imgs']
            rem = line_imgs.shape[3] % 32
            if rem != 0:
                imgshape = line_imgs.shape
                temp = torch.zeros(imgshape[0], imgshape[1], imgshape[2], imgshape[3] + (32 - rem))
                temp[:, :, :, :imgshape[3]] = line_imgs
                line_imgs = temp
                del temp
            line_imgs = Variable(line_imgs.type(dtype), requires_grad = False)

            labels = Variable(x['labels'], requires_grad=False)
            label_lengths = Variable(x['label_lengths'], requires_grad=False)

            preds = hw(line_imgs).cpu()
            preds_size = Variable(torch.IntTensor([preds.size(0)] * preds.size(1)))

            output_batch = preds.permute(1,0,2)
            out = output_batch.data.cpu().numpy()
            loss = criterion(preds, labels, preds_size, label_lengths)
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if i == 0:
            #    for i in xrange(out.shape[0]):
            #        pred, pred_raw = string_utils.naive_decode(out[i,...])
            #        pred_str = string_utils.label2str(pred_raw, idx_to_char, True)
            #        print(pred_str)

            for j in range(out.shape[0]):
                logits = out[j, ...]
                pred, raw_pred = string_utils.naive_decode(logits)
                pred_str = string_utils.label2str(pred, idx_to_char, False)
                gt_str = x['gt'][j]
                cer = error_rates.cer(gt_str, pred_str)
                wer = error_rates.wer(gt_str, pred_str)
                gt = gt_str
                ot = pred_str
                sum_loss += cer
                sum_wer_loss += wer
                steps += 1
            disp_loss = sum_loss/steps
        eTime = time.time()- startTime
        message = message + "\n" + "Epoch: " + str(epoch) + " Training CER: " + str(sum_loss / steps)+ " Training WER: " + str(sum_wer_loss / steps) + "\n"+"Time: " + str(eTime) + " Seconds"
        print("Epoch: " + str(epoch) + " Training CER", sum_loss / steps)
        print("Training WER: " + str(sum_wer_loss / steps))
        print("Time: " + str(eTime) + " Seconds")
        sum_loss = 0.0
        sum_wer_loss = 0.0
        steps = 0.0
        hw.eval()
        print("Validation Set Size = " + str(len(test_dataloader)))
        for x in tqdm(test_dataloader):
            torch.no_grad()
            line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False)
            # labels =  Variable(x['labels'], requires_grad=False, volatile=True)
            # label_lengths = Variable(x['label_lengths'], requires_grad=False, volatile=True)
            preds = hw(line_imgs).cpu()
            output_batch = preds.permute(1, 0, 2)
            out = output_batch.data.cpu().numpy()
            for i, gt_line in enumerate(x['gt']):
                logits = out[i, ...]
                pred, raw_pred = string_utils.naive_decode(logits)
                pred_str = string_utils.label2str(pred, idx_to_char, False)
                cer = error_rates.cer(gt_line, pred_str)
                wer = error_rates.wer(gt_line, pred_str)
                sum_wer_loss += wer
                sum_loss += cer
                steps += 1

        message = message + "\nTest CER: " + str(sum_loss / steps)
        message = message + "\nTest WER: " + str(sum_wer_loss / steps)
        print("Test CER", sum_loss / steps)
        print("Test WER", sum_wer_loss / steps)
        best_distance += 1
        metric="CER"
        if(metric == "CER"):
            if lowest_loss > sum_loss / steps:
                lowest_loss = sum_loss / steps
                print("Saving Best")
                message = message + "\nBest Result :)"
                torch.save(hw.state_dict(), os.path.join(model_save_path+str(epoch) + ".pt"))
                #email_update(message, jobID)
                best_distance = 0
            if best_distance > 800:
                break
        elif(metric == "WER"):
            if lowest_loss > sum_wer_loss / steps:
                lowest_loss = sum_wer_loss / steps
                print("Saving Best")
                message = message + "\nBest Result :)"
                torch.save(hw.state_dict(), os.path.join(model_save_path+str(epoch) + ".pt"))
                #email_update(message, jobID)
                best_distance = 0
            if best_distance > 80:
                break
        else:
            print("This is actually very bad")

if __name__ == "__main__":
    main()
