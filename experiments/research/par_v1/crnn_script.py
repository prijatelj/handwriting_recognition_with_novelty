"""Script for Training and Evaluating the CRNN by itself."""
import logging
import os
import time

import h5py
import numpy as np
from ruamel.yaml import YAML
import torch
from torch.autograd import Variable
from torch.nn.modules.loss import CTCLoss
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import exputils.io

from hwr_novelty.models.crnn import CRNN

from experiments.research.par_v1.grieggs import (
    character_set,
    error_rates,
    hw_dataset,
    string_utils,
)
#from experiments.research.par_v1.grieggs import mdlstm_hwr as model


def train_crnn(
    hw_crnn,
    optimizer,
    criterion,
    idx_to_char,
    train_dataloader,
    dtype,
    model_save_path=None,
    test_dataloader=None,
    epochs=1000,
    metric='CER',
    base_message='',
):
    """Streamline the training of the CRNN."""
    # Variables for training loop
    lowest_loss = float('inf')
    best_distance = 0

    # Training Epoch Loop
    for epoch in range(epochs):
        torch.enable_grad()

        startTime = time.time()
        message = base_message

        sum_loss = 0.0
        sum_wer_loss = 0.0
        steps = 0.0

        hw_crnn.train()

        disp_ctc_loss = 0.0
        disp_loss = 0.0
        gt = ""
        ot = ""
        loss = 0.0

        logging.info("Train Set Size = %d", len(train_dataloader))

        # Training Batch Loop
        prog_bar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
        )
        for i, x in prog_bar:
            prog_bar.set_description(' '.join([
                f'CER: {disp_loss} CTC: {loss} Ground Truth: |{gt}| Network',
                f'Output: |{ot}|',
            ]))

            line_imgs = x['line_imgs']
            rem = line_imgs.shape[3] % 32
            if rem != 0:
                imgshape = line_imgs.shape
                temp = torch.zeros(
                    imgshape[0],
                    imgshape[1],
                    imgshape[2],
                    imgshape[3] + (32 - rem),
                )
                temp[:, :, :, :imgshape[3]] = line_imgs
                line_imgs = temp
                del temp
            line_imgs = Variable(line_imgs.type(dtype), requires_grad=False)

            labels = Variable(x['labels'], requires_grad=False)
            label_lengths = Variable(x['label_lengths'], requires_grad=False)

            preds = hw_crnn(line_imgs).cpu()
            preds_size = Variable(torch.IntTensor([preds.size(0)] * preds.size(1)))

            output_batch = preds.permute(1, 0, 2)
            out = output_batch.data.cpu().numpy()
            loss = criterion(preds, labels, preds_size, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Training Eval loop on training data
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

        message = (
            message + "\n" + "Epoch: " + str(epoch) + " Training CER: "
            + str(sum_loss / steps)+ " Training WER: " + str(sum_wer_loss /
            steps) + "\n"+"Time: " + str(eTime) + " Seconds"
        )

        logging.info("Epoch: " + str(epoch) + " Training CER", sum_loss / steps)
        logging.info("Training WER: " + str(sum_wer_loss / steps))
        logging.info("Time: " + str(eTime) + " Seconds")

        sum_loss = 0.0
        sum_wer_loss = 0.0
        steps = 0.0
        hw_crnn.eval()

        # Validation loop per epoch
        if test_dataloader is not None:
            logging.info("Validation Set Size = " + str(len(test_dataloader)))

            for x in tqdm(test_dataloader):
                torch.no_grad()
                line_imgs = Variable(
                    x['line_imgs'].type(dtype),
                    requires_grad=False,
                )

                preds = hw_crnn(line_imgs).cpu()
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
            logging.info("Test CER %d", sum_loss / steps)
            logging.info("Test WER %d", sum_wer_loss / steps)
            best_distance += 1

            # Repeatedly saves the best performing model so-far based on Val.
            if metric == "CER":
                if lowest_loss > sum_loss / steps:
                    lowest_loss = sum_loss / steps
                    logging.info("Saving Best")
                    message = message + "\nBest Result :)"
                    torch.save(
                        hw_crnn.state_dict(),
                        os.path.join(
                            model_save_path,
                            f'crnn_ep{str(epoch)}.pt',
                        ),
                    )
                    best_distance = 0
                if best_distance > 800:
                    break
            elif metric == "WER":
                if lowest_loss > sum_wer_loss / steps:
                    lowest_loss = sum_wer_loss / steps
                    logging.info("Saving Best")
                    message = message + "\nBest Result :)"
                    torch.save(
                        hw_crnn.state_dict(),
                        os.path.join(
                            model_save_path,
                            f'crnn_ep{str(epoch)}.pt',
                        ),
                    )
                    best_distance = 0
                if best_distance > 80:
                    break
            else:
                raise ValueError("This is actually very bad")
    return


def eval_crnn(
    hw_crnn,
    dataloader,
    idx_to_char,
    dtype,
    output_crnn_eval=True,
    layer=None,
    return_logits=True,
    return_slice=False,
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
    return_logits : bool, optional
    return_slice : bool, optional
        Returns the indices of the perfect slices

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

    if return_slice:
        count = 0
        perfect_indices = []

    # For batch in dataloader
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
            logging.info(f'line_imgs.shape = {line_imgs.shape}')
            logging.info(f'line_imgs = {line_imgs}')
            logging.info(f'Preds shape: {preds.shape}')
            logging.info(f'RNN Out: {layer_out}')
            logging.info(f'RNN Out: {layer_out.shape}')
            logging.info(f'x["gt"] = {x["gt"]}')
            logging.info(f'x["gt"] len = {len(x["gt"][0])}')
            """
            # Swap 0 and 1 indices to have:
            #   batch sample, "character window", classes
            # Except, since batch sample is always 1 here, that dim is removed:
            #   "character windows", classes

            if layer is None or output_crnn_eval:
                output_batch = preds.permute(1, 0, 2)
                out = output_batch.data.cpu().numpy()

                # Consider MEVM input here after enough obtained to do batch
                # training Or save the layer_outs to be used in training the
                # MEVM

                # Loop through the batch
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

                    if return_slice and cer <= 0:
                        perfect_indices.append(count)

                if return_logits:
                    logits_list.append(out)

        if return_slice:
            # NOTE should this only increment when x is not None?
            count += 1


    if layer is None or output_crnn_eval:
        message = ''
        message = message + "\nTest CER: " + str(sum_loss / steps)
        message = message + "\nTest WER: " + str(sum_wer / steps)

        logging.info('CRNN results:')
        logging.info("Validation CER %d", sum_loss / steps)
        logging.info("Validation WER %d", sum_wer / steps)

        logging.info("Total character Errors: %d", tot_ce)
        logging.info("Total word errors %d", tot_we)

        tot_ce = 0.0
        tot_we = 0.0
        sum_loss = 0.0
        sum_wer = 0.0
        steps = 0.0

    # NOTE that the way this is setup, it always expects to return the layers
    if not (return_logits or isinstance(layer, str) or return_slice):
        return None

    return_list = []
    if return_logits:
        return_list.append(logits_list)

    if isinstance(layer, str):
        return_list.append(layer_outs)

    if return_slice:
        return_list.append(perfect_indices)

    #return tuple(return_list)
    return return_list


def find_perfect_indices(logits, target_transcript, idx_to_char):
    """Given the logits and expected transcript labels, find the pairs with
    perfect prediction from the logits.
    """
    perfect_indices = []
    for i, gt_line in enumerate(target_transcript):
        line_encoded = logits[i, ...]

        pred = string_utils.naive_decode(line_encoded)[0]
        pred_str = string_utils.label2str(pred, idx_to_char, False)

        if error_rates.cer(gt_line, pred_str) <= 0:
            perfect_indices.append(i)

    # TODO find the indicies within the logits that have perf predictions
    #   currently, only find indices of logits w/ correct line predictions,
    #   but must confirm that means that every character pred is correct within
    #   the logits. Note naive decode does not add char from rawPredData to
    #   pred if that char is the same as the prior, this means that if the cer
    #   is zero, then every character within the logits is correct and so all
    #   char indices of the logits, and their respective layer may be used to
    #   train the MEVM.

    return perfect_indices


def character_slices(
    layer,
    logits,
    perfect_lines,
    add_idx=False,
    mask_out=False,
):
    """Given the logits and their corresponding text, match the corresponding
    slices of the given representative feature layer of the transcript to the
    corresponding characters.

    Parameters
    ----------
    layer : np.ndarray(samples, timesteps, height)
    logits : np.ndarray(samples, timesteps, classes)
    perfect_lines:
        perfect indices
    append_idx : bool
        Appends the index of the character in the layer to the beginning of the
        prediction output representation.
    mask_out : bool
        Masks out the layer encoding of the logits with zeros where the perfect
        character does not occur and flattens the layer to represent the
        prediction.

    Returns
    -------
    np.ndarray, np.ndarray
        The layer encoding of the character as a np.ndarray of shape (samples,
        encoding_dim), paired with the character label.
    """
    if add_idx:
        return np.concatenate((
            np.array(perfect_lines).reshape(-1, 1),
            layer[perfect_lines, ...],
        ))

    if mask_out:
        # This expands the resulting output greatly
        raise NotImplementedError('mask_out is not implemented yet.')

    # NOTE if layer_type == 'rnn':
    return (
        layer[perfect_lines].reshape(
            len(perfect_lines) * layer.shape[1],
            -1,
        ),
        logits[perfect_lines].reshape(
            len(perfect_lines) * logits.shape[1],
            -1,
        ),
    )


def io_args(parser):
    parser.add_argument(
        'config_path',
        help='YAML experiment configuration file defining the ANN and data.',
    )

    parser.add_argument(
        '--train',
        action='store_true',
        help='Expect to train model if given.',
    )

    parser.add_argument(
        '--eval',
        default=None,
        nargs='+',
        help='The data splits to be evaluated.',
        choices=['train', 'val', 'test'],
    )

    parser.add_argument(
        '--slice',
        default=None,
        #nargs='+',
        choices=['rnn', 'cnn', 'conv'],
        help='Slice the given layers of the CRNN.',
    )


def main():
    # Handle argrument parsing
    args = exputils.io.parse_args(custom_args=io_args)

    with open(args.config_path) as openf:
        #config = json.load(openf)
        config = YAML(typ='safe').load(openf)

    model_save_path = exputils.io.create_dirs(config['model']['crnn']['save_path'])

    with open(args.config_path) as f:
        paramList = f.readlines()

    for x in paramList:
        logging.info(x[:-1])

    base_message = ""
    for line in paramList:
        base_message = base_message + line

    # Load and Prepare the data and labels
    idx_to_char, char_to_idx = character_set.load_label_set(
        config['data']['iam']['labels'],
    )

    train_dataset = hw_dataset.HwDataset(
        config['data']['iam']['train'],
        char_to_idx,
        img_height=config['model']['crnn']['init']['input_height'],
        root_path=config['data']['iam']['image_root_dir'],
        augmentation=config['model']['crnn']['augmentation'],
    )

    try:
        test_dataset = hw_dataset.HwDataset(
            config['data']['iam']['val'],
            char_to_idx,
            img_height=config['model']['crnn']['init']['input_height'],
            root_path=config['data']['iam']['image_root_dir'],
        )
    except KeyError as e:
        logging.info("No validation set found, generating one")

        master = train_dataset

        logging.info("Total of " +str(len(master)) +" Training Examples")

        n = len(master)  # how many total elements you have
        n_test = int(n * .1)
        n_train = n - n_test
        idx = list(range(n))  # indices to all elements

        train_idx = idx[:n_train]
        test_idx = idx[n_train:]

        test_dataset = Subset(master, test_idx)
        train_dataset = Subset(master, train_idx)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['model']['crnn']['batch_size'],
        shuffle=False,
        num_workers=1,
        collate_fn=hw_dataset.collate,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['model']['crnn']['batch_size'],
        shuffle=False,
        num_workers=1,
        collate_fn=hw_dataset.collate,
    )

    logging.info("Train Dataset Length: " + str(len(train_dataset)))
    logging.info("Test Dataset Length: " + str(len(test_dataset)))

    # Create Model (CRNN)
    hw_crnn = CRNN(**config['model']['crnn']['init'])

    if torch.cuda.is_available():
        hw_crnn.cuda()
        dtype = torch.cuda.FloatTensor
        logging.info("Using GPU")
    else:
        dtype = torch.FloatTensor
        logging.info("No GPU detected")

    if args.train:
        optimizer = torch.optim.Adadelta(
            hw_crnn.parameters(),
            lr=config['model']['crnn']['train']['learning_rate'],
        )
        criterion = CTCLoss(reduction='sum', zero_infinity=True)

        # Training Loop
        # TODO save CRNN output for ease of eval and comparison
        train_crnn(
            hw_crnn,
            optimizer,
            criterion,
            idx_to_char,
            train_dataloader,
            dtype,
            model_save_path,
            test_dataloader,
            base_message=base_message,
        )
    else:
        # If not train, then load model
        hw_crnn.load_state_dict(torch.load(config['model']['crnn']['load_path']))

    if args.eval is not None:
        for data_split in args.eval:
            if data_split == 'train':
                dataloader = train_dataloader
            # TODO elif data_split == 'train':
            #    dataloader = val_dataloader
            else:
                dataloader = test_dataloader

            logging.info('Evaluating CRNN on %s', data_split)
            out = eval_crnn(
                hw_crnn,
                dataloader,
                idx_to_char,
                dtype,
                output_crnn_eval=True,
                layer=args.slice,
                return_logits=True,
                return_slice=args.slice is not None,
            )

            # TODO save logits?

            # Obtain perfect RNN slices
            if args.slice:
                perf_sliced_layer, perf_sliced_logits = character_slices(
                    out[1],
                    out[0],
                    out[2],
                    #add_idx=False,
                    #mask_out=False,
                )

                # Save slices idx, sliced layer, sliced logits
                with h5py.File(
                    exputils.io.create_filepath(os.path.join(
                        model_save_path,
                        f'perf_slices_{data_split}.hdf5',
                    )),
                    'w',
                ) as h5f:
                    h5f.create_dataset('indices', data=out[2])
                    h5f.create_dataset('layer', data=perf_sliced_layer)
                    h5f.create_dataset('logits', data=perf_sliced_logits)


if __name__ == "__main__":
    main()
