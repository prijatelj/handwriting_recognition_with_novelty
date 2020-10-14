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

from experiments.research.par_v1 import crnn_data
from experiments.research.par_v1.grieggs import (
    #character_set,
    error_rates,
    hw_dataset,
    string_utils,
)
#from experiments.research.par_v1.grieggs import mdlstm_hwr as model


def train_crnn(
    hw_crnn,
    optimizer,
    criterion,
    char_encoder,
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
            # TODO output the Loss of the network!
            # TODO and change sum_loss name to sum_cer

            # Training Eval loop on training data
            for j in range(out.shape[0]):
                logits = out[j, ...]

                pred, raw_pred = string_utils.naive_decode(logits)

                pred_str = string_utils.label2str(
                    pred,
                    char_encoder.encoder.inverse,
                    False,
                    char_encoder.blank_char,
                    char_encoder.blank_idx,
                )

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

        logging.info("Epoch: %d: Training CER %f", epoch, sum_loss / steps)
        logging.info("Training WER: %f", sum_wer_loss / steps)
        logging.info("Time: %f Seconds.", eTime)

        sum_loss = 0.0
        sum_wer_loss = 0.0
        steps = 0.0
        hw_crnn.eval()

        # Validation loop per epoch
        if test_dataloader is not None:
            logging.info("Validation Set Size = %d", len(test_dataloader))

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

                    pred_str = string_utils.label2str(
                        pred,
                        char_encoder.encoder.inverse,
                        False,
                        char_encoder.blank_char,
                        char_encoder.blank_idx,
                    )

                    cer = error_rates.cer(gt_line, pred_str)
                    wer = error_rates.wer(gt_line, pred_str)
                    sum_wer_loss += wer
                    sum_loss += cer
                    steps += 1

            message = message + "\nTest CER: " + str(sum_loss / steps)
            message = message + "\nTest WER: " + str(sum_wer_loss / steps)
            logging.info("Test CER %f", sum_loss / steps)
            logging.info("Test WER %f", sum_wer_loss / steps)
            best_distance += 1

            # Repeatedly saves the best performing model so-far based on Val.
            if metric == "CER":
                if lowest_loss > sum_loss / steps:
                    # Save the weights for this epoch if the ANN has the lowest
                    # CER yet. NOTE that this is not the Loss of the network,
                    # but the CER.
                    # TODO include saving network w/ best ANN Loss on Val
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
    char_encoder,
    dtype,
    output_crnn_eval=True,
    layer=None,
    return_logits=True,
    return_slice=False,
    deterministic=True,
    random_seed=None,
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
    if deterministic:
        if random_seed is None:
            logging.warning(' '.join([
                'Model is being evaluated and deterministic is true, but no',
                'seed was provided. The default seed of `4816` is being used.',
            ]))
            random_seed = 4816
        logging.info('random seed = %d', random_seed)

        torch.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        logging.warning(' '.join([
            'Model is being evaluated and deterministic is False! The results',
            'may not be reproducible now due to GPU hardware, even if there',
            'is no shuffling or updating of the model.',
        ]))

    # Initialize metrics
    if output_crnn_eval or return_slice:
        tot_ce = 0.0
        tot_we = 0.0
        sum_loss = 0.0
        sum_wer = 0.0
        steps = 0.0
        total_chars = 0.0
        total_words = 0.0

    hw_crnn.eval()

    layer_outs = []

    if return_logits:
        logits_list = []

    if return_slice:
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

                    pred_str = string_utils.label2str(
                        pred,
                        char_encoder.encoder.inverse,
                        False,
                        char_encoder.blank_char,
                        char_encoder.blank_idx,
                    )

                    wer = error_rates.wer(gt_line, pred_str)
                    sum_wer += wer

                    cer = error_rates.cer(gt_line, pred_str)

                    tot_we += wer * len(gt_line.split())
                    tot_ce += cer * len(u' '.join(gt_line.split()))

                    total_words += len(gt_line.split())
                    total_chars += len(u' '.join(gt_line.split()))

                    sum_loss += cer


                    if return_slice and cer <= 0:
                        perfect_indices.append(steps)

                    if return_logits:
                        logits_list.append(logits)

                    steps += 1

    if layer is None or output_crnn_eval:
        logging.info('CRNN results:')
        logging.info("Eval CER %f", sum_loss / steps)
        logging.info("Eval WER %f", sum_wer / steps)

        logging.info("Total character Errors: %d", tot_ce)
        logging.info("Total characters: %d", total_chars)
        logging.info("Total character errors rate: %f", tot_ce / total_chars)

        logging.info("Total word errors %d", tot_we)
        logging.info("Total words: %d", total_words)
        logging.info("Total word error rate: %f", tot_we / total_words)

    # NOTE that the way this is setup, it always expects to return the layers
    if not (return_logits or isinstance(layer, str) or return_slice):
        return None

    logging.debug('perfect_indices len: %d', len(perfect_indices))

    logging.debug('perfect_indices:\n%s', perfect_indices)
    logging.debug(
        'logits_list shapes:\n%s',
        [logit.shape for logit in logits_list],
    )
    logging.debug('layer shapes:\n%s', [layer.shape for layer in layer_outs])

    return_list = []
    if return_logits:
        return_list.append(logits_list)

    if isinstance(layer, str):
        return_list.append(layer_outs)

    if return_slice:
        return_list.append(perfect_indices)

    #return tuple(return_list)
    return return_list


def find_perfect_indices(logits, target_transcript, char_encoder):
    """Given the logits and expected transcript labels, find the pairs with
    perfect prediction from the logits.
    """
    perfect_indices = []
    for i, gt_line in enumerate(target_transcript):
        line_encoded = logits[i, ...]

        pred = string_utils.naive_decode(line_encoded)[0]

        pred_str = string_utils.label2str(
            pred,
            char_encoder.encoder.inverse,
            False,
            char_encoder.blank_char,
            char_encoder.blank_idx,
        )

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
        #layer[perfect_lines].reshape(
        #    len(perfect_lines) * layer.shape[1],
        #    -1,
        #),
        #logits[perfect_lines].reshape(
        #    len(perfect_lines) * logits.shape[1],
        #    -1,
        #),
        np.concatenate(np.array(layer)[perfect_lines]),
        np.concatenate(np.array(logits)[perfect_lines]),
    )

    # TODO expand logits when each has a different length

    #np.concatenate([i.reshape(len(perfect_lines) * logits.shape[1], -1)
    #    for i in np.array(logits)[perfect_lines]
    #])


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

    parser.add_argument(
        '--nondeterministic',
        action='store_false',
        help='Force the model to operate nondeterministically during eval.',
        dest='deterministic'
    )

    parser.add_argument(
        '--random_seed',
        type=int,
        default=None,
        help='Initialize the model with this random seed.',
    )

    parser.add_argument(
        '--save_perfect_slices_only',
        action='store_true',
        help='When evaluating, only the perfect slices of outputs are saved.',
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
    if 'blank' in config['model']['crnn']['train']:
        blank = config['model']['crnn']['train']['blank']
    else:
        logging.warning(' '.join([
            'No `blank` for CTC Loss given in the config file!',
            'Default assumed is 0',
        ]))
        blank = 0

    if 'space_char' in config['model']['crnn']['train']:
        space_char = config['model']['crnn']['train']['space_char']
    else:
        logging.warning(
            'No `space_char` given in the config file! Default assumed is " "',
        )
        space_char = ' '

    if 'unknown_idx' in config['model']['crnn']['train']:
        unknown_idx = config['model']['crnn']['train']['unknown_idx']
    else:
        raise ValueError(
            'No `unknown_idx` given in the config file! No assumed default!',
        )

    char_encoder = crnn_data.load_char_encoder(
        config['data']['iam']['labels'],
        blank,
        space_char,
        unknown_idx,
    )

    logging.info(
        'blank: char = %s; idx = %d;',
        char_encoder.blank_char,
        char_encoder.blank_idx,
    )
    logging.info(
        'space: char = %s; idx = %d;',
        char_encoder.space_char,
        char_encoder.space_idx,
    )
    logging.info(
        'unknown: char = %s; idx = %d;',
        char_encoder.unknown_char,
        char_encoder.unknown_idx,
    )

    if 'augmentation' in config['model']['crnn']['train']:
        train_augmentation = config['model']['crnn']['train']['augmentation']
    else:
        train_augmentation = False

    # Handle image path prefixes in config
    if 'normal_image_prefix' in config['data']['iam']:
        normal_image_prefix = config['data']['iam']['normal_image_prefix']
    else:
        normal_image_prefix = ''

    if 'antique_image_prefix' in config['data']['iam']:
        antique_image_prefix = config['data']['iam']['antique_image_prefix']
    else:
        antique_image_prefix = ''

    if 'noise_image_prefix' in config['data']['iam']:
        noise_image_prefix = config['data']['iam']['noise_image_prefix']
    else:
        noise_image_prefix = ''

    train_dataset = hw_dataset.HwDataset(
        config['data']['iam']['train'],
        char_encoder,
        img_height=config['model']['crnn']['init']['input_height'],
        root_path=config['data']['iam']['image_root_dir'],
        augmentation=train_augmentation,
        normal_image_prefix=normal_image_prefix,
        antique_image_prefix=antique_image_prefix,
        noise_image_prefix=noise_image_prefix,
    )

    try:
        test_dataset = hw_dataset.HwDataset(
            config['data']['iam']['val'],
            char_encoder,
            img_height=config['model']['crnn']['init']['input_height'],
            root_path=config['data']['iam']['image_root_dir'],
            normal_image_prefix=normal_image_prefix,
            antique_image_prefix=antique_image_prefix,
            noise_image_prefix=noise_image_prefix,
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
        batch_size=config['model']['crnn']['train']['batch_size'],
        shuffle=False,
        num_workers=1,
        collate_fn=hw_dataset.collate,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['model']['crnn']['eval']['batch_size'],
        shuffle=False,
        num_workers=1,
        collate_fn=hw_dataset.collate,
    )

    logging.info("Train Dataset Length: " + str(len(train_dataset)))
    logging.info("Test Dataset Length: " + str(len(test_dataset)))

    # Create Model (CRNN)
    hw_crnn, dtype = crnn_data.init_CRNN(config)

    if args.train:
        if 'load_path' in config['model']['crnn']:
            logging.warning(
                '`load_path` in model config, and training the model!',
            )

        if (
            'optimizer' not in config['model']['crnn']['train']
            or config['model']['crnn']['train']['optimizer'].lower()
                == 'adadelta'
        ):
            optimizer = torch.optim.Adadelta(
                hw_crnn.parameters(),
                lr=config['model']['crnn']['train']['learning_rate'],
            )
        elif config['model']['crnn']['train']['optimizer'].lower() == 'adam':
            optimizer = torch.optim.Adam(
                hw_crnn.parameters(),
                lr=config['model']['crnn']['train']['learning_rate'],
            )
        else:
            raise ValueError('optimizer can only be ADADelta or ADAM.')

        criterion = CTCLoss(
            blank=blank,
            reduction='sum',
            zero_infinity=True,
        )

        # Training Loop
        # TODO save CRNN output for ease of eval and comparison
        train_crnn(
            hw_crnn,
            optimizer,
            criterion,
            char_encoder,
            train_dataloader,
            dtype,
            model_save_path,
            test_dataloader,
            base_message=base_message,
        )

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
                char_encoder,
                dtype,
                output_crnn_eval=True,
                layer=args.slice,
                return_logits=True,
                return_slice=args.slice is not None,
                deterministic=args.deterministic,
                random_seed=args.random_seed,
            )

            # Obtain perfect RNN slices
            if args.slice:
                perfect_slices = out[2]

                if args.save_perfect_slices_only:
                    layer, logits = character_slices(
                        out[1],
                        out[0],
                        perfect_slices,
                    )
                else:
                     layer = np.concatenate(np.array(out[1])),
                     logits = np.concatenate(np.array(out[0])),

                # Save slices idx, sliced layer, sliced logits
                with h5py.File(
                    exputils.io.create_filepath(os.path.join(
                        model_save_path,
                        f'layer_logits_slices_{data_split}.hdf5',
                    )),
                    'w',
                ) as h5f:
                    h5f.create_dataset('perfect_indices', data=perfect_slices)
                    h5f.create_dataset('layer', data=layer)
                    h5f.create_dataset('logits', data=logits)


if __name__ == "__main__":
    main()
