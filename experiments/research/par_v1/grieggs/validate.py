from __future__ import print_function


import json
import character_set
import sys
import hw_dataset
from hw_dataset import HwDataset
#import model.urnn as urnn
import model.crnn as crnn
import os
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import error_rates
import string_utils
import time
import numpy as np
from torch.utils import data
import sys
sys.path.insert(0, 'CTCDecoder/src')
import ctcdecode
# import approx_string_search as aps
# import generate_dict
import grid_distortion


def softmax(mat):
    "calc softmax such that labels per time-step form probability distribution"
    maxT, _ = mat.shape # dim0=t, dim1=c
    res = np.zeros(mat.shape)
    for t in range(maxT):
        y = mat[t, :]
        e = np.exp(y)
        s = np.sum(e)
        res[t, :] = e/s
    return res

def email_update(message):
    import smtplib
    gmail_user = "grieggsdiagnostic@gmail.com"
    gmail_password = "Y1<iquhH9<6k"

    sent_from = gmail_user
    to = ["smgrieggs@gmail.com"]
    subject = 'Job update!(:'

    email_text = 'Subject: {}\n\n{}'.format(subject, message)

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(gmail_user, gmail_password)
        server.sendmail(sent_from, to, email_text)
        server.close()
        print('Results emailed to ' + to[0])
        return True
    except Exception as e:
        print(e)
        print('Email failure, but this isn\'t important enough to stop everything')
        return False


def main():
    torch.manual_seed(68)
    torch.backends.cudnn.deterministic = True

    print(torch.LongTensor(10).random_(0, 10))

    config_path = sys.argv[1]
    RIMES = (config_path.lower().find('rimes') != -1)
    print(RIMES)
    with open(config_path) as f:
        config = json.load(f)

    with open(config_path) as f:
        paramList = f.readlines()

    baseMessage = ""

    for line in paramList:
        baseMessage = baseMessage + line


    # print(baseMessage)
    # lexicon = aps.ApproxLookupTable(generate_dict.get_lexicon())


    idx_to_char, char_to_idx = character_set.load_char_set(config['character_set_path'])
    # val_dataset = HwDataset(config['validation_set_path'], char_to_idx, img_height=config['network']['input_height'], root_path=config['image_root_directory'])
    # val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0, collate_fn=hw_dataset.collate)

    train_dataset = HwDataset(config['training_set_path'], char_to_idx, img_height=config['network']['input_height'],
                              root_path=config['image_root_directory'], augmentation=False)
    try:
        val_dataset = HwDataset(config['validation_set_path'], char_to_idx, img_height=config['network']['input_height'], root_path=config['image_root_directory'], remove_errors=True)

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
        val_dataset = data.Subset(master, test_idx)
        train_dataset = data.Subset(master, train_idx)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=1,
                                 collate_fn=hw_dataset.collate)




    if(not RIMES):
        val2_dataset = HwDataset(config['validation2_set_path'], char_to_idx, img_height=config['network']['input_height'],
                                root_path=config['image_root_directory'], remove_errors=True)
        val2_dataloader = DataLoader(val2_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0,
                                    collate_fn=hw_dataset.collate)

    test_dataset = HwDataset(config['test_set_path'], char_to_idx, img_height=config['network']['input_height'], root_path=config['image_root_directory'], remove_errors=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0, collate_fn=hw_dataset.collate)
    if config['model'] == "crnn":
        print("Using CRNN")
        hw = crnn.create_model({
            'input_height': config['network']['input_height'],
            'cnn_out_size': config['network']['cnn_out_size'],
            'num_of_channels': 3,
            'num_of_outputs': len(idx_to_char) + 1
        })
    #elif config['model'] == "urnn":
    #    print("Using URNN")
    #    hw = urnn.create_model({
    #        'input_height': config['network']['input_height'],
    #        'cnn_out_size': config['network']['cnn_out_size'],
    #        'num_of_channels': 3,
    #        'num_of_outputs': len(idx_to_char)+1,
    #        'bridge_width': config['network']['bridge_width']
    #    })
    # elif config['model'] == "urnn2":
    #     print("Using URNN with Curtis's recurrence")
    #     hw = urnn2.create_model({
    #         'input_height': config['network']['input_height'],
    #         'cnn_out_size': config['network']['cnn_out_size'],
    #         'num_of_channels': 3,
    #         'num_of_outputs': len(idx_to_char) + 1,
    #         'bridge_width': config['network']['bridge_width']
    #     })
    # elif config['model'] == "crnn2":
    #     print("Using original CRNN")
    #     hw = crnn2.create_model({
    #         'cnn_out_size': config['network']['cnn_out_size'],
    #         'num_of_channels': 3,
    #         'num_of_outputs': len(idx_to_char) + 1
    #     })
    # elif config['model'] == "urnn3":
    #     print("Using windowed URNN with Curtis's recurrence")
    #     hw = urnn_window.create_model({
    #         'input_height': config['network']['input_height'],
    #         'cnn_out_size': config['network']['cnn_out_size'],
    #         'num_of_channels': 3,
    #         'num_of_outputs': len(idx_to_char) + 1,
    #         'bridge_width': config['network']['bridge_width']
    #     })
    hw.load_state_dict(torch.load(config['model_load_path']))

    if torch.cuda.is_available():
        hw.cuda()
        dtype = torch.cuda.FloatTensor
        print("Using GPU")
    else:
        dtype = torch.FloatTensor
        print("No GPU detected")
    message = ""

    print(char_to_idx)
    voc = " "
    for x in range(1, len(idx_to_char) + 1):
        voc = voc + idx_to_char[x]
    print(voc)


    tot_ce = 0.0
    tot_we = 0.0
    sum_loss = 0.0
    sum_wer = 0.0
    sum_beam_loss = 0.0
    sum_beam_wer = 0.0
    steps = 0.0
    hw.eval()
    # idx_to_char[0] = ''
    print(idx_to_char)
    print("Validation Set Size = " + str(len(val_dataloader)))
    # , model_path="common_crawl_00.prune01111.trie.klm"
    #decoder = ctcdecode.CTCBeamDecoder(voc, beam_width=100, beta = 0, model_path="iam5.klm",  blank_id=0, log_probs_input=True)
    # for x in val_dataloader:
    #     if x is None:
    #         continue
    #     with torch.no_grad():
    #         line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False)
    #         # labels =  Variable(x['labels'], requires_grad=False, volatile=True)
    #         # label_lengths = Variable(x['label_lengths'], requires_grad=False, volatile=True)
    #         preds = hw(line_imgs)
    #         output_batch = preds.permute(1, 0, 2)
    #         beam_result, beam_scores, timesteps, out_seq_len = decoder.decode(output_batch)
    #         beam_result = beam_result[:,0,]
    #         beam_results = []
    #         o = 0
    #         for i in beam_result:
    #             beam_results.append(i[:out_seq_len[o,0].data.cpu().numpy()].data.cpu().numpy())
    #             o+=1
    #         beam_strings = []
    #         for i in beam_results:
    #             beam_strings.append(string_utils.label2str(i, idx_to_char, False))
    #         out = output_batch.data.cpu().numpy()
    #         for i, gt_line in enumerate(x['gt']):
    #             logits = out[i, ...]
    #             pred, raw_pred = string_utils.naive_decode(logits)
    #             pred_str = string_utils.label2str(pred, idx_to_char, False)
    #             # print(gt_line)
    #             # lex_pred = string_utils.lexicon_decode(preds[:,i:],pred_str,raw_pred,lexicon,char_to_idx = char_to_idx)
    #             # print("-----------------")
    #             # print(gt_line)
    #             # print(pred_str)
    #             # print(beam_strings[i])
    #             # print("-----------------")
    #             wer = error_rates.wer(gt_line, pred_str)
    #             beam_wer = error_rates.wer(gt_line, beam_strings[i])
    #             sum_wer += wer
    #             sum_beam_wer += beam_wer
    #             cer = error_rates.cer(gt_line, pred_str)
    #             beam_cer = error_rates.cer(gt_line, beam_strings[i])
    #             tot_we += wer * len(gt_line.split())
    #             tot_ce += cer * len(u' '.join(gt_line.split()))
    #             sum_loss += cer
    #             sum_beam_loss += beam_cer
    #             steps += 1
    #
    # message = message + "\nTest CER: " + str(sum_loss / steps)
    # message = message + "\nTest WER: " + str(sum_wer / steps)
    # message = message + "\nBeam CER: " + str(sum_beam_loss / steps)
    # message = message + "\nBeam WER: " + str(sum_beam_wer / steps)
    # print("Validation CER", sum_loss / steps)
    # print("Validation WER", sum_wer / steps)
    # print("Beam CER", sum_beam_loss / steps)
    # print("Beam wER", sum_beam_wer / steps)
    # print("Total character Errors:", tot_ce)
    # print("Total word errors", tot_we)
    # tot_ce = 0.0
    # tot_we = 0.0
    # sum_loss = 0.0
    # sum_wer = 0.0
    # sum_beam_loss = 0.0
    # sum_beam_wer = 0.0
    # steps = 0.0
    # hw.eval()
    #
    # if not RIMES:
    #     print("Validation 2 Set Size = " + str(len(val2_dataloader)))
    #     for x in val2_dataloader:
    #         if x is None:
    #             continue
    #         with torch.no_grad():
    #             line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False)
    #             # labels =  Variable(x['labels'], requires_grad=False, volatile=True)
    #             # label_lengths = Variable(x['label_lengths'], requires_grad=False, volatile=True)
    #             preds = hw(line_imgs)
    #             preds = preds.permute(1, 0, 2)
    #             beam_result, beam_scores, timesteps, out_seq_len = decoder.decode(preds)
    #             beam_result = beam_result[:, 0, ]
    #             beam_results = []
    #             o = 0
    #             for i in beam_result:
    #                 beam_results.append(i[:out_seq_len[o, 0].data.cpu().numpy()].data.cpu().numpy())
    #                 o += 1
    #             beam_strings = []
    #             for i in beam_results:
    #                 beam_strings.append(string_utils.label2str(i, idx_to_char, False))
    #             output_batch = preds
    #             out = output_batch.data.cpu().numpy()
    #             for i, gt_line in enumerate(x['gt']):
    #                 logits = out[i, ...]
    #                 pred, raw_pred = string_utils.naive_decode(logits)
    #                 pred_str = string_utils.label2str(pred, idx_to_char, False)
    #                 # print("-----------------")
    #                 # print(gt_line)
    #                 # print(pred_str)
    #                 # print(beam_strings[i])
    #                 # print("-----------------")
    #                 wer = error_rates.wer(gt_line, pred_str)
    #                 beam_wer = error_rates.wer(gt_line, beam_strings[i])
    #                 sum_wer += wer
    #                 sum_beam_wer += beam_wer
    #                 cer = error_rates.cer(gt_line, pred_str)
    #                 beam_cer = error_rates.cer(gt_line, beam_strings[i])
    #                 tot_we += wer * len(gt_line.split())
    #                 tot_ce += cer * len(u' '.join(gt_line.split()))
    #                 sum_loss += cer
    #                 sum_beam_loss += beam_cer
    #                 steps += 1
    #
    #     message = message + "\nTest CER: " + str(sum_loss / steps)
    #     message = message + "\nTest WER: " + str(sum_wer / steps)
    #     message = message + "\nBeam CER: " + str(sum_beam_loss / steps)
    #     message = message + "\nBeam WER: " + str(sum_beam_wer / steps)
    #     print("Validation CER", sum_loss / steps)
    #     print("Validation WER", sum_wer / steps)
    #     print("Beam CER", sum_beam_loss / steps)
    #     print("Beam wER", sum_beam_wer / steps)
    #     print("Total character Errors:", tot_ce)
    #     print("Total word errors", tot_we)
    #     tot_ce = 0.0
    #     tot_we = 0.0
    #     sum_loss = 0.0
    #     sum_wer = 0.0
    #     sum_beam_loss = 0.0
    #     sum_beam_wer = 0.0
    #     steps = 0.0
    #     hw.eval()
    # print("Test Set Size = " + str(len(test_dataloader)))


    for x in test_dataloader:
        if x is None:
            continue
        with torch.no_grad():
            line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False)
            # labels =  Variable(x['labels'], requires_grad=False, volatile=True)
            # label_lengths = Variable(x['label_lengths'], requires_grad=False, volatile=True)
            preds = hw(line_imgs)
            preds = preds.permute(1, 0, 2)
            #beam_result, beam_scores, timesteps, out_seq_len = decoder.decode(preds)
            #beam_result = beam_result[:, 0, ]
            #beam_results = []
            #o = 0
            #for i in beam_result:
            #    beam_results.append(i[:out_seq_len[o, 0].data.cpu().numpy()].data.cpu().numpy())
            #    o += 1
            #beam_strings = []
            #for i in beam_results:
            #    beam_strings.append(string_utils.label2str(i, idx_to_char, False))
            output_batch = preds
            out = output_batch.data.cpu().numpy()
            for i, gt_line in enumerate(x['gt']):
                logits = out[i, ...]
                pred, raw_pred = string_utils.naive_decode(logits)
                pred_str = string_utils.label2str(pred, idx_to_char, False)
                # print("-----------------")
                # print(gt_line)
                # print(pred_str)
                # print(beam_strings[i])
                # print("-----------------")
                wer = error_rates.wer(gt_line, pred_str)
                #beam_wer = error_rates.wer(gt_line, beam_strings[i])
                sum_wer += wer
                #sum_beam_wer += beam_wer
                cer = error_rates.cer(gt_line, pred_str)
                #beam_cer = error_rates.cer(gt_line, beam_strings[i])
                tot_we += wer * len(gt_line.split())
                tot_ce += cer * len(u' '.join(gt_line.split()))
                sum_loss += cer
                #sum_beam_loss += beam_cer
                steps += 1

    message = message + "\nTest CER: " + str(sum_loss / steps)
    message = message + "\nTest WER: " + str(sum_wer / steps)
    #message = message + "\nBeam CER: " + str(sum_beam_loss / steps)
    #message = message + "\nBeam WER: " + str(sum_beam_wer / steps)
    print("Validation CER", sum_loss / steps)
    print("Validation WER", sum_wer / steps)
    #print("Beam CER", sum_beam_loss / steps)
    #print("Beam wER", sum_beam_wer / steps)
    print("Total character Errors:", tot_ce)
    print("Total word errors", tot_we)
    tot_ce = 0.0
    tot_we = 0.0
    sum_loss = 0.0
    sum_wer = 0.0
    #sum_beam_loss = 0.0
    #sum_beam_wer = 0.0
    steps = 0.0



if __name__ == "__main__":
    main()
