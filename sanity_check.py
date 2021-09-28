import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils

from utils import batch_iter

#----------
# CONSTANTS
#----------
BATCH_SIZE = 5
EMBED_SIZE = 3
HIDDEN_SIZE = 3
DROPOUT_RATE = 0.0

def reinitialize_layers(model):
    """ Reinitialize the Layer Weights for Sanity Checks.
    """
    def init_weights(m):
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.3)
            if m.bias is not None:
                m.bias.data.fill_(0.1)
        elif type(m) == nn.Embedding:
            m.weight.data.fill_(0.15)
        elif type(m) == nn.Dropout:
            nn.Dropout(DROPOUT_RATE)
    with torch.no_grad():
        model.apply(init_weights)


def question_1_4_sanity_check(nmt_cls, vocab_cls, utils_dir, encode_fn):
    """ Sanity check for question 1.4. 
        Compares student output to that of model with dummy data.
    """
    print("Running Sanity Check for Question 1.4: Encode")
    print ("-"*80)
    model, src_sents, _, _ = data_for_sanity_check(nmt_cls, vocab_cls, utils_dir)

    # Configure for Testing
    reinitialize_layers(model)
    source_lengths = [len(s) for s in src_sents]
    source_padded = model.vocab.src.to_input_tensor(src_sents, device=model.device)

    # Load Outputs
    enc_hiddens_target = torch.load(f'{utils_dir}/sanity_check_en_es_data/enc_hiddens.pkl')
    dec_init_state_target = torch.load(f'{utils_dir}/sanity_check_en_es_data/dec_init_state.pkl')

    # Test
    with torch.no_grad():
        enc_hiddens_pred, dec_init_state_pred = encode_fn(model, source_padded, source_lengths)
    assert(enc_hiddens_target.shape == enc_hiddens_pred.shape), "enc_hiddens shape is incorrect: it should be:\n {} but is:\n{}".format(enc_hiddens_target.shape, enc_hiddens_pred.shape)
    assert(np.allclose(enc_hiddens_target.numpy(), enc_hiddens_pred.numpy())), "enc_hiddens is incorrect: it should be:\n {} but is:\n{}".format(enc_hiddens_target, enc_hiddens_pred)
    print("enc_hiddens Sanity Checks Passed!")
    assert(dec_init_state_target[0].shape == dec_init_state_pred[0].shape), "dec_init_state[0] shape is incorrect: it should be:\n {} but is:\n{}".format(dec_init_state_target[0].shape, dec_init_state_pred[0].shape)
    assert(np.allclose(dec_init_state_target[0].numpy(), dec_init_state_pred[0].numpy())), "dec_init_state[0] is incorrect: it should be:\n {} but is:\n{}".format(dec_init_state_target[0], dec_init_state_pred[0])
    print("dec_init_state[0] Sanity Checks Passed!")
    assert(dec_init_state_target[1].shape == dec_init_state_pred[1].shape), "dec_init_state[1] shape is incorrect: it should be:\n {} but is:\n{}".format(dec_init_state_target[1].shape, dec_init_state_pred[1].shape) 
    assert(np.allclose(dec_init_state_target[1].numpy(), dec_init_state_pred[1].numpy())), "dec_init_state[1] is incorrect: it should be:\n {} but is:\n{}".format(dec_init_state_target[1], dec_init_state_pred[1])
    print("dec_init_state[1] Sanity Checks Passed!")
    print ("-"*80)
    print("All Sanity Checks Passed for Question 1.4: Encode!")
    print ("-"*80)


def question_1_5_sanity_check(nmt_cls, vocab_cls, utils_dir, decode_fn):
    """ Sanity check for question 1.5. 
        Compares student output to that of model with dummy data.
    """
    print ("-"*80)
    print("Running Sanity Check for Question 1.5: Decode")
    print ("-"*80)
    model, _, _, _ = data_for_sanity_check(nmt_cls, vocab_cls, utils_dir)

    # Load Inputs
    dec_init_state = torch.load(f'{utils_dir}/sanity_check_en_es_data/dec_init_state.pkl')
    enc_hiddens = torch.load(f'{utils_dir}/sanity_check_en_es_data/enc_hiddens.pkl')
    enc_masks = torch.load(f'{utils_dir}/sanity_check_en_es_data/enc_masks.pkl')
    target_padded = torch.load(f'{utils_dir}/sanity_check_en_es_data/target_padded.pkl')

    # Load Outputs
    combined_outputs_target = torch.load(f'{utils_dir}/sanity_check_en_es_data/combined_outputs.pkl')
    print(combined_outputs_target.shape)

    # Configure for Testing
    reinitialize_layers(model)
    COUNTER = [0]
    def stepFunction(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks):
       dec_state = torch.load(f'{utils_dir}/sanity_check_en_es_data/step_dec_state_{COUNTER[0]}.pkl')
       o_t = torch.load(f'{utils_dir}/sanity_check_en_es_data/step_o_t_{COUNTER[0]}.pkl')
       COUNTER[0]+=1
       return dec_state, o_t, None
    model.step = stepFunction

    # Run Tests
    with torch.no_grad():
        combined_outputs_pred = decode_fn(model, enc_hiddens, enc_masks, dec_init_state, target_padded)
    assert(combined_outputs_target.shape == combined_outputs_pred.shape), "combined_outputs shape is incorrect: it should be:\n {} but is:\n{}".format(combined_outputs_target.shape, combined_outputs_pred.shape)
    assert(np.allclose(combined_outputs_pred.numpy(), combined_outputs_target.numpy())), "combined_outputs is incorrect: it should be:\n {} but is:\n{}".format(combined_outputs_target, combined_outputs_pred)
    print("combined_outputs Sanity Checks Passed!")
    print ("-"*80)
    print("All Sanity Checks Passed for Question 1.5: Decode!")
    print ("-"*80)

def question_1_6_sanity_check(nmt_cls, vocab_cls, utils_dir, step_fn):
    """ Sanity check for question 1.6. 
        Compares student output to that of model with dummy data.
    """
    print ("-"*80)
    print("Running Sanity Check for Question 1.6: Step")
    print ("-"*80)
    model, _, _, _ = data_for_sanity_check(nmt_cls, vocab_cls, utils_dir)
    reinitialize_layers(model)

    # Inputs
    Ybar_t = torch.load(f'{utils_dir}/sanity_check_en_es_data/Ybar_t.pkl')
    dec_init_state = torch.load(f'{utils_dir}/sanity_check_en_es_data/dec_init_state.pkl')
    enc_hiddens = torch.load(f'{utils_dir}/sanity_check_en_es_data/enc_hiddens.pkl')
    enc_masks = torch.load(f'{utils_dir}/sanity_check_en_es_data/enc_masks.pkl')
    enc_hiddens_proj = torch.load(f'{utils_dir}/sanity_check_en_es_data/enc_hiddens_proj.pkl')

    # Output
    dec_state_target = torch.load(f'{utils_dir}/sanity_check_en_es_data/dec_state.pkl')
    o_t_target = torch.load(f'{utils_dir}/sanity_check_en_es_data/o_t.pkl')
    e_t_target = torch.load(f'{utils_dir}/sanity_check_en_es_data/e_t.pkl')

    # Run Tests
    with torch.no_grad():
        dec_state_pred, o_t_pred, e_t_pred= step_fn(model, Ybar_t, dec_init_state, enc_hiddens, enc_hiddens_proj, enc_masks)
    assert(dec_state_target[0].shape == dec_state_pred[0].shape), "decoder_state[0] shape is incorrect: it should be:\n {} but is:\n{}".format(dec_state_target[0].shape, dec_state_pred[0].shape)
    assert(np.allclose(dec_state_target[0].numpy(), dec_state_pred[0].numpy())), "decoder_state[0] is incorrect: it should be:\n {} but is:\n{}".format(dec_state_target[0], dec_state_pred[0])
    print("dec_state[0] Sanity Checks Passed!")
    assert(dec_state_target[1].shape == dec_state_pred[1].shape), "decoder_state[1] shape is incorrect: it should be:\n {} but is:\n{}".format(dec_state_target[1].shape, dec_state_pred[1].shape)
    assert(np.allclose(dec_state_target[1].numpy(), dec_state_pred[1].numpy())), "decoder_state[1] is incorrect: it should be:\n {} but is:\n{}".format(dec_state_target[1], dec_state_pred[1])
    print("dec_state[1] Sanity Checks Passed!")
    assert(np.allclose(o_t_target.numpy(), o_t_pred.numpy())), "combined_output is incorrect: it should be:\n {} but is:\n{}".format(o_t_target, o_t_pred)
    print("combined_output  Sanity Checks Passed!")
    assert(np.allclose(e_t_target.numpy(), e_t_pred.numpy())), "e_t is incorrect: it should be:\n {} but is:\n{}".format(e_t_target, e_t_pred)
    print("e_t Sanity Checks Passed!")
    print ("-"*80)    
    print("All Sanity Checks Passed for Question 1.6: Step!")
    print ("-"*80)


def sanity_read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = nltk.word_tokenize(line)
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data

def data_for_sanity_check(nmt_cls, vocab_cls, utils_dir):

    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    # Load training data & vocabulary
    train_data_src = sanity_read_corpus(f'{utils_dir}/sanity_check_en_es_data/train_sanity_check.es', 'src')
    train_data_tgt = sanity_read_corpus(f'{utils_dir}/sanity_check_en_es_data/train_sanity_check.en', 'tgt')
    train_data = list(zip(train_data_src, train_data_tgt))

    for src_sents, tgt_sents in batch_iter(train_data, batch_size=BATCH_SIZE, shuffle=True):
        src_sents = src_sents
        tgt_sents = tgt_sents
        break
    vocab = vocab_cls.load(f'{utils_dir}/sanity_check_en_es_data/vocab_sanity_check.json') 

    # Create NMT Model
    model = nmt_cls(
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        dropout_rate=DROPOUT_RATE,
        vocab=vocab)
    return model, src_sents, tgt_sents, vocab
