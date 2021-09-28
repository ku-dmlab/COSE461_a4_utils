from collections import namedtuple
import os
import math
import sys
import time

import sacrebleu
import numpy as np
from typing import List
from tqdm import tqdm
import torch
import torch.nn.utils
import torch.nn.functional as F
import torch.nn.utils

from utils import read_corpus, batch_iter
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

def beam_search_sent(model, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param model (NMT): a model to perform beam_search
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        src_sents_var = model.vocab.src.to_input_tensor([src_sent], model.device)

        src_encodings, dec_init_vec = encode(model, src_sents_var, [len(src_sent)])
        src_encodings_att_linear = model.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, model.hidden_size, device=model.device)

        eos_id = model.vocab.tgt['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=model.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([model.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=model.device)
            y_t_embed = model.model_embeddings.target(y_tm1)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _  = step(model, x, h_tm1,
                                                      exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(model.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos // len(model.vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(model.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = model.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=model.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=model.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses


def beam_search(model, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model (NMT): NMT Model
    @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    was_training = model.training
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            example_hyps = beam_search_sent(model, src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

            hypotheses.append(example_hyps)

    if was_training: model.train(was_training)

    return hypotheses


def evaluate_ppl(model, dev_data, batch_size=32):
    """ Evaluate perplexity on dev sentences
    @param model (NMT): NMT Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
    @param references (List[List[str]]): a list of gold-standard reference target sentences
    @param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
    @returns bleu_score: corpus-level BLEU score
    """
    # remove the start and end tokens
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]
    
    # detokenize the subword pieces to get full sentences
    detokened_refs = [''.join(pieces).replace('▁', ' ') for pieces in references]
    detokened_hyps = [''.join(hyp.value).replace('▁', ' ') for hyp in hypotheses]

    # sacreBLEU can take multiple references (golden example per sentence) but we only feed it one
    bleu = sacrebleu.corpus_bleu(detokened_hyps, [detokened_refs])

    return bleu.score


def set_seed(seed=0, use_cuda=True):
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)


def train(model_cls, vocab_cls, util_dir):
    set_seed()
    train_data_src = read_corpus(os.path.join(util_dir, "chr_en_data/train.chr"),
                                 source='src', vocab_size=21000)       
    train_data_tgt = read_corpus(os.path.join(util_dir, "chr_en_data/train.en"),
                                 source='tgt', vocab_size=8000)

    dev_data_src = read_corpus(os.path.join(util_dir, "chr_en_data/dev.chr"), 
                               source='src', vocab_size=3000)
    dev_data_tgt = read_corpus(os.path.join(util_dir, "chr_en_data/dev.en"), 
                               source='tgt', vocab_size=2000)

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = 32
    clip_grad = 5.0
    valid_niter = 200
    log_every = 10
    patience_limit = 1
    max_num_trial = 5
    lr_decay = 0.5
    max_epoch = 30
    model_save_path = "model.bin"

    vocab = vocab_cls.load("vocab.json")
    model = model_cls(embed_size=1024,
                hidden_size=1024,
                dropout_rate=0.3,
                vocab=vocab)
    model.train()

    uniform_init = 0.1
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0

    device = torch.device("cuda:0")
    print('use device: %s' % device, file=sys.stderr)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1

            optimizer.zero_grad()

            batch_size = len(src_sents)

            example_losses = -model(src_sents, tgt_sents) # (batch_size,)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cum_examples,
                                                                                         np.exp(cum_loss / cum_tgt_words),
                                                                                         cum_examples), file=sys.stderr)

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = evaluate_ppl(model, dev_data, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                elif patience < patience_limit:
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == patience_limit:
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial >= max_num_trial:
                            print('early stop!', file=sys.stderr)
                            return

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * lr_decay
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

                if epoch == max_epoch:
                    print('reached maximum number of epochs!', file=sys.stderr)
                    return


def test(model_cls, vocab_cls, util_dir):
    """ Performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """
    set_seed()
    test_source_file = os.path.join(util_dir, "chr_en_data/test.chr")
    test_target_file = os.path.join(util_dir, "chr_en_data/test.en")
    model_path = "model.bin"

    print("load test source sentences from [{}]".format(test_source_file), file=sys.stderr)
    test_data_src = read_corpus(test_source_file, source='src', vocab_size=3000)
    print("load test target sentences from [{}]".format(test_target_file), file=sys.stderr)
    test_data_tgt = read_corpus(test_target_file, source='tgt', vocab_size=2000)

    print("load model from {}".format(model_path), file=sys.stderr)
    model = model_cls.load(model_path)
    model = model.to(torch.device("cuda:0"))

    hypotheses = beam_search(model, test_data_src,          
                             beam_size=10,
                             max_decoding_time_step=70)

    top_hypotheses = [hyps[0] for hyps in hypotheses]
    bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
    print('Corpus BLEU: {}'.format(bleu_score), file=sys.stderr)

    with open("test_outputs.txt", 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ''.join(top_hyp.value).replace('▁', ' ')
            f.write(hyp_sent + '\n')
