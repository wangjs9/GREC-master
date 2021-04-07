import config
import os
import torch
from torch.nn.init import xavier_uniform_
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from models.common_layer import evaluate, count_parameters, make_infinite
from generation.trs import Transformer
from generation.trs_multihop import MultiHopCause
from generation.trs_cause_effect import CauseEffect
from generation.trs_mulexpert import Transformer_experts
from generation.wo_refering import Wo_Refer
from generation.wo_encoder import Wo_Encoder
from generation.wo_graph import Wo_Graph
from generation.mime import MIME
from generation.seq2seq import S2S
from generation.trs_act import Effect
from dataprocess.data_loader import prepare_data_seq

def find_model_path(save_path):
    if not os.path.exists(save_path):
        return None
    list = os.listdir(save_path)
    list = [ele for ele in list if ele[:12] == 'best_model_1']
    if list == []:
        return None
    list.sort(key=lambda fn: os.path.getmtime(save_path + fn))
    model_path = os.path.join(save_path, list[-1])
    print(model_path)
    return model_path

def train_eval():
    data_loader_tra, data_loader_val, data_loader_tst, vocab, program_number = prepare_data_seq(
        batch_size=config.bz)

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    model_file_path = find_model_path(config.save_path)

    if config.model == "trs":
        model = Transformer(vocab, decoder_number=program_number, model_file_path=model_file_path)
    elif config.model == 'multihop':
        model = MultiHopCause(vocab, decoder_number=program_number, model_file_path=model_file_path)
    elif config.model == 'multiexpert':
        model = Transformer_experts(vocab, decoder_number=program_number, model_file_path=model_file_path)
    elif config.model == 'mime':
        model = MIME(vocab, decoder_number=program_number, model_file_path=model_file_path)
    elif config.model == 's2s':
        model = S2S(vocab, decoder_number=program_number, model_file_path=model_file_path)
    elif config.model == 'w.o/refer':
        model = Wo_Refer(vocab, decoder_number=program_number, model_file_path=model_file_path)
    elif config.model == 'w.o/encoder':
        model = Wo_Encoder(vocab, decoder_number=program_number, model_file_path=model_file_path)
    elif config.model == 'w.o/graph':
        model = Wo_Graph(vocab, decoder_number=program_number, model_file_path=model_file_path)
    elif config.model == 'act':
        model = Effect(vocab, decoder_number=program_number, model_file_path=model_file_path)
    elif config.model == 'cause-effect':
        model = CauseEffect(vocab, decoder_number=program_number, model_file_path=model_file_path)

    if model_file_path is None:
        for n, p in model.named_parameters():
            if p.dim() > 1 and n != "embedding.lut.weight" and config.pretrain_emb:
                xavier_uniform_(p)

    if config.test:
        print('Test model', config.model)
        print('TRAINABLE PARAMETERS', count_parameters(model))
        model.to(config.device)
        model = model.eval()
        evaluate(model, data_loader_tst, config.dataset, config.save_path, ty="test", max_dec_step=50, save=True)

    print('MODEL USED', config.model)
    print('TRAINABLE PARAMETERS', count_parameters(model))

    check_iter = 1000

    try:
        model.to(config.device)
        init_iter = model.iter + 1
        best_ppl = model.current_loss
        patient = 0
        weights_best = deepcopy(model.state_dict())
        data_iter = make_infinite(data_loader_tra)
        model = model.train()
        for n_iter in tqdm(range(50000)):
            loss, ppl, bce, acc = model.train_one_batch(next(data_iter), n_iter)

            if (n_iter + 1) % check_iter == 0:
                model = model.eval()
                loss_val, ppl_val, bce_val, acc_val, bleu_score_g, bleu_score_b, rouge_score_g, rouge_score_b = evaluate(model, data_loader_val,
                                                                                           config.dataset, config.save_path, ty="valid", max_dec_step=50)
                model = model.train()

                if n_iter + init_iter < 2600:
                    continue

                if ppl_val <= best_ppl:
                    best_ppl = ppl_val
                    patient = 0
                    model.save_model(best_ppl, n_iter+init_iter, 0, 0, bleu_score_g, bleu_score_b)
                    weights_best = deepcopy(model.state_dict())
                else:
                    patient += 1
                if patient > 2:
                    break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    model.load_state_dict({name: weights_best[name] for name in weights_best})
    model.eval()
    model.epoch = 100
    loss_test, ppl_test, bce_test, acc_test, bleu_score_g, bleu_score_b, rouge_score_g, rouge_score_b = evaluate(model,
               data_loader_tst, config.dataset, config.save_path, ty="test", max_dec_step=50, save=True)

    file_summary = config.save_path + "summary.txt"
    with open(file_summary, 'w') as the_file:
        the_file.write("EVAL\tLoss\tPPL\tAccuracy\tBleu_g\tBleu_b\tROUGE_g\tROUGE_b\n")
        the_file.write(
            "{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n".format("test", loss_test, ppl_test, acc_test,
                                                                                  bleu_score_g, bleu_score_b,
                                                                                  rouge_score_g, rouge_score_b))



if __name__ == '__main__':
    train_eval()