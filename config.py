UNK_idx = 0
PAD_idx = 1
EOS_idx = 2
SOS_idx = 3
USR_idx = 4
SYS_idx = 5
SIT_idx = 6
CLS_idx = 7
SEP_idx = 8

import argparse, os, torch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='empathetic_dialogues', help='`empathetic_dialogues`, `cornell_movie` or `dailydialog`')
parser.add_argument("--model", type=str, default="multihop", help='model can be one of `trs`, `cause`, `strategy`, and `multihop`')
dataset = parser.parse_args().dataset
if dataset not in ['empathetic_dialogues', 'cornell_movie', 'dailydialog']:
    raise ValueError('dataste not be one of `empathetic_dialogues`, `cornell_movie` or `dailydialog`')
parser.add_argument('--data_dict', type=str, default='../data/{}/'.format(dataset), help='name of data dictionary')
parser.add_argument('--data_path', type=str, default='../data/{}/dataset_preproc.p'.format(dataset), help='name of data file')
parser.add_argument('--data_vocab', type=str, default='../data/{}/vocab.txt'.format(dataset), help='name of data vocabulary file')
parser.add_argument('--glove_path', type=str, default='../glove.6B/glove.6B.300d.txt'.format(dataset), help='name of vocab embedding file')
parser.add_argument("--emb_path", type=str, default="utils/embedding.txt")
parser.add_argument("--emb_file", type=str, default="../glove.6B/glove.6B.{}d.txt")
parser.add_argument('--save_path', type=str, default='', help='name of save file')
parser.add_argument("--posembedding_path", type=str, default="../data/{}/embedding_pos.txt".format(dataset))
parser.add_argument('--concept_vocab', type=str, default='', help='name of concept vocabulary file')
parser.add_argument('--concept_rel', type=str, default='', help='name of concept relation file')
parser.add_argument('--conceptnet', type=str, default='', help='name of conceptnet file')
parser.add_argument('--conceptnet_graph', type=str, default='', help='name of conceptnet graph file')
parser.add_argument('--triple_dict', type=str, default='', help='name of conceptnet graph file')

parser.add_argument("--hidden_dim", type=int, default=300)
parser.add_argument("--cause_hidden_dim", type=int, default=128)
parser.add_argument("--emb_dim", type=int, default=300)
parser.add_argument('--bz', type=int, default=32, help='the size of batch')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--schedule', type=int, default=500, help='schedule step')
parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay rate')
parser.add_argument('--gs', type=int, default=10000, help='total number of global steps')
parser.add_argument("--beam_size", type=int, default=5)

parser.add_argument("--weight_sharing", type=bool, default=True)#action="store_true")
parser.add_argument("--label_smoothing", type=bool, default=True)#, action="store_true")
parser.add_argument("--noam", type=bool, default=True)#action="store_true")
parser.add_argument("--universal", type=bool, default=True)#action="store_true")
parser.add_argument("--emo_multitask", type=bool, default=True)#action="store_true")
parser.add_argument("--cause_multitask", action="store_true")
parser.add_argument("--act", type=bool, default=True)#action="store_true")
parser.add_argument("--act_loss_weight", type=float, default=0.001)
parser.add_argument("--pretrain_emb", type=bool, default=True)#action="store_true")
parser.add_argument("--test", type=bool, default=False)#action="store_true")


## transformer
parser.add_argument("--hop", type=int, default=6)
parser.add_argument("--heads", type=int, default=8)
parser.add_argument("--depth", type=int, default=40)
parser.add_argument("--filter", type=int, default=50)

## graph
parser.add_argument("--hop_num", type=int, default=2)
parser.add_argument("--max_graph", type=int, default=5)
parser.add_argument("--max_mem_size", type=int, default=400)
parser.add_argument("--max_triple_size", type=int, default=1000)

def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)


arg = parser.parse_args()
print_opts(arg)
model = arg.model

# >>>>>>>>>> hyperparameters >>>>>>>>>> #
emb_dim = arg.emb_dim
hidden_dim = arg.hidden_dim
cause_hidden_dim = arg.cause_hidden_dim
### transformer
hop = arg.hop
heads = arg.heads
depth = arg.depth
filter = arg.filter
hop_num = arg.hop_num
pretrain_emb = arg.pretrain_emb
label_smoothing = arg.label_smoothing
weight_sharing = arg.weight_sharing
noam = arg.noam
universal = arg.universal
act = arg.act
act_loss_weight = arg.act_loss_weight
emo_multitask = arg.emo_multitask
cause_multitask = arg.cause_multitask

# >>>>>>>>>> data path >>>>>>>>>> #
dataset = arg.dataset
data_dict = arg.data_dict
data_npy_dict = arg.data_dict + 'npyFile/'
data_concept_dict = arg.data_dict + 'concept/'
if not os.path.exists(data_npy_dict):
    os.mkdir(data_npy_dict)
data_path = arg.data_path
data_vocab = arg.data_vocab
embed_path = arg.glove_path
emb_path = arg.emb_path
emb_file = arg.emb_file.format(str(emb_dim))
save_path = arg.save_path if arg.save_path else './save/{}/'.format(model)
posembedding_path = arg.posembedding_path
concept_vocab = arg.concept_vocab if arg.concept_vocab else '../conceptnet/concept.txt'
concept_rel = arg.concept_rel if arg.concept_rel else '../conceptnet/relation.txt'
conceptnet = arg.conceptnet if arg.conceptnet else '../conceptnet/concept.en.csv'
conceptnet_graph = arg.conceptnet_graph if arg.conceptnet_graph else '../conceptnet/concept.graph'

# >>>>>>>>>> training parameters >>>>>>>>>> #
bz = arg.bz
lr = arg.lr
schedule = arg.schedule
weight_decay = arg.weight_decay
test = arg.test
beam_size = arg.beam_size
device = 'cuda' if torch.cuda.is_available() else 'cpu'




