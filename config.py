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

parser.add_argument('--data_dict', type=str, default='./data/', help='name of data dictionary')
parser.add_argument('--data_path', type=str, default='./data/dataset_preproc.p', help='name of data file')
parser.add_argument('--data_vocab', type=str, default='./data/vocab.txt', help='name of data vocabulary file')
parser.add_argument('--glove_path', type=str, default='/home/csjwang/Documents/glove.6B/glove.6B.300d.txt', help='name of vocab embedding file')
parser.add_argument('--save_path', type=str, default='', help='name of save file')
parser.add_argument("--posembedding_path", type=str, default="./data/embedding_pos.txt")
parser.add_argument('--concept_vocab', type=str, default='', help='name of concept vocabulary file')
parser.add_argument('--concept_rel', type=str, default='', help='name of concept relation file')
parser.add_argument('--conceptnet', type=str, default='', help='name of conceptnet file')
parser.add_argument('--conceptnet_graph', type=str, default='', help='name of conceptnet graph file')
parser.add_argument('--triple_dict', type=str, default='', help='name of conceptnet graph file')
# parser.add_argument('--cp_path', type=str, default='', help='name of concept path file')

parser.add_argument('--dataprocess', type=bool, default=True, help='whether to process the data')

parser.add_argument("--hidden_dim", type=int, default=300)
parser.add_argument("--emb_dim", type=int, default=300)
parser.add_argument('--bz', type=int, default=64, help='the size of batch')
parser.add_argument('--lr', type=int, default=32, help='learning rate')
parser.add_argument('--gs', type=int, default=10000, help='total number of global steps')
parser.add_argument("--beam_size", type=int, default=5)
# parser.add_argument('')

parser.add_argument('--pointer_gen', action="store_true")
parser.add_argument("--weight_sharing", action="store_true")
parser.add_argument("--label_smoothing", action="store_true")
parser.add_argument("--noam", action="store_true")
parser.add_argument("--universal", action="store_true")
parser.add_argument("--emo_multitask", action="store_true")
parser.add_argument("--cause_multitask", action="store_true")
parser.add_argument("--act", action="store_true")
parser.add_argument("--pretrain_emb", action="store_true")
parser.add_argument("--act_loss_weight", type=float, default=0.001)

## transformer
parser.add_argument("--hop", type=int, default=6)
parser.add_argument("--heads", type=int, default=1)
parser.add_argument("--depth", type=int, default=40)
parser.add_argument("--filter", type=int, default=50)

## graph
parser.add_argument("--hop_num", type=int, default=2)

arg = parser.parse_args()

dataprocess = True if arg.dataprocess else False

data_dict = arg.data_dict
data_npy_dict = arg.data_dict + 'npyFile/'
data_concept_dict = arg.data_dict + 'concept/'
if not os.path.exists(data_npy_dict):
    os.mkdir(data_npy_dict)
data_path = arg.data_path
data_vocab = arg.data_vocab
embed_path = arg.glove_path
save_path = arg.save_path if arg.save_path else './save/'
posembedding_path = arg.posembedding_path

concept_vocab = arg.concept_vocab if arg.concept_vocab else '../conceptnet/concept.txt'
concept_rel = arg.concept_rel if arg.concept_rel else '../conceptnet/relation.txt'
conceptnet = arg.conceptnet if arg.conceptnet else '../conceptnet/concept.en.csv'
conceptnet_graph = arg.conceptnet_graph if arg.conceptnet_graph else '../conceptnet/concept.graph'
# cp_path = arg.cp_path if arg.cp_path else './data/paths.json'

# Hyperparameters
hidden_dim = arg.hidden_dim
emb_dim = arg.emb_dim
bz = arg.bz
lr = arg.lr
beam_size = arg.beam_size

USE_CUDA = True if torch.cuda.is_available() else False
pointer_gen = arg.pointer_gen
label_smoothing = arg.label_smoothing
weight_sharing = arg.weight_sharing
noam = arg.noam
universal = arg.universal
act = arg.act
act_loss_weight = arg.act_loss_weight
emo_multitask = arg.emo_multitask
cause_multitask = arg.cause_multitask

### transformer
hop = arg.hop
heads = arg.heads
depth = arg.depth
filter = arg.filter
