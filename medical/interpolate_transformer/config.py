import argparse
import wandb

parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
# * Basics
parser.add_argument('--output_dir',     default=".",        type=str)
parser.add_argument('--modelname',      default="IPT",      type=str)
# * Dataset
parser.add_argument('--sr_size',        default="512*22",      type=str)
# * Training
parser.add_argument('--lr',             default=1e-4,       type=float)
parser.add_argument('--weight_decay',   default=1e-4,       type=float)
parser.add_argument('--batch_size',     default=32,          type=int)
parser.add_argument('--epochs',         default=300,        type=int)
parser.add_argument('--lr_drop',        default=0.85,       type=float)
# * Transformer
parser.add_argument('--enc_layers',         default=6,      type=int,       help="Number of encoding layers in the transformer")
parser.add_argument('--dec_layers',         default=6,      type=int,       help="Number of decoding layers in the transformer")
parser.add_argument('--dim_feedforward',    default=1024,   type=int,       help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--hidden_dim',         default=256,    type=int,       help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--dropout',            default=0.1,    type=float,     help="Dropout applied in the transformer")
parser.add_argument('--nheads',             default=8,      type=int,       help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--num_queries',        default=100,    type=int,       help="Number of query slots")
parser.add_argument('--pre_norm',           action='store_true')
args = parser.parse_args()

wandb.init(project="pytorch-Inp-transformer")
wandb.watch_called = False

# WandB – Config is a variable that holds and saves hyperparameters and inputs
config                  = wandb.config          # Initialize config
config.output_dir       = args.output_dir
config.modelname        = args.modelname

config.lr               = args.lr               # learning rate (default: 0.01)
config.weight_decay     = args.weight_decay          # SGD momentum (default: 0.5) 
config.batch_size       = args.batch_size          # input batch size for training (default: 64)
config.epochs           = args.epochs             # number of epochs to train (default: 10)
config.lr_drop          = args.lr_drop

config.enc_layers       = args.enc_layers
config.dec_layers       = args.dec_layers 
config.dim_feedforward  = args.dim_feedforward
config.hidden_dim       = args.hidden_dim
config.dropout          = args.dropout
config.nheads           = args.nheads
config.num_queries      = args.num_queries
config.pre_norm         = args.pre_norm
config.log_interval     = 10     # how many batches to wait before logging training status