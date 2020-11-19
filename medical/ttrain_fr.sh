# See what problems, models, and hyperparameter sets are available.
# You can easily swap between them (and add new ones).
# t2t-trainer --registry_help

PROBLEM=translate_enfr_wmt32k
MODEL=transformer
HPARAMS=transformer_base_single_gpu

DATA_DIR=$HOME/t2t_data/fr
TMP_DIR=/tmp/t2t_datagen/fr
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS/fr

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Generate data
t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM

# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR
