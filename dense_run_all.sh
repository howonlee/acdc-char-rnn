echo 500 units boop doop
th train.lua -data_dir data/warandpeace -checkpoint_dir densecv500 -rnn_size 500 -num_layers 1 -dropout 0.5 -seq_length 12 -batch_size 30 -grad_clip 15 -print_every 20 -eval_val_every 4000 -max_epochs 2 -model denselstm
echo 1000 units boop doop
th train.lua -data_dir data/warandpeace -checkpoint_dir densecv1000 -rnn_size 1000 -num_layers 1 -dropout 0.5 -seq_length 12 -batch_size 30 -grad_clip 15 -print_every 20 -eval_val_every 4000 -max_epochs 2 -model denselstm
echo 1500 units boop doop
th train.lua -data_dir data/warandpeace -checkpoint_dir densecv1500 -rnn_size 1500 -num_layers 1 -dropout 0.5 -seq_length 12 -batch_size 30 -grad_clip 15 -print_every 20 -eval_val_every 4000 -max_epochs 2 -model denselstm
echo 2000 units boop doop
th train.lua -data_dir data/warandpeace -checkpoint_dir densecv2000 -rnn_size 2000 -num_layers 1 -dropout 0.5 -seq_length 12 -batch_size 30 -grad_clip 15 -print_every 20 -eval_val_every 4000 -max_epochs 2 -model denselstm
echo after this the damned things don't fit in GPU memory anymore
