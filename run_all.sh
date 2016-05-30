echo 500 units boinky doink
th train.lua -data_dir data/warandpeace -checkpoint_dir cv500 -rnn_size 500 -num_layers 1 -dropout 0.5 -seq_length 12 -batch_size 30 -grad_clip 15 -print_every 20 -eval_val_every 4000 -max_epochs 2 -fudge_factor 1.0 -fudge_recurrent 1.0
echo 1000 units boinky doink
th train.lua -data_dir data/warandpeace -checkpoint_dir cv1000 -rnn_size 1000 -num_layers 1 -dropout 0.5 -seq_length 12 -batch_size 30 -grad_clip 15 -print_every 20 -eval_val_every 4000 -max_epochs 2 -fudge_factor 1.0 -fudge_recurrent 1.0
echo 1500 units boinky doink
th train.lua -data_dir data/warandpeace -checkpoint_dir cv1500 -rnn_size 1500 -num_layers 1 -dropout 0.5 -seq_length 12 -batch_size 30 -grad_clip 15 -print_every 20 -eval_val_every 4000 -max_epochs 2 -fudge_factor 1.0 -fudge_recurrent 1.0
echo 2000 units boinky doink
th train.lua -data_dir data/warandpeace -checkpoint_dir cv2000 -rnn_size 2000 -num_layers 1 -dropout 0.5 -seq_length 12 -batch_size 30 -grad_clip 15 -print_every 20 -eval_val_every 20000 -max_epochs 2 -fudge_factor 1.0 -fudge_recurrent 1.0
echo 2500 units boinky doink
th train.lua -data_dir data/warandpeace -checkpoint_dir cv2500 -rnn_size 2500 -num_layers 1 -dropout 0.5 -seq_length 12 -batch_size 30 -grad_clip 15 -print_every 20 -eval_val_every 4000 -max_epochs 2 -fudge_factor 1.0 -fudge_recurrent 1.0
echo 3000 units boinky doink
th train.lua -data_dir data/warandpeace -checkpoint_dir cv3000 -rnn_size 3000 -num_layers 1 -dropout 0.5 -seq_length 12 -batch_size 30 -grad_clip 15 -print_every 20 -eval_val_every 4000 -max_epochs 2 -fudge_factor 1.0 -fudge_recurrent 1.0
echo 3500 units boinky doink
th train.lua -data_dir data/warandpeace -checkpoint_dir cv3500 -rnn_size 3500 -num_layers 1 -dropout 0.5 -seq_length 12 -batch_size 30 -grad_clip 15 -print_every 20 -eval_val_every 4000 -max_epochs 2 -fudge_factor 1.0 -fudge_recurrent 1.0
echo 4000 units boinky doink
th train.lua -data_dir data/warandpeace -checkpoint_dir cv4000 -rnn_size 4000 -num_layers 1 -dropout 0.5 -seq_length 12 -batch_size 30 -grad_clip 15 -print_every 20 -eval_val_every 4000 -max_epochs 2 -fudge_factor 1.0 -fudge_recurrent 1.0
echo 4500 units boinky doink
th train.lua -data_dir data/warandpeace -checkpoint_dir cv4500 -rnn_size 4500 -num_layers 1 -dropout 0.5 -seq_length 12 -batch_size 30 -grad_clip 15 -print_every 20 -eval_val_every 4000 -max_epochs 2 -fudge_factor 1.0 -fudge_recurrent 1.0
echo 5000 units boinky doink
th train.lua -data_dir data/warandpeace -checkpoint_dir cv5000 -rnn_size 5000 -num_layers 1 -dropout 0.5 -seq_length 12 -batch_size 30 -grad_clip 15 -print_every 20 -eval_val_every 4000 -max_epochs 2 -fudge_factor 1.0 -fudge_recurrent 1.0
