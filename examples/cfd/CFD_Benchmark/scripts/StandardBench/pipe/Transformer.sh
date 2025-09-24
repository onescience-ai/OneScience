mpirun -np 4 python run.py \
--gpu 0 \
--data_path ../Neural-Solver-Library/data/pipe \
--loader pipe \
--geotype structured_2D \
--space_dim 2 \
--fun_dim 2 \
--out_dim 1 \
--model Transformer \
--n_hidden 128 \
--n_heads 8 \
--n_layers 8 \
--mlp_ratio 2 \
--slice_num 64 \
--unified_pos 0 \
--ref 8 \
--batch_size 1 \
--epochs 500 \
--eval 1 \
--use_checkpoint 1 \
--checkpoint_layers blocks.0.Attn,blocks.1.Attn,blocks.2.Attn,blocks.3.Attn,blocks.4.Attn \
--normalize 1 \
--save_name pipe_Transformer
# --use_checkpoint 1 \
# --resume \
