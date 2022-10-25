SAVE_PATH=/yrfs1/intern/tjshen/PIE-model/Tr_1kw_450_sh_PIE_off
DATA_PATH=/yrfs1/intern/tjshen/OGB/wikikg90m_v1/
#source /home/intern/tjshen/.bashrc
#conda activate wk90m-3-py37
# --hidden_dim 600 --gamma 10  --valid --test  -adv --mix_cpu_gpu --num_proc 4 --num_thread 4 \
CUDA_VISIBLE_DEVICES=0,1,2,3  dglke_train --model_name TransE_l2 \
  --hidden_dim 450 --gamma 10  --valid  -adv --mix_cpu_gpu --num_proc 4 --num_thread 4 \
  --gpu 0 1 2 3 \
  --async_update \
  --print_on_screen \
  --encoder_model_name shallow \
  --save_path $SAVE_PATH \
  --data_path $DATA_PATH \
  --neg_sample_size 8192 --batch_size 8192 --lr 0.1 --regularization_coef 1.0e-9 \
  --max_step 10000000 --force_sync_interval 1000 --eval_interval 20000 --save_step 500000\
  --val_candidate_path /train14/sli/jswang19/OGB/wikikg90m-v2/candidate/val_t_candidate_PIE.npy \
  --trainhrt_path /train14/sli/jswang19/OGB/wikikg90m-v2/processed/train_hrt_origin.npy \
  --no_load \
  #--load_path /yrfs1/intern/tjshen/OGB/wikikg90m_v1/wikikg90m_kddcup2021/processed/ \
  #--index 6499999 \
  
  #--no_edit_vr_emb \
  #--no_load \