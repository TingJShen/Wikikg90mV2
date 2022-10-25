SAVE_PATH=/yrfs1/intern/tjshen/PIE-eval/Cont_Cp_2m9_sh_TD_PIE_100100_Transe_index_precise
DATA_PATH=/train14/sli/jswang19/OGB/
#source /home/intern/tjshen/.bashrc
#conda activate wk90m-3-py37
# --hidden_dim 600 --gamma 10  --valid --test  -adv --mix_cpu_gpu --num_proc 4 --num_thread 4 \
CUDA_VISIBLE_DEVICES=0,1,2,3  dglke_train --model_name ComplEx \
  --hidden_dim 200 --gamma 10  --valid  -adv --mix_cpu_gpu --num_proc 4 --num_thread 4 \
  --async_update \
  --gpu 0 1 2 3 \
  --print_on_screen \
  --encoder_model_name shallow \
  --save_path $SAVE_PATH \
  --data_path $DATA_PATH \
  --neg_sample_size 8192 --batch_size 8192 --lr 0.1 --regularization_coef 1.0e-9 \
  --max_step 250000  --force_sync_interval 1000 --eval_interval 5000 --save_step 40000\
  --val_candidate_path /train14/sli/jswang19/OGB/wikikg90m-v2/processed/val_t_candidate_PIE.npy \
  --trainhrt_path /yrfs1/intern/tjshen/OGB/new_sample_faiss_100for100.npy \
  --load_path /yrfs1/intern/tjshen/PIE-model/Cp_5m_sh_PIE_off/ComplEx_wikikg90m_shallow_d_200_g_10.00 \
  --index 2999999 --no_save_emb \
  #--val_candidate_path /train14/sli/jswang19/OGB/wikikg90m-v2/processed/val_t_candidate_PIE.npy \
  #--val_candidate_path /train14/sli/jswang19/OGB/wikikg90m-v2/merge_candidate/PIE_RM2_.npy \
  #--no_load \