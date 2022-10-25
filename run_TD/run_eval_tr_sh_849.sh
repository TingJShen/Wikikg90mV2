SAVE_PATH=/yrfs1/intern/tjshen/PIE-eval/Tr_8m4_sh_TD_PIE6w_RM32_RM
DATA_PATH=/train14/sli/jswang19/OGB/
#source /home/intern/tjshen/.bashrc
#conda activate wk90m-3-py37
# --hidden_dim 600 --gamma 10  --valid --test  -adv --mix_cpu_gpu --num_proc 4 --num_thread 4 \
#CUDA_VISIBLE_DEVICES=0,1,2,3  
#--gpu 0 1 2 3 \
dglke_train --model_name TransE_l2 \
  --hidden_dim 200 --gamma 10  --valid --test --test_mode test-dev -adv --mix_cpu_gpu --num_proc 1 --num_thread 1 \
  --async_update \
  --print_on_screen \
  --encoder_model_name shallow \
  --save_path $SAVE_PATH \
  --data_path $DATA_PATH \
  --neg_sample_size 512 --batch_size 512 --lr 0 --regularization_coef 1.0e-9 \
  --max_step 3 --eval_interval 1 \
  --val_candidate_path /train14/sli/jswang19/OGB/wikikg90m-v2/merge_candidate/TD_PIE6w_RM32.npy \
  --trainhrt_path /train14/sli/jswang19/OGB/wikikg90m-v2/processed/new_sample_hrt.npy \
  --load_path /yrfs1/intern/tjshen/PIE-model/Tr_1kw_sh_PIE_RM/TransE_l2_wikikg90m_shallow_d_200_g_10.02 \
  --index 8499999 --no_save_emb\
  
  #--val_candidate_path /train14/sli/jswang19/OGB/wikikg90m-v2/merge_candidate/PIE6w_RM32.npy \
  #--no_load \