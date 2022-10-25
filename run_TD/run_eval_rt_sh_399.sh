SAVE_PATH=/yrfs1/intern/tjshen/PIE-eval/Rt_3m9_sh_TD_PIE6w_RM32_off
DATA_PATH=/train14/sli/jswang19/OGB/
#source /home/intern/tjshen/.bashrc
#conda activate wk90m-3-py37
# --hidden_dim 600 --gamma 10  --valid --test  -adv --mix_cpu_gpu --num_proc 4 --num_thread 4 \
#CUDA_VISIBLE_DEVICES=0,1,2,3  
#--gpu 0 1 2 3 \
#
dglke_train --model_name RotatE \
  --hidden_dim 200 --gamma 12  --valid --test --test_mode test-dev  -adv --num_proc 4 --num_thread 4  \
  --async_update \
  --print_on_screen -de \
  --encoder_model_name shallow \
  --save_path $SAVE_PATH \
  --data_path $DATA_PATH \
  --neg_sample_size 64 --batch_size 64 --lr 0 --regularization_coef 1.0e-9 \
  --max_step 4 --force_sync_interval 1 --eval_interval 1 \
  --val_candidate_path /train14/sli/jswang19/OGB/wikikg90m-v2/merge_candidate/TD_PIE6w_RM32.npy \
  --trainhrt_path /train14/sli/jswang19/OGB/wikikg90m-v2/processed/new_sample_hrt.npy \
  --load_path /train14/sli/jswang19/OGB/hcgu/feat_trained/3999999/ \
  --index 3999999 --no_save_emb \
  #--val_candidate_path /train14/sli/jswang19/OGB/wikikg90m-v2/processed/val_t_candidate_PIE.npy \
  #--val_candidate_path /train14/sli/jswang19/OGB/wikikg90m-v2/merge_candidate/PIE_RM2_.npy \
  #--no_load \