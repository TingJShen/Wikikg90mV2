SAVE_PATH=/yrfs1/intern/tjshen/PIE-eval/Cont_Rt_7m9_sh_PIE_100100_Rotate_index_precise
DATA_PATH=/train14/sli/jswang19/OGB/
#source /home/intern/tjshen/.bashrc
#conda activate wk90m-3-py37
# --hidden_dim 600 --gamma 10  --valid --test  -adv --mix_cpu_gpu --num_proc 4 --num_thread 4 \
#CUDA_VISIBLE_DEVICES=0,1,2,3  
#--gpu 0 1 2 3 \
#
CUDA_VISIBLE_DEVICES=0,1,2,3 dglke_train --model_name RotatE \
  --hidden_dim 200 --gamma 12  --valid  -adv --num_proc 4 --num_thread 4  \
  --async_update \
  --gpu 0 1 2 3 \
  --print_on_screen -de \
  --encoder_model_name shallow \
  --save_path $SAVE_PATH \
  --data_path $DATA_PATH \
  --neg_sample_size 1024 --batch_size 1024 --lr 0.1 --regularization_coef 1.0e-9 \
  --max_step 250000  --force_sync_interval 1000 --eval_interval 5000 --save_step 40000\
  --val_candidate_path /train14/sli/jswang19/OGB/wikikg90m-v2/processed/val_t_candidate_PIE.npy \
  --trainhrt_path /yrfs1/intern/tjshen/OGB/new_sample_faissRt_100for100.npy \
  --load_path /train14/sli/jswang19/OGB/hcgu/feat_trained/7999999_RotateE/ \
  --index 7999999 \
  #--val_candidate_path /train14/sli/jswang19/OGB/wikikg90m-v2/processed/val_t_candidate_PIE.npy \
  #--val_candidate_path /train14/sli/jswang19/OGB/wikikg90m-v2/merge_candidate/PIE_RM2_.npy \
  #--no_load \