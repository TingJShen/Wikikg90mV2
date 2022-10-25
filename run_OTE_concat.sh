SAVE_PATH=/yrfs1/intern/tjshen/PIE-model/OTE_1kw_co_PIE_off/
DATA_PATH=/yrfs1/intern/tjshen/OGB/wikikg90m_v1/
#source /home/intern/tjshen/.bashrc
#conda activate PIEOTE-py37
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 dglke_train --model_name OTE \
  --hidden_dim 200 --gamma 10  --valid --test  -adv --mix_cpu_gpu --num_proc 8 --num_thread 8 \
  --gpu 0 1 2 3 4 5 6 7 \
  --async_update --ote_size 40 \
  --print_on_screen --encoder_model_name concat --save_path $SAVE_PATH \
  --data_path $DATA_PATH \
  --neg_sample_size 8192 --batch_size 8192 --lr 0.1 --regularization_coef 1.0e-9 \
  --max_step 100000000 --force_sync_interval 1000 --eval_interval 20000 --save_step 500000 \
  --val_candidate_path /train14/sli/jswang19/OGB/wikikg90m-v2/merge_candidate/PIE12w_RM32.npy \
  --trainhrt_path /train14/sli/jswang19/OGB/wikikg90m-v2/processed/train_hrt_origin.npy \
  --no_load \
  # --no_save_emb \
  # --LRE --LRE_rank 200
