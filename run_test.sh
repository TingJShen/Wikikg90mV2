SAVE_PATH=/home/hcgu/PIE_results/OTE_wikikg90m_shallow_d_200_g_10.04/
DATA_PATH=/data/wikikg90m-v2/
VAL_CANDIDATE_PATH=/data/wikikg90m-v2/wikikg90m-v2/processed/candidate/val_candidate_official.npy
TEST_CANDIDATE_PATH=/data/wikikg90m-v2/wikikg90m-v2/processed/candidate/test_candidate_official.npy

NUM_PROC=2

python save_test_submission.py $SAVE_PATH $NUM_PROC "test-dev" False $VAL_CANDIDATE_PATH $TEST_CANDIDATE_PATH

