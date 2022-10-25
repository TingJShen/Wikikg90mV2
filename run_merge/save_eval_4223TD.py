from functools import total_ordering
import numpy as np
from tqdm import tqdm
import random
import torch
from ogb.lsc import WikiKG90Mv2Evaluator

evaluator = WikiKG90Mv2Evaluator()
def MRR(vat,vat10):
    ans=0
    for i in range(vat10.shape[0]):
        for j in range(vat10.shape[1]):
            if vat10[i][j]==vat[i]:
                #print(j)
                ans+=float(1)/(j+1)
                break
    return float(ans)/vat10.shape[0]
def MRRC(vat,candidate,vat10):
    ans=0
    for i in tqdm(range(vat10.shape[0])):
        for j in range(vat10.shape[1]):
            if candidate[i][vat10[i][j]]==vat[i]:
                #print(j)
                ans+=float(1)/(j+1)
                break
    return float(ans)/vat10.shape[0]
if __name__ == '__main__':
    candidate = np.load("/train14/sli/jswang19/OGB/wikikg90m-v2/merge_candidate/TD_PIE6w_RM32.npy")
    #TransE_849 = torch.load("/train14/sli/jswang19/OGB/wikikg90m-v2/eval-original/standard/TransE_849/valid_0_2.pkl")
    #ComplEx_299 = torch.load("/train14/sli/jswang19/OGB/wikikg90m-v2/eval-original/standard/ComplEx_299/valid_0_2.pkl")
    #TransE_199 = torch.load("/train14/sli/jswang19/OGB/wikikg90m-v2/eval-original/standard/TransE_199/valid_0_2.pkl")
    val_t=np.load('/train14/sli/jswang19/OGB/wikikg90m-v2/processed/val_t.npy')
    #print(MRRC(TransE_849['h,r->t']['t'].numpy(),candidate,TransE_849['h,r->t']['t_pred_top10'].numpy()))
    #TransE_849_s=TransE_849['h,r->t']['t_pred_score'].numpy()
    #ComplEx_299_s=ComplEx_299['h,r->t']['t_pred_score'].numpy()
    #TransE_199_s=TransE_199['h,r->t']['t_pred_score'].numpy()
    TransE_849_s=torch.load('/yrfs1/intern/tjshen/PIE-eval/Tr_8m4_sh_TD_PIE6w_RM32_RM/TransE_l2_wikikg90m_shallow_d_200_g_10.05/test_0_2.pkl')['h,r->t']['t_pred_score'].numpy()
    ComplEx_299_s=torch.load('/yrfs1/intern/tjshen/PIE-eval/Cp_2m9_sh_TD_PIE6w_RM32_off/ComplEx_wikikg90m_shallow_d_200_g_10.00/test_0_2.pkl')['h,r->t']['t_pred_score'].numpy()
    TransE_199_s=torch.load('/yrfs1/intern/tjshen/PIE-eval/Tr_1m9_sh_TD_PIE6w_RM32_off/TransE_l2_wikikg90m_shallow_d_200_g_10.00/test_0_2.pkl')['h,r->t']['t_pred_score'].numpy()
    RotatE_399_s=np.load('/yrfs1/intern/tjshen/PIE-eval/Rt_3m9_sh_TD_PIE6w_RM32_off/RotatE_wikikg90m_shallow_d_200_g_12.00/TD_RotatE_399_s.npy')
    SimplE_99_s=torch.load('/yrfs1/intern/tjshen/PIE-eval/Sp_1m_sh_TD_PIE6w_RM32_RM/SimplE_wikikg90m_shallow_d_400_g_10.00/test_0_2.pkl')['h,r->t']['t_pred_score'].numpy()
    multi_s=1*TransE_849_s+0.42*ComplEx_299_s+0.23*TransE_199_s+0.75*RotatE_399_s+0.25*SimplE_99_s
    w_top10=[]
    for t_index in tqdm(range(len(val_t))):
        a=np.array(multi_s[t_index]).argsort()[-10:][::-1]
        for eacha in range(len(a)):
            a[eacha]=candidate[t_index][a[eacha]]
            if a[eacha]==-1:
                test_idx=0
                a[eacha]=random.randint(0,10000000)
                while test_idx<eacha:
                    if a[test_idx]==a[eacha]:
                        a[eacha]=random.randint(0,10000000)
                        test_idx=0
                        continue
                    test_idx+=1
        w_top10.append(a)
    w_top10=np.array(w_top10)
    ans=MRR(val_t,w_top10)
    print(ans)
    input_dict = {}
    input_dict['h,r->t'] = {'t_pred_top10': w_top10}
    np.save('/yrfs1/intern/tjshen/OGB/TD_multi_s100_42_23_75_25.npy',multi_s)
    np.save('/yrfs1/intern/tjshen/OGB/w_top10.npy',w_top10)
    evaluator.save_test_submission(input_dict = input_dict, dir_path = '/train14/sli/jswang19/OGB/tjshen/final/', mode = 'test-dev')

            
            
