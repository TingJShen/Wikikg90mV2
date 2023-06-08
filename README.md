# Test-dev-Subs

### 1.using PIE  to generate 6w candidates

### 2.run shells  in ./run_TD to train 5 models

changing train_hrt to official train_hrt in wikikg90mV2

changing training steps to 10000000

### 3.run Merging Models

cd run_merge

python save_eval_4223TD.py

this will sum the scores saving by trainging by weight of

1*TransE_849_s+0.42*ComplEx_299_s+0.23*TransE_199_s+0.75*RotatE_399_s+0.25*SimplE_99_s

### 











#### Detailed steps will be suppled after by the reason of time.
