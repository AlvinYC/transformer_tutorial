# transformer_tutorial (self-attention)
  - modified w/ https://github.com/jadore801120/attention-is-all-you-need-pytorch

# train model
  - python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 256 -warmup 128000 -epoch 400
  - vs_code args: \
  "args": ["-data_pkl", "m30k_deen_shr.pkl", "-log", " m30k_deen_shr", "-embs_share_weight", "-proj_share_weight", "-label_smoothing", "-save_model", "trained", "-b", "256", "-warmup", "128000", "-epoch", "400"]

# plot curve
  - python bokeh_plot_line.py\
  do the following setting
  ```python
    train_path = '/home/alvin/git_repository/transformer_tutorial/log/epoch400/m30k_deen_shr.train.log'
    valid_path = '/home/alvin/git_repository/transformer_tutorial/log/epoch400/m30k_deen_shr.valid.log'
    target_field = 'Accuracy'   # set target filed to be y-axis, need one of fileds which get_example() setup
  ```
  ![bokeh epoch 400](https://i.imgur.com/T6vNU0e.jpg|width=400px)
