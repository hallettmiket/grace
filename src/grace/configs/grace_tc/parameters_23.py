
_base_=[ 'config_2_nopre.py' ]

total_epochs = 3000

optimizer = dict(lr=0.01, momentum=0.99, weight_decay=0.0001, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))

load_from = '/home/data/refined/candescence/production/models/candescence_version_1.0/model.pth'
