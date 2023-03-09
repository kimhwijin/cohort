from yacs.config import CfgNode as CN


def get_default_config():
    _C = CN()
    # environment
    _C.device = 'cpu'

    # data loader
    _C.num_workers = 0
    _C.batch_size = 64

    # optimizer
    _C.lr = 1e-4

    # train
    _C.kfold = 5
    _C.epochs = 1
    _C.checkpoint_path = './checkpoint/base.pth'
    _C.seed = 1004
    
    # model
    _C.image_size = (224, 224)
    _C.batch_norm = True
    _C.dropout_rate = 0.5
    
    _C.backbone_name = 'resnet18'

    # metrics
    _C.threshold = 0.5

    # features
    _C.NUMERIC_FEATURES = ['OT_final', 
                    'UCVA_final', 
                    'CVA_final', 
                    'SPH_final', 
                    'CYL_final', 
                    'AXIS_final', 
                    'K1_final',
                    'K2_final',
                    'AXL_final',
                    'PupilSize2_final',
                    'PupilSize1_final',
                    'WTW_final',
                    'PACHY_final',
                    'age_at_surgery'
                    ]
    _C.CATEGORICAL_FEATURES = ['sex', 'surgery_type']
    _C.ONE_HOT_CAT_FEATURES = [
        ['sex_M', 'sex_F'],
        ['surgery_type_lasek', 'surgery_type_lasik', 'surgery_type_smile']
        ]
    
    return _C


