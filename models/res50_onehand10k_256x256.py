checkpoint_config = dict(interval=10)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'


evaluation = dict(interval=10, metric=['PCK', 'AUC', 'EPE'], save_best='AUC')

optimizer = dict(
    type='Adam',
    lr=5e-4,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
total_epochs = 210
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=21,
    dataset_joints=21,
    dataset_channel=[
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20
        ],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20
    ])

# model settings
model = dict(
    type='TopDown',
    pretrained='torchvision://resnet50',
    backbone=dict(type='ResNet', depth=50),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=2048,
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'])

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownRandomShiftBboxCenter', shift_factor=0.16, prob=0.3),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=90, scale_factor=0.3),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['image_file', 'center', 'scale', 'rotation', 'flip_pairs']),
]

test_pipeline = val_pipeline
dataset_info = dict(
    dataset_name='onehand10k',
    paper_info=dict(
        author='Wang, Yangang and Peng, Cong and Liu, Yebin',
        title='Mask-pose cascaded cnn for 2d hand pose estimation '
        'from single color image',
        container='IEEE Transactions on Circuits and Systems '
        'for Video Technology',
        year='2018',
        homepage='https://www.yangangwang.com/papers/WANG-MCC-2018-10.html',
    ),
    keypoint_info={
        0:
        dict(name='wrist', id=0, color=[255, 255, 255], type='', swap=''),
        1:
        dict(name='thumb1', id=1, color=[255, 128, 0], type='', swap=''),
        2:
        dict(name='thumb2', id=2, color=[255, 128, 0], type='', swap=''),
        3:
        dict(name='thumb3', id=3, color=[255, 128, 0], type='', swap=''),
        4:
        dict(name='thumb4', id=4, color=[255, 128, 0], type='', swap=''),
        5:
        dict(
            name='forefinger1', id=5, color=[255, 153, 255], type='', swap=''),
        6:
        dict(
            name='forefinger2', id=6, color=[255, 153, 255], type='', swap=''),
        7:
        dict(
            name='forefinger3', id=7, color=[255, 153, 255], type='', swap=''),
        8:
        dict(
            name='forefinger4', id=8, color=[255, 153, 255], type='', swap=''),
        9:
        dict(
            name='middle_finger1',
            id=9,
            color=[102, 178, 255],
            type='',
            swap=''),
        10:
        dict(
            name='middle_finger2',
            id=10,
            color=[102, 178, 255],
            type='',
            swap=''),
        11:
        dict(
            name='middle_finger3',
            id=11,
            color=[102, 178, 255],
            type='',
            swap=''),
        12:
        dict(
            name='middle_finger4',
            id=12,
            color=[102, 178, 255],
            type='',
            swap=''),
        13:
        dict(
            name='ring_finger1', id=13, color=[255, 51, 51], type='', swap=''),
        14:
        dict(
            name='ring_finger2', id=14, color=[255, 51, 51], type='', swap=''),
        15:
        dict(
            name='ring_finger3', id=15, color=[255, 51, 51], type='', swap=''),
        16:
        dict(
            name='ring_finger4', id=16, color=[255, 51, 51], type='', swap=''),
        17:
        dict(name='pinky_finger1', id=17, color=[0, 255, 0], type='', swap=''),
        18:
        dict(name='pinky_finger2', id=18, color=[0, 255, 0], type='', swap=''),
        19:
        dict(name='pinky_finger3', id=19, color=[0, 255, 0], type='', swap=''),
        20:
        dict(name='pinky_finger4', id=20, color=[0, 255, 0], type='', swap='')
    },
    skeleton_info={
        0:
        dict(link=('wrist', 'thumb1'), id=0, color=[255, 128, 0]),
        1:
        dict(link=('thumb1', 'thumb2'), id=1, color=[255, 128, 0]),
        2:
        dict(link=('thumb2', 'thumb3'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('thumb3', 'thumb4'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('wrist', 'forefinger1'), id=4, color=[255, 153, 255]),
        5:
        dict(link=('forefinger1', 'forefinger2'), id=5, color=[255, 153, 255]),
        6:
        dict(link=('forefinger2', 'forefinger3'), id=6, color=[255, 153, 255]),
        7:
        dict(link=('forefinger3', 'forefinger4'), id=7, color=[255, 153, 255]),
        8:
        dict(link=('wrist', 'middle_finger1'), id=8, color=[102, 178, 255]),
        9:
        dict(
            link=('middle_finger1', 'middle_finger2'),
            id=9,
            color=[102, 178, 255]),
        10:
        dict(
            link=('middle_finger2', 'middle_finger3'),
            id=10,
            color=[102, 178, 255]),
        11:
        dict(
            link=('middle_finger3', 'middle_finger4'),
            id=11,
            color=[102, 178, 255]),
        12:
        dict(link=('wrist', 'ring_finger1'), id=12, color=[255, 51, 51]),
        13:
        dict(
            link=('ring_finger1', 'ring_finger2'), id=13, color=[255, 51, 51]),
        14:
        dict(
            link=('ring_finger2', 'ring_finger3'), id=14, color=[255, 51, 51]),
        15:
        dict(
            link=('ring_finger3', 'ring_finger4'), id=15, color=[255, 51, 51]),
        16:
        dict(link=('wrist', 'pinky_finger1'), id=16, color=[0, 255, 0]),
        17:
        dict(
            link=('pinky_finger1', 'pinky_finger2'), id=17, color=[0, 255, 0]),
        18:
        dict(
            link=('pinky_finger2', 'pinky_finger3'), id=18, color=[0, 255, 0]),
        19:
        dict(
            link=('pinky_finger3', 'pinky_finger4'), id=19, color=[0, 255, 0])
    },
    joint_weights=[1.] * 21,
    sigmas=[])


data_root = 'data/onehand10k'
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='OneHand10KDataset',
        ann_file=f'{data_root}/annotations/onehand10k_train.json',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info=dataset_info),
    val=dict(
        type='OneHand10KDataset',
        ann_file=f'{data_root}/annotations/onehand10k_test.json',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info=dataset_info),
    test=dict(
        type='OneHand10KDataset',
        ann_file=f'{data_root}/annotations/onehand10k_test.json',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info=dataset_info),
)

