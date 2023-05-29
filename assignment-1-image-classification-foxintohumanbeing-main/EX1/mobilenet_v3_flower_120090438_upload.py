from mmcls.utils import auto_select_device
_base_ = [
    '_base_/models/mobilenet_v3_large_imagenet.py',
    '_base_/datasets/imagenet_bs32_pil_resize.py',
    '_base_/default_runtime.py'
]
deivce = auto_select_device()
# define the model
model = dict(
    backbone = dict(
        init_cfg = dict(
           type = 'Pretrained',
           checkpoint = './configs/mobilenet_v3_large-3ea3c186.pth',
           prefix = 'backbone'
        )
    ),
    head = dict(num_classes = 5,topk = (1,)),
)
# define the dataset config
dataset_type = 'CustomDataset'
classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=256),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu = 32,
    workers_per_gpu = 1,
    train = dict(data_prefix = 'C:/Users/mayn/Desktop/assignment-1-image-classification-foxintohumanbeing/EX1/flower_dataset/train',ann_file = 'C:/Users/mayn/Desktop/assignment-1-image-classification-foxintohumanbeing/EX1/flower_dataset/train.txt',classes = 'C:/Users/mayn/Desktop/assignment-1-image-classification-foxintohumanbeing/EX1/flower_dataset/classes.txt',pipeline=train_pipeline), 
    val = dict(data_prefix = 'C:/Users/mayn/Desktop/assignment-1-image-classification-foxintohumanbeing/EX1/flower_dataset/val',ann_file = 'C:/Users/mayn/Desktop/assignment-1-image-classification-foxintohumanbeing/EX1/flower_dataset/val.txt',classes = 'C:/Users/mayn/Desktop/assignment-1-image-classification-foxintohumanbeing/EX1/flower_dataset/classes.txt', pipeline=test_pipeline)
)


evaluation = dict(metric_options={'topk':(1,)})
# setting optimizer
optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

optimizer_config = dict(grad_clip = None)
# learning rate decay
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5)
runner = dict(type='EpochBasedRunner', max_epochs=20)
work_dir = './work_dirs/flower_dataset'
log_config = dict(interval=20)
from mmcls.apis import set_random_seed
seed = 0
set_random_seed(0, deterministic=True)

gpu_ids = range(1)