_base_ = '../../_base_/default_runtime.py'

# ═══════════════════════════════════════════════════════════════════════════
#  Kabaddi ST-GCN  —  12 classes  —  COCO 17-keypoint  —  RTX 2050 4GB
# ═══════════════════════════════════════════════════════════════════════════
#  Class mapping (sorted folder names):
#   0: anklehold    1: backkick    2: block      3: chaintackle
#   4: dash         5: dubki       6: handtouch  7: lion_jump
#   8: sidekick     9: thighhold  10: toetouch  11: waist_hold
# ═══════════════════════════════════════════════════════════════════════════

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        graph_cfg=dict(layout='coco', mode='stgcn_spatial')),
    cls_head=dict(type='GCNHead', num_classes=12, in_channels=256))

dataset_type  = 'PoseDataset'
ann_file_train = 'data/kabaddi/kabaddi_train.pkl'
ann_file_val   = 'data/kabaddi/kabaddi_val.pkl'

train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs'),
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs'),
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=100, num_clips=10, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs'),
]

# ── Dataloaders  (batch=4 → safe for 4 GB VRAM) ─────────────────────────────
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=10,                        # repeat to get more iterations/epoch
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file_train,
            pipeline=train_pipeline)))

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator  = [dict(type='AccMetric')]
test_evaluator = val_evaluator

# ── Training loop ────────────────────────────────────────────────────────────
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=80,
    val_begin=1,
    val_interval=1)
val_cfg  = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ── LR schedule ──────────────────────────────────────────────────────────────
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-5,
        T_max=80,
        by_epoch=True,
        convert_to_iter_based=True)
]

# ── Optimizer ────────────────────────────────────────────────────────────────
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.05,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True))

# ── Hooks: checkpoint every 5 epochs, keep best + latest ────────────────────
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        save_best='auto',
        max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))

# ── Output dir ───────────────────────────────────────────────────────────────
work_dir = 'work_dirs/stgcn_kabaddi'

auto_scale_lr = dict(enable=False, base_batch_size=128)
