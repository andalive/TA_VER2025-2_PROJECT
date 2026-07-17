_base_ = '../mmaction2/configs/_base_/default_runtime.py'

default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10000, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, save_best='auto', max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))


# load_from='../../../work_dirs/ciis_10_0lr2_1/epoch_24.pth'


      # model (:obj:`torch.nn.Module` or dict): The model to be run. It can be
        #     a dict used for build a model.
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
	# SlowOnly backbone based on ResNet3dPathway.
	# Args:
	#     conv1_kernel (Sequence[int]): Kernel size of the first conv layer.
	# 	Defaults to ``(1, 7, 7)``.
	#     conv1_stride_t (int): Temporal stride of the first conv layer.
	# 	Defaults to 1.
	#     pool1_stride_t (int): Temporal stride of the first pooling layer.
	# 	Defaults to 1.
	#     inflate (Sequence[int]): Inflate dims of each block.
	# 	Defaults to ``(0, 0, 1, 1)``.
	#     with_pool2 (bool): Whether to use pool2. Defaults to False.
        depth=50,
        pretrained=None,  # can be changed for ciis
        in_channels=17,
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(4, 6, 3),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2),
        dilations=(1, 1, 1)),
    cls_head=dict(
        type='I3DHead',
	# Classification head for I3D.
	# Args:
	#     num_classes (int): Number of classes to be classified.
	#     in_channels (int): Number of channels in input feature.
	#     loss_cls (dict or ConfigDict): Config for building loss.
	# 	Default: dict(type='CrossEntropyLoss', loss_weight=1.0)
	#     spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
	#     dropout_ratio (float): Probability of dropout layer. Default: 0.5.
	#     init_std (float): Std value for Initiation. Default: 0.01.
	#     kwargs (dict, optional): Any keyword argument to be used to initialize
	# 	the head.	
        in_channels=512,
        num_classes=7,  # changed for ciis  # Number of classes to be classified.
        dropout_ratio=0.5,
        spatial_type='avg',
        loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
        
        # BaseHead Args: Base Classification head
        average_clips='prob'
        # multi_class=False  # multi_class (bool): Determines whether it is a multi-class
			      # recognition dataset. Defaults to False.
        ))


        # work_dir (str): The working directory to save checkpoints. The logs
        #     will be saved in the subdirectory of `work_dir` named
        #     :attr:`timestamp`.
# work_dir = ""

dataset_type = 'PoseDataset'
ann_file = '03_dataset/034_fin_dataset/<sesuaikan>.pkl'  # changed for ciis
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=4),  # changed for ciis  # To sample an n-frame clip from the video. UniformSampleFrames basically divide the video into n segments of equal length and randomly sample one frame from each segment. To make the testing results reproducible, a random seed is set during testing, to make the sampling results deterministic.
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,  # The sigma of the generated gaussian map. Default: 0.6.
        use_score=True,  # Use the confidence score of keypoints as the maximum of the gaussian maps. Default: True.
        with_kp=True,  # Generate pseudo heatmaps for keypoints. Default: True.
        with_limb=False),  # Generate pseudo heatmaps for limbs. At least one of 'with_kp' and 'with_limb' should be True. Default: False.
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),  # 'NCTHW', 'NCHW', 'NCTHW_Heatmap', 'NPTCHW'
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=4, num_clips=1, test_mode=True),  # changed for ciis
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=4, num_clips=1, test_mode=True),  # changed for ciis
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False,
        double=True,
        left_kp=left_kp,
        right_kp=right_kp),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='PackActionInputs')
]


        # train_dataloader (Dataloader or dict, optional): A dataloader object or
        #     a dict to build a dataloader. If ``None`` is given, it means
        #     skipping training steps. Defaults to None.
        #     See :meth:`build_dataloader` for more details.
train_dataloader = dict(
    batch_size=128,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),  #sesuaikan shuffle Train (sebaiknya True)
    # dataset=dict(
        # type='RepeatDataset',
        # times=10,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            split='xsub_train',
            pipeline=train_pipeline))#)


        # val_dataloader (Dataloader or dict, optional): A dataloader object or
        #     a dict to build a dataloader. If ``None`` is given, it means
        #     skipping validation steps. Defaults to None.
        #     See :meth:`build_dataloader` for more details.
val_dataloader = dict(
    batch_size=128,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        split='xsub_val',
        pipeline=val_pipeline,
        test_mode=True))


        # test_dataloader (Dataloader or dict, optional): A dataloader object or
        #     a dict to build a dataloader. If ``None`` is given, it means
        #     skipping test steps. Defaults to None.
        #     See :meth:`build_dataloader` for more details.
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        split='xsub_val',
        pipeline=test_pipeline,
        test_mode=True))


        # val_evaluator (Evaluator or dict or list, optional): A evaluator object
        #     used for computing metrics for validation. It can be a dict or a
        #     list of dict to build a evaluator. If specified,
        #     :attr:`val_dataloader` should also be specified. Defaults to None.
val_evaluator = [dict(type='AccMetric')]


        # test_evaluator (Evaluator or dict or list, optional): A evaluator
        #     object used for computing metrics for test steps. It can be a dict
        #     or a list of dict to build a evaluator. If specified,
        #     :attr:`test_dataloader` should also be specified. Defaults to None.
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=300, val_begin=10, val_interval=10)


        # val_cfg (dict, optional): A dict to build a validation loop. If it does
        #     not provide "type" key, :class:`ValLoop` will be used by default.
        #     If ``val_cfg`` specified, :attr:`val_dataloader` should also be
        #     specified. If ``ValLoop`` is built with `fp16=True``,
        #     ``runner.val()`` will be performed under fp16 precision.
        #     Defaults to None. See :meth:`build_val_loop` for more details.
val_cfg = dict(type='ValLoop')


        # test_cfg (dict, optional): A dict to build a test loop. If it does
        #     not provide "type" key, :class:`TestLoop` will be used by default.
        #     If ``test_cfg`` specified, :attr:`test_dataloader` should also be
        #     specified. If ``ValLoop`` is built with `fp16=True``,
        #     ``runner.val()`` will be performed under fp16 precision.
        #     Defaults to None. See :meth:`build_test_loop` for more details.
test_cfg = dict(type='TestLoop')


        # auto_scale_lr (dict, Optional): Config to scale the learning rate
        #     automatically. It includes ``base_batch_size`` and ``enable``.
        #     ``base_batch_size`` is the batch size that the optimizer lr is
        #     based on. ``enable`` is the switch to turn on and off the feature.
# `base_batch_size` = (8 GPUs) x (12 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=96)


        # param_scheduler (_ParamScheduler or dict or list, optional):
        #     Parameter scheduler for updating optimizer parameters. If
        #     specified, :attr:`optimizer` should also be specified.
        #     Defaults to None.
        #     See :meth:`build_param_scheduler` for examples.
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
	# Set the learning rate of each parameter group using a cosine annealing
	# schedule, where :math:`\eta_{max}` is set to the initial value and
	# :math:`T_{cur}` is the number of epochs since the last restart in SGDR.

	# Stochastic Gradient Descent with Warm Restarts (SGDR): https://arxiv.org/abs/1608.03983
	    
	# Note that this only implements the cosine annealing part of SGDR, and not
	# the restarts.
        eta_min=0.01,
        T_max=300,  # https://www.youtube.com/+watch?v=WgwBRqhdIrQ&t=468s
        by_epoch=True,
        convert_to_iter_based=True)
]


        # optim_wrapper (OptimWrapper or dict, optional):
        #     Computing gradient of model parameters. If specified,
        #     :attr:`train_dataloader` should also be specified. If automatic
        #     mixed precision or gradient accmulation
        #     training is required. The type of ``optim_wrapper`` should be
        #     AmpOptimizerWrapper. See :meth:`build_optim_wrapper` for
        #     examples. Defaults to None.
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.037, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))



        # data_preprocessor (dict, optional): The pre-process config of
        #     :class:`BaseDataPreprocessor`. If the ``model`` argument is a dict
        #     and doesn't contain the key ``data_preprocessor``, set the argument
        #     as the ``data_preprocessor`` of the ``model`` dict.
        #     Defaults to None.
# data_preprocessor = dict()


        # launcher (str): Way to launcher multi-process. Supported launchers
        #     are 'pytorch', 'mpi', 'slurm' and 'none'. If 'none' is provided,
        #     non-distributed environment will be launched.
# launcher = ""


        # randomness (dict): Some settings to make the experiment as reproducible
        #     as possible like seed and deterministic.
        #     Defaults to ``dict(seed=None)``. If seed is None, a random number
        #     will be generated and it will be broadcasted to all other processes
        #     if in distributed environment. If ``cudnn_benchmark`` is
        #     ``True`` in ``env_cfg`` but ``deterministic`` is ``True`` in
        #     ``randomness``, the value of ``torch.backends.cudnn.benchmark``
        #     will be ``False`` finally.
# randomness = dict()


        # experiment_name (str, optional): Name of current experiment. If not
        #     specified, timestamp will be used as ``experiment_name``.
        #     Defaults to None.
# experiment_name = ""


        # cfg (dict or Configdict or :obj:`Config`, optional): Full config.
        #     Defaults to None.
# cfg = dict()
