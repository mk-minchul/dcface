import os

def post_process(cfg):

    img_size = cfg.datamodule.img_size
    division = 1
    proj_img_size = int(img_size / division)

    cfg.trainer.image_size = proj_img_size
    cfg.model.unet_config.params.image_size = proj_img_size

    if 'casia' in cfg.datamodule.dataset_name or 'faces_webface_112x112':
        if cfg.model.cond_stage_config is not None:
            cfg.model.cond_stage_config.params.n_classes = 10572
    elif 'ffhq' in cfg.datamodule.dataset_name:
        # dummy
        if cfg.model.cond_stage_config is not None:
            cfg.model.cond_stage_config.params.n_classes = 10572
    else:
        raise ValueError('')

    if cfg.ckpt_path:
        cfg.ckpt_path = os.path.join(cfg.paths.repo_root, cfg.ckpt_path)
    if cfg.trainer.ckpt_path:
        cfg.trainer.ckpt_path = os.path.join(cfg.paths.repo_root, cfg.trainer.ckpt_path)

    return cfg