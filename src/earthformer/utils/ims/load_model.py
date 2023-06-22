import omegaconf as OmegaConf
from src.earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel


def load_model(model_cfg):
    # ---- compute fields that require arithmetic operations on config values  ---- #
    num_blocks = len(model_cfg["enc_depth"])
    if isinstance(model_cfg["self_pattern"], str):
        enc_attn_patterns = [model_cfg["self_pattern"]] * num_blocks
    else:
        enc_attn_patterns = OmegaConf.to_container(model_cfg["self_pattern"])
    if isinstance(model_cfg["cross_self_pattern"], str):
        dec_self_attn_patterns = [model_cfg["cross_self_pattern"]] * num_blocks
    else:
        dec_self_attn_patterns = OmegaConf.to_container(model_cfg["cross_self_pattern"])
    if isinstance(model_cfg["cross_pattern"], str):
        dec_cross_attn_patterns = [model_cfg["cross_pattern"]] * num_blocks
    else:
        dec_cross_attn_patterns = OmegaConf.to_container(model_cfg["cross_pattern"])
    return CuboidTransformerModel(
        # --------------------------- network arch/size --------------------------- #
        # --- general encoder/decoder configs
        enc_depth=model_cfg["enc_depth"],
        dec_depth=model_cfg["dec_depth"],
        enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
        dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
        # --- attention related
        enc_attn_patterns=enc_attn_patterns,
        dec_self_attn_patterns=dec_self_attn_patterns,
        dec_cross_attn_patterns=dec_cross_attn_patterns,
        dec_cross_last_n_frames=model_cfg["dec_cross_last_n_frames"],
        dec_use_first_self_attn=model_cfg["dec_use_first_self_attn"],
        num_heads=model_cfg["num_heads"],
        attn_drop=model_cfg["attn_drop"],
        dec_hierarchical_pos_embed=model_cfg["dec_hierarchical_pos_embed"],
        pos_embed_type=model_cfg["pos_embed_type"],
        use_relative_pos=model_cfg["use_relative_pos"],
        self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
        # --- in/out shape
        input_shape=[model_cfg["in_len"], *model_cfg["hwc"]],
        target_shape=[model_cfg["out_len"], *model_cfg["hwc"]],
        padding_type=model_cfg["padding_type"],
        base_units=model_cfg["base_units"],
        block_units=model_cfg["block_units"],
        scale_alpha=model_cfg["scale_alpha"],
        # --- voodoo stuff that hopefully helps so everyone do it
        proj_drop=model_cfg["proj_drop"],
        ffn_drop=model_cfg["ffn_drop"],
        upsample_type=model_cfg["upsample_type"],
        downsample=model_cfg["downsample"],
        downsample_type=model_cfg["downsample_type"],
        ffn_activation=model_cfg["ffn_activation"],
        gated_ffn=model_cfg["gated_ffn"],
        norm_layer=model_cfg["norm_layer"],
        # --- initial_downsample
        initial_downsample_type=model_cfg["initial_downsample_type"],
        initial_downsample_activation=model_cfg["initial_downsample_activation"],
        # these are relevant when (initial_downsample_type == "stack_conv")
        initial_downsample_stack_conv_num_layers=model_cfg["initial_downsample_stack_conv_num_layers"],
        initial_downsample_stack_conv_dim_list=model_cfg["initial_downsample_stack_conv_dim_list"],
        initial_downsample_stack_conv_downscale_list=model_cfg["initial_downsample_stack_conv_downscale_list"],
        initial_downsample_stack_conv_num_conv_list=model_cfg["initial_downsample_stack_conv_num_conv_list"],
        # --- global vectors
        num_global_vectors=model_cfg["num_global_vectors"],
        use_dec_self_global=model_cfg["use_dec_self_global"],
        dec_self_update_global=model_cfg["dec_self_update_global"],
        use_dec_cross_global=model_cfg["use_dec_cross_global"],
        use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
        use_global_self_attn=model_cfg["use_global_self_attn"],
        separate_global_qkv=model_cfg["separate_global_qkv"],
        global_dim_ratio=model_cfg["global_dim_ratio"],
        # ----------------------------- initialization ---------------------------- #
        attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
        ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
        conv_init_mode=model_cfg["conv_init_mode"],
        down_up_linear_init_mode=model_cfg["down_up_linear_init_mode"],
        norm_init_mode=model_cfg["norm_init_mode"],
        # ----------------------------------- misc -------------------------------- #
        z_init_method=model_cfg["z_init_method"],
        checkpoint_level=model_cfg["checkpoint_level"],
    )
