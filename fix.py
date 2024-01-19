import argparse
import importlib
import torch
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt


if __name__ == "__main__":
    #TODO this gotta be replaced
    #checkpoint_path = 'epicrealism_naturalSinRC1VAE.safetensors'
    checkpoint_path = 'absolutereality_v181.safetensors'
    original_config_file = None
    config_files = None
    num_in_channels = None
    scheduler_type = "pndm"
    pipeline_type = None
    image_size = None
    prediction_type = None
    #store_true
    extract_ema = False
    #store_true
    upcast_attention = False
    from_safetensors = True
    #store_true
    to_safetensors = False 
    #TODO this gotta be replaced
    dump_path = 'future'
    device = 'cuda:0'
    stable_unclip = None
    stable_unclip_prior = None
    clip_stats_path = None
    #Set flag if this is a controlnet checkpoint
    controlnet = None
    #store_true
    half = False
    vae_path = None
    pipeline_class_name = None
    pipeline_class = None

    pipe = download_from_original_stable_diffusion_ckpt(
    checkpoint_path_or_dict=checkpoint_path,
    original_config_file=original_config_file,
    config_files=config_files,
    image_size=image_size,
    prediction_type=prediction_type,
    model_type=pipeline_type,
    extract_ema=extract_ema,
    scheduler_type=scheduler_type,
    num_in_channels=num_in_channels,
    upcast_attention=upcast_attention,
    from_safetensors=from_safetensors,
    device=device,
    stable_unclip=stable_unclip,
    stable_unclip_prior=stable_unclip_prior,
    clip_stats_path=clip_stats_path,
    controlnet=controlnet,
    vae_path=vae_path,
    pipeline_class=pipeline_class,
    )
    pipe.save_pretrained(dump_path, safe_serialization=to_safetensors)
"""
if args.half:
    pipe.to(torch_dtype=torch.float16)

if args.controlnet:
    # only save the controlnet model
    pipe.controlnet.save_pretrained(args.dump_path, safe_serialization=args.to_safetensors)
else:
    pipe.save_pretrained(args.dump_path, safe_serialization=args.to_safetensors)
"""
