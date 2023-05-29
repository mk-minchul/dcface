
python unconditional_sampling.py \
      --attention_resolutions 16 \
      --class_cond False \
      --diffusion_steps 1000 \
      --num_samples 16 \
      --batch_size 8 \
      --image_size 256 \
      --learn_sigma True \
      --noise_schedule linear \
      --num_channels 128 \
      --num_head_channels 64 \
      --num_res_blocks 1 \
      --resblock_updown True \
      --use_fp16 False \
      --use_scale_shift_norm True \
      --timestep_respacing 100 \
      --down_N 32 \
      --range_t 20 \
      --save_dir unconditional_samples