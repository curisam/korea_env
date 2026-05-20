#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH

# ---- 커널 안전 모드
export HF_USE_FLASH_ATTENTION=0
export XFORMERS_DISABLED=1

# ---- HF Hub 완전 로컬
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# ---- NCCL/디버깅
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
export TORCH_SHOW_CPP_STACKTRACES=1

# ---- CUDA allocator
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# ---- CPU 스레드 억제
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ---- 실행 옵션
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_MODULE_LOADING=EAGER
export FS_LOG_SUMMARY_ALL_RANKS=1
export TRANSFORMERS_VERBOSITY=error

# ==========================
# for-loop over rounds
# ==========================
# for R in 25 50 75 100 125 150 175 200 225 250
for R in 225 200 175 150 125 100 75 50 25         

do
  echo "=============================="
  echo " Running round ${R}"
  echo "=============================="

  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  accelerate launch \
    --config_file fedbiscuit_script/accelerator_config_bf16_ver1.yaml \
    --main_process_port 29500 \
    federatedscope/main.py \
    --cfg fedbiscuit_script/tldr/finetune_centralize.yaml \
    train.local_update_steps 80 \
    eval.outdir "exp/tldr/choice_qwen/pfl_anal_round/aggregation_analysis_global_1.0/round_${R}/raw" \
    outdir "exp/tldr/choice_qwen/pfl_anal_round/aggregation_analysis_global_1.0/round_${R}/" \
    model.load_from_local_pretrained_model_path "checkpoints_1.0_oracle/aggregation_analysis_one_adapter/final_tldr_choice_qwen_aggregation_analysis_u4_one_adapter_round_${R}.ckpt" \
    llm.accelerator.use True
done


    

