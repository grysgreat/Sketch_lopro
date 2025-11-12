#!/bin/bash
sketch_dir=/home/guhongyaoxing/llm/vtt/sketch_with_calib
data_base_dir=/data/ghyx
save_dir=/data/ghyx
lm_eval_dir=/home/guhongyaoxing/llm/vtt//lm-evaluation-harness
ppl_eval_dir=/home/guhongyaoxing/llm/vtt/gptq
ft_dir=/home/guhongyaoxing/llm/vtt/RILQ
method=gptq
declare -a qbit_s=(2)
#  "opt-125m" "opt-1.3b" "opt-2.7b" "opt-6.7b" 
#"opt-125m" "opt-1.3b" 
#declare -a models=("llemma_7b_safe")
#declare -a models=("opt-125m")
#declare -a models=("bloom-560m" "bloom-1b7" "bloom-7b1")
#declare -a models=("llemma_7b_safe")
#declare -a models=("Llama-3-8B")
declare -a models=("Mixtral-8x7B")
#declare -a models=("Qwen3-0.6B")
declare -a low_bit=(8)
declare -a fix_ranks=(16)

divider="------------------------------------------------------------------------"

# # # 遍历数组中的每一个模型名称
for bit in "${qbit_s[@]}"; do
  for testmodle in "${models[@]}"; do
    echo "$divider"
    echo ">>>>>>>>Starting model: $testmodle" 
    echo "$divider"
    for rank in "${fix_ranks[@]}"; do
      echo ">>>>>>>> model: $testmodle; rank = $rank, bit = $bit, method = $method, lora_bit=$low_bit" 
      echo ">>>>>>>> bit = $bit"
      export HF_ENDPOINT=https://hf-mirror.com 

      echo "begin time: $(date '+%Y-%m-%d %H:%M:%S')"
      python $sketch_dir/run_quantize.py \
          --model_path "$data_base_dir/$testmodle" \
          --output_path "$save_dir/${testmodle}-calib" \
          --qbit "$bit" \
          --groupsize "128" \
          --fix_rank $rank  \
          --lora_bit $low_bit \
          --method $method
          # --save_lora
      echo "end time: $(date '+%Y-%m-%d %H:%M:%S')"
      echo ">>>>>>>>Starting test ppl: $testmodle, lora_bit:$low_bit, fix_rank:$rank, method:$method"
      CUDA_VISIBLE_DEVICES=0 python $ppl_eval_dir/mixtural.py "$save_dir/${testmodle}-calib" "wikitext2"

      #accelerate launch -m 
      export HF_ENDPOINT=https://hf-mirror.com 
      echo ">>>>>>>>Starting zero shot task: $testmodle, lora_bit:$low_bit, fix_rank:$rank, method:$method"
      echo lm_eval --model hf  --model_args "pretrained=$data_base_dir/${testmodle}-calib,device_map=auto"  --tasks arc_challenge
      accelerate launch -m  lm_eval --model hf  --model_args "pretrained=$data_base_dir/${testmodle}-calib,device_map=auto,parallelize=True"  --tasks arc_challenge --batch_size auto

      echo lm_eval --model hf  --model_args "pretrained=$data_base_dir/${testmodle}-calib,device_map=auto"  --tasks arc_easy
      accelerate launch -m  lm_eval --model hf  --model_args "pretrained=$data_base_dir/${testmodle}-calib,device_map=auto,parallelize=True"  --tasks arc_easy --batch_size auto

      echo lm_eval --model hf  --model_args "pretrained=$data_base_dir/${testmodle}-calib,device_map=auto"  --tasks winogrande
      accelerate launch -m  lm_eval --model hf  --model_args "pretrained=$data_base_dir/${testmodle}-calib,device_map=auto,parallelize=True"  --tasks winogrande --batch_size auto


      echo lm_eval --model hf  --model_args "pretrained=$data_base_dir/${testmodle}-calib,device_map=auto"  --tasks piqa
      accelerate launch -m  lm_eval --model hf  --model_args "pretrained=$data_base_dir/${testmodle}-calib,device_map=auto,parallelize=True"  --tasks piqa --batch_size auto

      # export HF_ENDPOINT=https://hf-mirror.com 
      # echo ">>>>>>>>Starting running fine turning: $testmodle, lora_bit:$low_bit, fix_rank:$rank, method:$method"
      # sh $ft_dir/rilq-llama2_7b-r64.sh

      # export HF_ENDPOINT=https://hf-mirror.com 
      # echo ">>>>>>>>Starting test FT-ppl: $testmodle, lora_bit:$low_bit, fix_rank:$rank, method:$method"
      # CUDA_VISIBLE_DEVICES=0 python $ppl_eval_dir/llama_rilq.py "$save_dir/${testmodle}-calib" "wikitext2"

      # export HF_ENDPOINT=https://hf-mirror.com 
      # echo ">>>>>>>>Starting FT-zero shot task: $testmodle, lora_bit:$low_bit, fix_rank:$rank, method:$method"
      # lm_eval --model hf  --model_args "pretrained=$data_base_dir/${testmodle}-calib,peft=$data_base_dir/Llama-2-7b-calib-rilq-r64/approx_init"  --tasks arc_challenge,arc_easy,winogrande,openbookqa

      # echo ">>>>>>>>Finish test: $testmodle, ratio:$ratio, delete files!"
      # rm -rf $data_base_dir/${testmodle}-calib
      # rm -rf $data_base_dir/${testmodle}-calib-rilq-r64
    done
  done
done


