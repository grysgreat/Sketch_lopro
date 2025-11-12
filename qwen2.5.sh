#!/bin/bash
sketch_dir=/home/guhongyaoxing/llm/vtt/Sketch_lopro
data_base_dir=/data/ghyx
save_dir=/data/ghyx
lm_eval_dir=/home/guhongyaoxing/llm/vtt//lm-evaluation-harness
ppl_eval_dir=/home/guhongyaoxing/llm/vtt/gptq
ft_dir=/home/guhongyaoxing/llm/vtt/RILQ
qbit=3
low_bit=8
method=gptq
declare -a qbit_s=(2)
#  "opt-125m" "opt-1.3b" "opt-2.7b" "opt-6.7b" 
#"opt-125m" "opt-1.3b" 
#declare -a models=("llemma_7b_safe")
#declare -a models=("opt-125m")
#declare -a models=("bloom-560m" "bloom-1b7" "bloom-7b1")
#declare -a models=("llemma_7b_safe")
#declare -a models=("Llama-3-8B")
declare -a models=("Qwen2.5-7B-Instruct")
#declare -a models=("Qwen3-0.6B")
declare -a lora_bit=(8)
declare -a fix_ranks=(16 32)

divider="------------------------------------------------------------------------"

# # # 遍历数组中的每一个模型名称
for bit in "${qbit_s[@]}"; do
  for testmodle in "${models[@]}"; do
    echo "$divider"
    echo ">>>>>>>>Starting model: $testmodle" 
    echo "$divider"
    for rank in "${fix_ranks[@]}"; do
      echo ">>>>>>>> model: $testmodle; rank = $rank, bit = $bit, method = $method, lora_bit=$lora_bit" 
      echo ">>>>>>>> bit = $bit"
      export HF_ENDPOINT=https://hf-mirror.com 
      python $sketch_dir/run_quantize.py \
          --model_path "$data_base_dir/$testmodle" \
          --output_path "$save_dir/${testmodle}-calib" \
          --qbit "$bit" \
          --groupsize "128" \
          --fix_rank $rank  \
          --lora_bit $low_bit \
          --method $method
          # --save_lora
      echo ">>>>>>>>Starting test ppl: $testmodle, lora_bit:$low_bit, fix_rank:$rank, method:$method"
      CUDA_VISIBLE_DEVICES=0 python $ppl_eval_dir/llama.py "$save_dir/${testmodle}-calib" "wikitext2"

      export HF_ENDPOINT=https://hf-mirror.com 
      echo ">>>>>>>>Starting zero shot task: $testmodle, lora_bit:$low_bit, fix_rank:$rank, method:$method"
      lm_eval --model hf  --model_args "pretrained=$data_base_dir/${testmodle}"  --tasks arc_challenge,arc_easy,winogrande,openbookqa

      echo ">>>>>>>>Finish test: $testmodle, ratio:$ratio, delete files!"
      rm -rf $data_base_dir/${testmodle}-calib
    done
  done
done




