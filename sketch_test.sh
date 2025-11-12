#!/bin/bash
sketch_dir=/rjs/ghyx/llm/codes/sketch_with_calib
data_base_dir=/rjs/ghyx/data
save_dir=/rjs/ghyx/data
lm_eval_dir=/rjs/ghyx/llm/codes/lm-evaluation-harness
ppl_eval_dir=/rjs/ghyx/llm/codes/gptq
qbit=3
declare -a qbit_s=(2)
#  "opt-125m" "opt-1.3b" "opt-2.7b" "opt-6.7b" 
#"opt-125m" "opt-1.3b" 
#declare -a models=("llemma_7b_safe")
#declare -a models=("opt-125m")
#declare -a models=("bloom-560m" "bloom-1b7" "bloom-7b1")
#declare -a models=("llemma_7b_safe")
#declare -a models=("Llama-3-8B")
declare -a models=("Llama-2-7b")
#declare -a models=("Qwen3-0.6B")
declare -a ratios=(0.2)
declare -a fix_ranks=(32 128)

divider="------------------------------------------------------------------------"

# # # 遍历数组中的每一个模型名称
for bit in "${qbit_s[@]}"; do
  for testmodle in "${models[@]}"; do
    echo "$divider"
    echo ">>>>>>>>Starting model: $testmodle"
    echo "$divider"
    for ratio in "${ratios[@]}"; do
      # echo ">>>>>>>> ratio = $ratio, bit = $bit"
      # python $sketch_dir/run_quantize.py \
      #     --model_path "$data_base_dir/$testmodle" \
      #     --output_path "$save_dir/${testmodle}-calib" \
      #     --qbit "$bit" \
      #     --groupsize "128" \
      #     --lora_ratio $ratio \
      #     --fix_rank 0  \
      #     --lora_bit 16 \
      #     # --method "gptq"
      #     # --save_lora
      # echo ">>>>>>>>Starting test ppl: $testmodle, ratio:$ratio"
      # CUDA_VISIBLE_DEVICES=0 python $ppl_eval_dir/llama.py "$save_dir/${testmodle}-calib" "wikitext2"
      #CUDA_VISIBLE_DEVICES=0 python $ppl_eval_dir/llama.py "$save_dir/${testmodle}-calib" "wikitext2"
      echo ">>>>>>>>Starting zero shot task: $testmodle, ratio:$ratio"
      lm_eval --model hf  --model_args "pretrained=$data_base_dir/${testmodle}-calib"  --tasks mmlu
      

      # echo ">>>>>>>>Starting zero shot task: $ , ratio:$ratio"
      # lm_eval --model hf  --model_args "pretrained=$data_base_dir/${testmodle}-calib"  --tasks gsm8k


      # echo ">>>>>>>>Finish test: $testmodle, ratio:$ratio, delete files!"
      # rm -rf $data_base_dir/${testmodle}-calib

    done
  done
done


