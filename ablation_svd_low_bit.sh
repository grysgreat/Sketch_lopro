#!/bin/bash
sketch_dir=/rjs/ghyx/llm/codes/sketch_with_calib
data_base_dir=/rjs/ghyx/data
save_dir=/rjs/ghyx/data
lm_eval_dir=/rjs/ghyx/llm/codes/lm-evaluation-harness
ppl_eval_dir=/rjs/ghyx/llm/codes/gptq
ft_dir=/rjs/ghyx/llm/codes/RILQ

declare -a low_bits=(8 16)
declare -a qbit_s=(2)
declare -a lmethods=("sketch" "svd")
#  "opt-125m" "opt-1.3b" "opt-2.7b" "opt-6.7b" 
#"opt-125m" "opt-1.3b" 
#declare -a models=("llemma_7b_safe")
#declare -a models=("opt-125m")
#declare -a models=("bloom-560m" "bloom-1b7" "bloom-7b1")
#declare -a models=("llemma_7b_safe")
declare -a models=("Llama-2-7b" "Llama-2-13b")
#declare -a models=("Llama-2-7b")
#declare -a models=("Qwen3-0.6B")
# declare -a lora_bit=(16)
declare -a rank=(16)

divider="------------------------------------------------------------------------"


method=gptq
# # # 遍历数组中的每一个模型名称
for bit in "${qbit_s[@]}"; do
  for testmodle in "${models[@]}"; do
    echo "$divider"
    echo ">>>>>>>>Starting model: $testmodle" 
    echo "$divider"
    for low_bit in "${low_bits[@]}"; do
      for lmthod in "${lmethods[@]}"; do
        echo ">>>>>>>> model: $testmodle; rank = $rank, bit = $bit, method = $method, lora_bit=$low_bit, lmethod=$lmthod" 
        echo ">>>>>>>> bit = $bit"
        echo "begin time: $(date '+%Y-%m-%d %H:%M:%S')"
        python $sketch_dir/run_quantize.py \
            --model_path "$data_base_dir/$testmodle" \
            --output_path "$save_dir/${testmodle}-calib" \
            --qbit "$bit" \
            --groupsize "128" \
            --fix_rank $rank  \
            --lora_bit $low_bit \
            --method $method \
            --loratool $lmthod
            # --save_lora
        echo "end time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ">>>>>>>>Starting test ppl: $testmodle, lora_bit:$low_bit, fix_rank:$rank, method:$method"
        CUDA_VISIBLE_DEVICES=0 python $ppl_eval_dir/llama.py "$save_dir/${testmodle}-calib" "wikitext2"

        echo ">>>>>>>>Starting zero shot task: $testmodle, lora_bit:$low_bit, fix_rank:$rank, method:$method"
        HF_DATASETS_CACHE="/root/.cache/huggingface/datasets" lm_eval --model hf  --model_args "pretrained=$data_base_dir/${testmodle}-calib"  --tasks arc_challenge,arc_easy,winogrande,piqa

        echo ">>>>>>>>Finish test: $testmodle, ratio:$ratio, delete files!"
        rm -rf $data_base_dir/${testmodle}-calib
      done
    done
  done
done


method=gptvq
# # # 遍历数组中的每一个模型名称
for bit in "${qbit_s[@]}"; do
  for testmodle in "${models[@]}"; do
    echo "$divider"
    echo ">>>>>>>>Starting model: $testmodle" 
    echo "$divider"
    for low_bit in "${low_bits[@]}"; do
      for lmthod in "${lmethods[@]}"; do
        echo ">>>>>>>> model: $testmodle; rank = $rank, bit = $bit, method = $method, lora_bit=$low_bit, lmethod=$lmthod" 
        echo ">>>>>>>> bit = $bit"
        echo "begin time: $(date '+%Y-%m-%d %H:%M:%S')"
        python $sketch_dir/run_quantize.py \
            --model_path "$data_base_dir/$testmodle" \
            --output_path "$save_dir/${testmodle}-calib" \
            --qbit "$bit" \
            --groupsize "128" \
            --fix_rank $rank  \
            --lora_bit $low_bit \
            --method $method \
            --loratool $lmthod
            # --save_lora
        echo "end time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ">>>>>>>>Starting test ppl: $testmodle, lora_bit:$low_bit, fix_rank:$rank, method:$method, lmethod=$lmthod"
        CUDA_VISIBLE_DEVICES=0 python $ppl_eval_dir/llama.py "$save_dir/${testmodle}-calib" "wikitext2"

        echo ">>>>>>>>Starting zero shot task: $testmodle, lora_bit:$low_bit, fix_rank:$rank, method:$method, lmethod=$lmthod"
        HF_DATASETS_CACHE="/root/.cache/huggingface/datasets" lm_eval --model hf  --model_args "pretrained=$data_base_dir/${testmodle}-calib"  --tasks arc_challenge,arc_easy,winogrande,piqa

        echo ">>>>>>>>Finish test: $testmodle, ratio:$ratio, delete files!"
        rm -rf $data_base_dir/${testmodle}-calib
      done
    done
  done
done
