#!/bin/bash
sketch_dir=/rjs/ghyx/llm/codes/sketch_with_calib
data_base_dir=/rjs/ghyx/data
save_dir=/rjs/ghyx/data
lm_eval_dir=/rjs/ghyx/llm/codes/lm-evaluation-harness
ppl_eval_dir=/rjs/ghyx/llm/codes/gptq
ft_dir=/rjs/ghyx/llm/codes/RILQ
low_bit=8
method=gptq
declare -a qbit_s=(2)
#  "opt-125m" "opt-1.3b" "opt-2.7b" "opt-6.7b" 
#"opt-125m" "opt-1.3b" 
#declare -a models=("llemma_7b_safe")
#declare -a models=("opt-125m")
#declare -a models=("bloom-560m" "bloom-1b7" "bloom-7b1")
#declare -a models=("llemma_7b_safe")
declare -a models=("Llama-2-7b")
#declare -a models=("Llama-2-7b")
#declare -a models=("Qwen3-0.6B")
# declare -a lora_bit=(16)
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
      python $sketch_dir/run_quantize.py \
          --model_path "$data_base_dir/$testmodle" \
          --output_path "$save_dir/${testmodle}-calib2" \
          --qbit "$bit" \
          --groupsize "128" \
          --fix_rank $rank  \
          --lora_bit $low_bit \
          --method $method \
          --id_bsize 256 \
          --ha_bsize 256 \
          # --save_lora
      echo ">>>>>>>>Starting test ppl: $testmodle, lora_bit:$low_bit, fix_rank:$rank, method:$method"
      CUDA_VISIBLE_DEVICES=0 python $ppl_eval_dir/llama_waquant.py "$save_dir/${testmodle}-calib2" "wikitext2"

      # echo ">>>>>>>>Starting zero shot task: $testmodle, lora_bit:$low_bit, fix_rank:$rank, method:$method"
      # HF_DATASETS_CACHE="/root/.cache/huggingface/datasets" lm_eval --model hf  --model_args "pretrained=$data_base_dir/${testmodle}-calib"  --tasks arc_challenge,arc_easy,winogrande,piqa


      # echo ">>>>>>>>Starting running fine turning: $testmodle, lora_bit:$low_bit, fix_rank:$rank, method:$method"
      # sh $ft_dir/rilq-llama2_7b-r64.sh

      
      # echo ">>>>>>>>Starting test FT-ppl: $testmodle, lora_bit:$low_bit, fix_rank:$rank, method:$method"
      # CUDA_VISIBLE_DEVICES=0 python $ppl_eval_dir/llama_rilq.py "$save_dir/${testmodle}-calib" "wikitext2"

      # echo ">>>>>>>>Starting FT-zero shot task: $testmodle, lora_bit:$low_bit, fix_rank:$rank, method:$method"
      # lm_eval --model hf  --model_args "pretrained=$data_base_dir/${testmodle}-calib,peft=$data_base_dir/${testmodle}-calib-rilq-r64/approx_init"  --tasks arc_challenge,arc_easy,winogrande,piqa

      # echo ">>>>>>>>Finish test: $testmodle, ratio:$ratio, delete files!"
      # rm -rf $data_base_dir/${testmodle}-calib
      # rm -rf $data_base_dir/${testmodle}-calib-rilq-r64
    done
  done
done


# method=gptvq
# # # # 遍历数组中的每一个模型名称
# for bit in "${qbit_s[@]}"; do
#   for testmodle in "${models[@]}"; do
#     echo "$divider"
#     echo ">>>>>>>>Starting model: $testmodle" 
#     echo "$divider"
#     for rank in "${fix_ranks[@]}"; do
#       echo ">>>>>>>> model: $testmodle; rank = $rank, bit = $bit, method = $method, lora_bit=$low_bit" 
#       echo ">>>>>>>> bit = $bit"
#       python $sketch_dir/run_quantize.py \
#           --model_path "$data_base_dir/$testmodle" \
#           --output_path "$save_dir/${testmodle}-calib" \
#           --qbit "$bit" \
#           --groupsize "128" \
#           --fix_rank $rank  \
#           --lora_bit $low_bit \
#           --method $method
#           # --save_lora
#       echo ">>>>>>>>Starting test ppl: $testmodle, lora_bit:$low_bit, fix_rank:$rank, method:$method"
#       CUDA_VISIBLE_DEVICES=0 python $ppl_eval_dir/llama.py "$save_dir/${testmodle}-calib" "wikitext2"

#       echo ">>>>>>>>Starting zero shot task: $testmodle, lora_bit:$low_bit, fix_rank:$rank, method:$method"
#       HF_DATASETS_CACHE="/root/.cache/huggingface/datasets" lm_eval --model hf  --model_args "pretrained=$data_base_dir/${testmodle}-calib"  --tasks arc_challenge,arc_easy,winogrande,openbookqa


#       echo ">>>>>>>>Starting running fine turning: $testmodle, lora_bit:$low_bit, fix_rank:$rank, method:$method"
#       sh $ft_dir/rilq-llama2_7b-r64.sh

      
#       echo ">>>>>>>>Starting test FT-ppl: $testmodle, lora_bit:$low_bit, fix_rank:$rank, method:$method"
#       CUDA_VISIBLE_DEVICES=0 python $ppl_eval_dir/llama_rilq.py "$save_dir/${testmodle}-calib" "wikitext2"

#       echo ">>>>>>>>Starting FT-zero shot task: $testmodle, lora_bit:$low_bit, fix_rank:$rank, method:$method"
#       lm_eval --model hf  --model_args "pretrained=$data_base_dir/${testmodle}-calib,peft=$data_base_dir/${testmodle}-calib-rilq-r64/approx_init"  --tasks arc_challenge,arc_easy,winogrande,openbookqa

#       echo ">>>>>>>>Finish test: $testmodle, ratio:$ratio, delete files!"
#       rm -rf $data_base_dir/${testmodle}-calib
#       rm -rf $data_base_dir/${testmodle}-calib-rilq-r64
#     done
#   done
# done


