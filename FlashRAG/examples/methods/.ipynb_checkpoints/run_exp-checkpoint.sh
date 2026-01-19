

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export MASTER_PORT=12334

########################################################################################################################################################
# --------------------- retrievers based on our agent ---------------------
########################################################################################################################################################
################ deepr #################
deepr-iter0
generator_model=triviaqa_hotpotqa_train-search-r1-ppo-qwen2.5-7b-em
retriever=deepr-bge_triviaqa_hotpotqa_train-search-r1-ppo-qwen2.5-7b-em_global-local_question-currentq
dataset_names_all="nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle"
python run_exp.py \
    --generator_model_path /root/paddlejob/workspace/trained_models/$generator_model \
    --retrieval_model_path /root/paddlejob/workspace/trained_models/$retriever/ \
    --index_path /root/paddlejob/workspace/data/FlashRAG_Dataset/retrieval_corpus/${retriever}_Flat.index \
    --method_name search-r1 \
    --retrieval_method e5 \
    --dataset_names_all $dataset_names_all \
    --gpu_id "0,1,2,3" \
    --agentic_retriever_input \

done

