workspace_dir=$(grep "WORKSPACE_DIR" ../config.py | cut -d "'" -f 2)
project_dir=$(grep "PROJECT_DIR" ../config.py | cut -d "'" -f 2)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ----------------------------- e5-base-v2  -----------------------------
file_path=${workspace_dir}/data/FlashRAG_Dataset/retrieval_corpus
index_file=$file_path/wiki18_100w_e5_index.index
corpus_file=$file_path/wiki18_100w.jsonl
retriever_name=e5
retriever_path=${workspace_dir}/llm/e5-base-v2

python search_r1/search/retrieval_server.py \
            --index_path $index_file \
            --corpus_path $corpus_file \
            --topk 3 \
            --retriever_name $retriever_name \
            --retriever_model $retriever_path \
            --faiss_gpu


