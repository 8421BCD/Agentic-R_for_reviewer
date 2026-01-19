## ⚡ Quick Start for testing Agentic-R

### **📘** Environment and Preparation

##### Environment

In this step, we will describe the required packages for inferencing with Agentic-R. We strongly recommend using a separate conda environment.

```bash
# ---------------------------------- create env ----------------------------------
conda create -n agentic-r python=3.10 -y
source ~/.bashrc
conda activate agentic-r
# ---------------------------------- install packages ----------------------------------
cd FlashRAG
pip install -e .
pip install vllm==0.10.1
pip install sentence-transformers
pip install pyserini
pip install GPUtil
pip install nvitop
pip install termcolor
pip install numpy==1.26
pip install deepspeed==0.18.0
pip install qwen_omni_utils
pip install modelscope
pip install faiss_gpu==1.7.3
pip install transformers==4.57.1
```

##### Preparation

After installing the necessary packages, remember to **update** the ``WORKSPACE_DIR`` and ``PROJECT_DIR`` (both should be absolute paths) in ``config.py``. These two parameters will be used both in our inference codes and training codes. Here is a recommended directory structure:

```bash
{WORKSPACE_DIR}
├── trained_models
│   ├── Agentic-R_e5
│   └── triviaqa_hotpotqa_train-search-r1-ppo-qwen2.5-7b-em-iter1
├── data
│   ├── FlashRAG_Dataset
│   └──── nq
│   └──── hotpotqa
│   └──── retrieval_corpus
│   └──── ...
└── {PROJECT_DIR} (i.e., Agentic-R)
    ├── FlashRAG
    └── Search-R1
    └── tevatron
    └── config.py
```

**b**. Download the datasets for testing (such as nq, hotpotqa, ...). 

**c**. Build the wikipedia index based on the following code:

```shell
conda activate agentic-r
model_name=Agentic-R_e5
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m flashrag.retriever.index_builder \
    --retrieval_method ${model_name} \
    --model_path {WORKSPACE_DIR}/trained_models/${model_name} \
    --corpus_path {WORKSPACE_DIR}/data/FlashRAG_Dataset/retrieval_corpus/wiki18_100w.jsonl \
    --save_dir {WORKSPACE_DIR}/data/FlashRAG_Dataset/retrieval_corpus/ \
    --use_fp16 \
    --max_length 256 \
    --batch_size 128 \
    --faiss_type Flat \
    --sentence_transformer \
    --instruction "passage: "
```

#### Testing Agentic-R based on our trained Agent

```shell
conda activate agentic-r
cd FlashRAG/examples/methods
bash run_exp.sh
```

*Note: For our Agentic-R, the parameter `agentic_retriever_input` is set as True, which uses 'Question [SEP] query' for retrieval.*

