# Search-R1 Web Demo

A web demo of the Search-R1 paper. Ask a question in your browser and follow the model's reasoning, search queries, retrieved Wikipedia passages, and final answer.

The demo uses `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo` with an `intfloat/e5-base-v2` retriever and the Wikipedia 2018 corpus.

## Requirements

- Linux with Conda and an NVIDIA GPU compatible with CUDA 12.1.
- Sufficient GPU memory for the 7B model, retriever, and FAISS GPU index.
- Internet access to download dependencies, model weights, and corpus files, plus sufficient disk space for these downloads.
- `curl` and `nohup` available on your system.

Run the following commands from the repository root. On a cluster, launch the demo on an allocated GPU node.

## 1. Create the environment

```bash
conda create -n searchr1 python=3.10
conda activate searchr1

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install -r requirements.txt
```

## 2. Download and prepare the corpus

```bash
save_path=./corpus
python scripts/download.py --save_path "$save_path"
cat "$save_path"/part_aa "$save_path"/part_ab > "$save_path/e5_Flat.index"
gzip -d "$save_path/wiki-18.jsonl.gz"
```

After these commands finish successfully, remove the downloaded index parts to reclaim disk space:

```bash
rm "$save_path/part_aa" "$save_path/part_ab"
```

The resulting directory should contain:

```text
corpus/
├── e5_Flat.index
└── wiki-18.jsonl
```

## 3. Start the web demo

With the `searchr1` environment active, run:

```bash
sh run_demo.sh
```

The launcher starts the retrieval server, waits for it to become ready, and then loads the model and starts the web server. Initial startup can take several minutes while model weights download and the index loads.

Once ready, open **http://127.0.0.1:7860** in your browser and enter a question.

## Acknowledgments

This demo is based on the **Search-R1** paper and uses the Search-R1 model and retrieval resources. Credit for the original research and released resources goes to the Search-R1 authors.
