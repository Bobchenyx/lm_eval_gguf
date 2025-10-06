## GGUF Benchmarking

We here provide 2 options for gguf benchmarking.

### llama.cpp native perplexity tools.
clone and build any llama.cpp version which supports your gguf models.
```
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF
cmake --build build --config Release -j --clean-first
```

several benchmarking options available at [llama.cpp/scripts](https://github.com/ggml-org/llama.cpp/tree/master/scripts)

```
# sh scripts/get-wikitext-2.sh
# sh scripts/get-winogrande.sh
# sh scripts/get-hellaswag.sh

build/bin/llama-perplexity -m "<Path-To-GGUF-Model>" -f <Pah-To-Benchmark-file> [other params]
# -f wikitext-2-raw/wiki.test.raw [other params]
# -f winogrande-debiased-eval.csv --winogrande [--winogrande-tasks N] [other params]
# -f hellaswag_val_full.txt --hellaswag [--hellaswag-tasks N] [other params]
```

### gguf patch code for lm-evaluation-harness

The [llama_server.py](llama_server.py) is a patch code for evaluating local gguf by [llama-server](https://github.com/ggml-org/llama.cpp/tree/master/examples/server) that is much faster than [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) provided in lm-harness.

```bash
git clone git@github.com:EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness

# step 1: place llama_server.py under lm_eval/models/
# step 2: add llama_server to lm_eval/models/__init__.py

pip install -e .
```

Start a local server by [llama-server](https://github.com/ggml-org/llama.cpp/tree/master/examples/server).
```bash
./llama.cpp/llama-server -m "<Path-To-GGUF-Model>" \
# -ngl 99 \                  (gpu-layers option)
# --jinja \                  (chat-template option)
# --temp 0.6 \               (inference params option)
# --top_p 0.95 \             ...
# --min_p 0.01 \             ...
# -ctx-size 8192 \           (context length.)
# -ot ".ffn_.*_exps.=CPU"    (expert offloading)
```

Use lm-eval to evaluate the model in another terminal. Since [llama-server](https://github.com/ggml-org/llama.cpp/tree/master/examples/server)'s response is different from [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), not all benchmarks were supported.

```bash
lm-eval --model llama-server --tasks mmlu --model_args base_url=http://127.0.0.1:8080
# --num_fewshot 5     (few shot option)
```
