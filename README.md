# lm_eval_gguf

The [llama_server.py](llama_server.py) is a patch code for evaluating local gguf by [llama-server](https://github.com/ggml-org/llama.cpp/tree/master/examples/server) that is much faster than [llama-cpp-python](https://github.com/abetlen/llama-cpp-python).

```bash
git clone git@github.com:EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
curl -L https://raw.githubusercontent.com/tflsxyy/lm_eval_gguf/main/llama_server.py -o lm_eval/models/llama_server.py
sed -i '11i\    llama_server,' lm_eval/models/__init__.py
pip install -e .
```

Start a local server by [llama-server](https://github.com/ggml-org/llama.cpp/tree/master/examples/server). Refer to [this repo](https://github.com/tflsxyy/DeepSeek-V3).

```bash
./llama.cpp/llama-server -m /root/dataDisk/tflsxyy/DeepSeek-V3-0324-MoE-Pruner-E160-IQ1_S/DeepSeek-V3-0324-MoE-Pruner-E160-IQ1_S-00001-of-00018.gguf -ngl 62
```

Use lm-eval to evaluate the model in another terminal. Since [llama_server.py](llama_server.py)'s response is different from [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), only MMLU is supported.

```bash
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"
HF_DATASETS_TRUST_REMOTE_CODE=1 lm-eval --model llama-server --tasks mmlu --model_args base_url=http://127.0.0.1:8080
```
