# lm_eval_gguf

The [llama_server.py](llama_server.py) is a patch code for evaluating local gguf by [llama-server](https://github.com/ggml-org/llama.cpp/tree/master/examples/server) that is much faster than [llama-cpp-python](https://github.com/abetlen/llama-cpp-python).

```bash
git clone git@github.com:EleutherAI/lm-evaluation-harness.git
```

Copy [llama_server.py](llama_server.py) to lm-evaluation-harness/lm_eval/models/

Add llama_server in lm-evaluation-harness/lm_eval/models/__init__.py

```bash
cd lm-evaluation-harness
pip install -e .
```

Start a local server by [llama-server](https://github.com/ggml-org/llama.cpp/tree/master/examples/server). Refer to [this repo](https://github.com/tflsxyy/DeepSeek-V3).

```bash
./llama.cpp/llama-server -m /root/dataDisk/deepseek-ai/DeepSeek-V3-0324-MoE-Pruner-E160-IQ1_S/DeepSeek-V3-0324-MoE-Pruner-E160-IQ1_S-00001-of-00018.gguf -ngl 62
```

Use lm-eval to evaluate the model in another terminal.

```bash
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"
HF_DATASETS_TRUST_REMOTE_CODE=1 nohup lm-eval --model llama-server --tasks arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,piqa,rte,winogrande --model_args base_url=http://127.0.0.1:8080
```
