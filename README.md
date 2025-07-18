# Synexis

Yet another binding for llama.cpp. But this one is kinda different because it supports continuous batching with a slots
system mechanism (adapted from llama-server).

Currently, it supports only text completion

## Build

To build the project, run the following command from the root directory:

```bash
pip install .
```

## Usage

```python
from synexis_llm import SynexisLLM
llm = SynexisLLM(model_path)
stream= llm.chat.completions.create([
    {"role": "system", "content": "You are good assistant"},
    {"role": "user", "content": "Hello World!"}
],stream=True)
for token in stream:
    print(token, end='', flush=True)
```