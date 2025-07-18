# Synexis

Yet another binding for llama.cpp. But this one is kinda different because it supports continous 

## Build

To build the project, run the following command from the root directory:

```bash
pip install .
```

## Usage

```python
import synexis

synexis.init_backend()
print(synexis.get_system_info())
synexis.free_backend()
```
