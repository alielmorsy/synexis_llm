__all__ = ["SynexisLLM"]

import os
import sys
import time
import uuid
from typing import Optional, List, Dict, Any, Tuple, Generator, Union

if sys.platform == "win32":
    dll_dir = os.path.join(os.path.dirname(__file__), "lib")
    if os.path.exists(dll_dir):
        os.add_dll_directory(dll_dir)
try:
    from .synexis_python import Synexis, TaskParams, SamplingParams, SynexisArguments
except:
    # Loading DLLs manually. For some reason sometimes it works normally but most of the time DLLs has to be loaded manually
    import ctypes
    for dll_file in os.listdir(dll_dir):
        if dll_file.lower().endswith(".dll"):
            path = os.path.join(dll_dir, dll_file)
            try:
                ctypes.WinDLL(path,winmode=0)
            except Exception as e:
                print(f"Failed loading {path}: {e}")
    from .synexis_python import Synexis, TaskParams, SamplingParams, SynexisArguments

from jinja2 import Template


class SynexisLLM:
    """
    A large language model interface for Synexis, providing an OpenAI-like API.
    """

    def __init__(self, model_path: str,
                 model_projector_path: Optional[str] = None,
                 n_slots: int = 8,
                 n_ctx: int = 4096,
                 n_batch: int = 2048,
                 n_keep: int = 512,
                 use_mmap: bool = True,
                 number_of_threads: int = 10,
                 number_gpu_layers: int = -1
                 ):
        """
        Initializes the SynexisLLM model.

        :param model_path: Path to the GGUF model file.
        :param model_projector_path: Optional path to a multimodal projector file.
        :param n_slots: Number of parallel processing slots.
        :param n_ctx: Context size.
        :param n_batch: Batch size for prompt processing.
        :param n_keep: Number of tokens to keep from the initial prompt.
        :param use_mmap: Whether to use memory-mapped files.
        :param number_of_threads: Number of threads for processing.
        :param number_gpu_layers: Number of layers to offload to GPU (-1 for all).
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_path = model_path
        args = SynexisArguments(model_path)
        if model_projector_path is not None and os.path.exists(model_projector_path):
            args.model_projector_path = model_projector_path

        args.n_slots = n_slots
        args.n_ctx = n_ctx
        args.n_batch = n_batch
        args.n_keep = n_keep
        args.use_mmap = use_mmap
        args.number_of_threads = number_of_threads
        args.number_of_gpu_layers = number_gpu_layers

        self.handle = Synexis(args)
        self.chat = Chat(self)
        self.jinja_template = Template(self.handle.get_template())
        self.handle.run()

    def _apply_chat_template(self, messages: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """
        Applies a chat template to a list of messages to create a single prompt string
        and extracts a list of media file paths.
        """
        prompt = self.jinja_template.render(messages=messages, **self.handle.get_tokens())
        files = []
        for message in messages:
            content = message.get('content')
            if isinstance(content, list):
                for content_part in content:
                    if content_part.get('type') in ('image', 'audio') and 'path' in content_part:
                        files.append(content_part['path'])
        return prompt, files


class Chat:
    def __init__(self, llm: 'SynexisLLM'):
        self.completions = Completions(llm)
        self.media_cache: Dict[str, bytes] = {}


class Completions:
    def __init__(self, llm: 'SynexisLLM'):
        self._llm = llm

    def create(self,
               messages: List[Dict[str, Any]],
               temperature: float = 0.8,
               top_k: int = 40,
               top_p: float = 0.95,
               min_p: float = 0.05,
               repeat_penalty: float = 1.1,
               max_tokens: int = 256,
               stop: Optional[List[str]] = None,
               stream: bool = False) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """
        Creates a chat completion response.

        :param messages: A list of messages, where each message is a dict with "role" and "content".
        :param temperature: Sampling temperature.
        :param top_k: Top-k sampling.
        :param top_p: Top-p (nucleus) sampling.
        :param min_p: Min-p sampling.
        :param repeat_penalty: Penalty for repeating tokens.
        :param max_tokens: Maximum number of tokens to generate.
        :param stop: A list of strings to stop generation at.
        :param stream: Whether to stream the response.
        :return: A dictionary with the completion response, or an iterator for streaming.
        """
        prompt, file_paths = self._llm._apply_chat_template(messages)

        sampling_params = SamplingParams(
            temp=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            penalty_repeat=repeat_penalty,
        )

        task_params = TaskParams(
            prompt
            # TODO: Support stop tokens
            # stop_prompts=stop if stop is not None else []
        )

        for file_path in file_paths:
            if file_path in self._llm.chat.media_cache:
                media_bytes = self._llm.chat.media_cache[file_path]
            else:
                try:
                    with open(file_path, "rb") as f:
                        media_bytes = f.read()
                    self._llm.chat.media_cache[file_path] = media_bytes
                except FileNotFoundError:
                    raise FileNotFoundError(f"Media file not found: {file_path}")
                except IOError as e:
                    raise IOError(f"Error reading media file {file_path}: {e}")

            task_params.add_media(media_bytes)

        task_params.sampling_params = sampling_params

        if stream:
            return self._create_stream(task_params)

        result_text = self._llm.handle.complete(task_params)

        response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self._llm.model_path,
            "result": {
                "message": {
                    "role": "assistant",
                    "content": result_text,
                },
                "finish_reason": "stop",
            },
            "usage": {
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1,
            }
        }
        return response

    def _create_stream(self, task_params: TaskParams):
        for token in self._llm.handle.complete_stream(task_params):
            yield token
