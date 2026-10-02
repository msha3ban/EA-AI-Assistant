from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from typing import Any
from urllib.request import Request, urlopen

from .config import LLMCallConfig

Transport = Callable[[str, str, bytes | None, float], bytes]


class OllamaClient:
    def __init__(
        self,
        endpoint: str,
        timeout: float = 120,
        transport: Transport | None = None,
        *,
        unload_timeout: float = 15.0,
        poll_interval: float = 0.25,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.endpoint, self.timeout = endpoint.rstrip("/"), timeout
        self.transport = transport or self._request
        self.unload_timeout = unload_timeout
        self.poll_interval = poll_interval
        self.clock = clock
        self.sleep = sleep
        self.models: dict[str, str] = {}

    def _request(
        self, method: str, url: str, data: bytes | None, timeout: float
    ) -> bytes:
        req = Request(
            url, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        with urlopen(req, timeout=timeout) as response:
            return bytes(response.read())

    def _call(
        self,
        method: str,
        path: str,
        data: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        raw = self.transport(
            method,
            self.endpoint + path,
            json.dumps(data).encode() if data is not None else None,
            self.timeout if timeout is None else timeout,
        )
        value: Any = json.loads(raw)
        if not isinstance(value, dict):
            raise TypeError("Ollama returned a non-object response")
        return value

    def model_digest(self, model: str) -> str:
        if model not in self.models:
            models = self._call("GET", "/api/tags").get("models", [])
            self.models[model] = next(
                (
                    str(x.get("digest", "unknown"))
                    for x in models
                    if x.get("name") == model
                ),
                "unknown",
            )
        return self.models[model]

    def chat(
        self,
        instructions: str,
        content: str,
        config: LLMCallConfig,
        schema: dict[str, Any] | None = None,
    ) -> str:
        prompt = (
            instructions + "\n<meeting-content>\n" + content + "\n</meeting-content>"
        )
        estimate = math.ceil(len(prompt) / 2.5)
        if estimate + config.num_predict > config.num_ctx:
            raise ValueError(
                f"Ollama request exceeds token budget: estimated {estimate} input + {config.num_predict} output exceeds num_ctx {config.num_ctx}"
            )
        self.model_digest(config.model)
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": instructions},
                {
                    "role": "user",
                    "content": "<meeting-content>\n" + content + "\n</meeting-content>",
                },
            ],
            "stream": False,
            "think": config.think,
            "options": {
                "num_ctx": config.num_ctx,
                "num_predict": config.num_predict,
                "temperature": config.temperature,
            },
        }
        if schema is not None:
            payload["format"] = schema
        result = self._call("POST", "/api/chat", payload, timeout=config.timeout)
        message = result.get("message", {})
        text = message.get("content", "")
        prompt_count = result.get("prompt_eval_count")
        eval_count = result.get("eval_count", 0)
        if isinstance(prompt_count, int) and (
            prompt_count >= config.num_ctx
            or prompt_count + int(eval_count or 0) >= config.num_ctx
        ):
            raise RuntimeError("Ollama response exhausted num_ctx and may be truncated")
        if (
            result.get("done_reason") == "length"
            or result.get("done") is False
            or not text
        ):
            raise RuntimeError("Ollama returned an incomplete or empty response")
        if schema is not None:
            try:
                json.loads(text)
            except (ValueError, TypeError) as exc:
                raise RuntimeError("Ollama returned invalid JSON") from exc
        return str(text)

    def _loaded_models(self) -> list[str]:
        try:
            response = self._call("GET", "/api/ps")
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError(
                "Unable to verify GPU is free: Ollama /api/ps failed"
            ) from exc
        models = response.get("models", [])
        if not isinstance(models, list):
            raise TypeError(
                "Unable to verify GPU is free: invalid Ollama /api/ps response"
            )
        return list(
            dict.fromkeys(
                str(item.get("name") or item.get("model"))
                for item in models
                if isinstance(item, dict) and (item.get("name") or item.get("model"))
            )
        )

    def unload(self, model: str) -> None:
        self._call(
            "POST",
            "/api/chat",
            {"model": model, "messages": [], "stream": False, "keep_alive": 0},
        )
        deadline = self.clock() + self.unload_timeout
        while True:
            if model not in self._loaded_models():
                return
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise RuntimeError(
                    f"Ollama model {model} is still loaded after {self.unload_timeout:g} seconds"
                )
            self.sleep(min(self.poll_interval, remaining))

    def ensure_gpu_free(self) -> None:
        deadline = self.clock() + self.unload_timeout
        while True:
            loaded = self._loaded_models()
            if not loaded:
                return
            for model in loaded:
                self.unload(model)
            if self.clock() >= deadline and self._loaded_models():
                raise RuntimeError(
                    "Ollama GPU still has loaded models after cleanup timeout"
                )
