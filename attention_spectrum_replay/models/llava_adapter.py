from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tiny_mllm import MLLMOutput


class HFMLLMAdapter(nn.Module):
    """Adapter for LLaVA/Qwen/InternVL-style Hugging Face MLLMs.

    Local tiny-model runs do not require transformers. Real benchmark experiments
    requires installing the optional `hf` extras and using a checkpoint whose
    forward pass supports output_attentions=True and returns visual-token
    cross-attention or decoder attention over image tokens.

    Because model families expose image-token attentions differently, this
    adapter provides a documented extraction path and explicit errors when a
    requested family does not expose the needed tensors. It is meant to be
    subclassed for local checkpoint-specific processors if necessary.
    """

    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        image_token_start: Optional[int] = None,
        image_token_end: Optional[int] = None,
        torch_dtype: Optional[torch.dtype] = None,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        use_peft: bool = True,
    ):
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "HFMLLMAdapter requires optional dependencies. Install with: pip install -e '.[hf]'"
            ) from exc
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.num_classes = int(num_classes)
        hidden = getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "text_config", self.model.config).hidden_size
        self.classifier = nn.Linear(hidden, num_classes)
        self.image_token_start = image_token_start
        self.image_token_end = image_token_end
        if use_peft:
            self._attach_lora(lora_rank, lora_alpha, lora_dropout)

    def _attach_lora(self, rank: int, alpha: int, dropout: float) -> None:  # pragma: no cover - optional dependency
        try:
            from peft import LoraConfig, get_peft_model
        except Exception as exc:
            raise ImportError("LoRA requested but peft is not installed. Install with: pip install peft") from exc
        cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["q_proj", "v_proj", "query", "value"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, cfg)

    def forward(self, **batch: Any) -> MLLMOutput:  # pragma: no cover - optional dependency
        outputs = self.model(
            **{k: v for k, v in batch.items() if k not in {"labels", "functional_token_mask", "question_text", "skill_probs"}},
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        pooled = hidden.mean(dim=1)
        logits = self.classifier(pooled)
        attn = self.extract_cross_attentions(outputs, batch)
        image_embeds = pooled
        text_embeds = pooled
        return MLLMOutput(logits=logits, cross_attentions=attn, image_embeds=image_embeds, text_embeds=text_embeds)

    def extract_cross_attentions(self, outputs: Any, batch: Dict[str, Any]) -> torch.Tensor:  # pragma: no cover - optional dependency
        attentions = getattr(outputs, "cross_attentions", None)
        if attentions is not None:
            layers = []
            for a in attentions:
                # expected [B, heads, Tq, P] or [B, heads, Tq, Gh, Gw]
                layers.append(a)
            return torch.stack(layers, dim=1)
        decoder_attn = getattr(outputs, "attentions", None)
        if decoder_attn is None or self.image_token_start is None or self.image_token_end is None:
            raise RuntimeError(
                "Could not extract cross-modal attention. Provide image_token_start/image_token_end for decoder-attention models, "
                "or subclass HFMLLMAdapter.extract_cross_attentions for this checkpoint."
            )
        image_slice = slice(self.image_token_start, self.image_token_end)
        layers = []
        for a in decoder_attn:
            # decoder self-attention [B, heads, seq, seq]; use attention from text queries to image-token keys.
            layers.append(a[..., image_slice])
        return torch.stack(layers, dim=1)
