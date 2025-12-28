"""
T5 fine-tuning wrapper - uses Hugging Face T5 for Chinese-English translation
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class T5Wrapper(nn.Module):
    """
    T5 fine-tuning wrapper
    - Load pretrained T5 model
    - Automatically add translation task prefix
    - Support freezing Encoder
    """
    def __init__(
        self,
        model_name: str = 't5-small',
        task_prefix: str = 'translate Chinese to English: ',
        freeze_encoder: bool = False,
        max_length: int = 128
    ):
        super().__init__()

        if not HAS_TRANSFORMERS:
            raise ImportError("Please install transformers library: pip install transformers")

        self.task_prefix = task_prefix
        self.max_length = max_length

        # Load pretrained model
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        if freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass, returns loss and logits"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        return {'loss': outputs.loss, 'logits': outputs.logits}

    def generate(
        self,
        src_texts: list = None,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        num_beams: int = 4,
        max_length: int = None,
        **kwargs
    ) -> list:
        """Generate translation"""
        max_length = max_length or self.max_length

        if src_texts is not None:
            prefixed = [self.task_prefix + t for t in src_texts]
            encodings = self.tokenizer(
                prefixed, max_length=self.max_length,
                padding=True, truncation=True, return_tensors='pt'
            )
            input_ids = encodings['input_ids'].to(self.model.device)
            attention_mask = encodings['attention_mask'].to(self.model.device)

        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            **kwargs
        )

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)


def build_t5_model(config: dict) -> T5Wrapper:
    """Build T5 model from config"""
    return T5Wrapper(
        model_name=config.get('pretrained_model', 't5-small'),
        task_prefix=config.get('task_prefix', 'translate Chinese to English: '),
        freeze_encoder=config.get('freeze_encoder', False),
        max_length=config.get('max_length', 128)
    )
