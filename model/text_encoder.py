import torch.nn as nn
from transformers import BertModel, RobertaModel

# optional PEFT (QLoRA) support
try:
    from peft import get_peft_model, LoraConfig
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

from transformers import AutoModel


class Bert(nn.Module):
    """
    pretrained BERT 사용
    """
    def __init__(self, model_name='bert-base-uncased', hidden_dim=512):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(model_name)
        
        # 2. (bert-base-uncased dim = 768
        bert_output_dim = self.bert.config.hidden_size # 768
        
        # 3. (선택 사항) BERT dim hidden_dim mapping
        self.seq_projection = nn.Linear(bert_output_dim, hidden_dim)
        self.global_projection = nn.Linear(bert_output_dim, hidden_dim)
        
        self.output_dim = hidden_dim # 최종 출력 차원: 512

    def forward(self, input_ids, attention_mask):
        """
        데이터 로더에서 collate_fn을 통해 생성된
        input_ids와 attention_mask를 직접 받습니다.
        """
        
        outputs = self.bert(input_ids=input_ids, 
                            attention_mask=attention_mask)
        
        last_hidden_state = outputs.last_hidden_state # (B, L, 768)
        pooler_output = outputs.pooler_output     # (B, 768) [CLS] 토큰의 특징
        
        # 3. Projection Layer
        q_seq = self.seq_projection(last_hidden_state)     # (B, L, 512)
        q_global = self.global_projection(pooler_output) # (B, 512)
        
        return q_seq, q_global
    

class RoBerta(nn.Module):
    '''
    pretrained RoBerta 사용
    '''

    def __init__(self, model_name='roberta-base', hidden_dim=512):
        super().__init__()
        
        self.bert = RobertaModel.from_pretrained(model_name)
        
        # 2. (bert-base-uncased dim = 768
        bert_output_dim = self.bert.config.hidden_size # 768
        
        # 3. (선택 사항) BERT dim hidden_dim mapping
        self.seq_projection = nn.Linear(bert_output_dim, hidden_dim)
        self.global_projection = nn.Linear(bert_output_dim, hidden_dim)
        
        self.output_dim = hidden_dim # 최종 출력 차원: 512

    def forward(self, input_ids, attention_mask):
        """
        데이터 로더에서 collate_fn을 통해 생성된
        input_ids와 attention_mask를 직접 받습니다.
        """
        
        outputs = self.bert(input_ids=input_ids, 
                            attention_mask=attention_mask)
        
        last_hidden_state = outputs.last_hidden_state # (B, L, 768)
        pooler_output = outputs.pooler_output     # (B, 768) [CLS] 토큰의 특징
        
        # 3. Projection Layer
        q_seq = self.seq_projection(last_hidden_state)     # (B, L, 512)
        q_global = self.global_projection(pooler_output) # (B, 512)
        
        return q_seq, q_global


class BertQLoRA(nn.Module):
    """
    BERT wrapped with PEFT LoRA (QLoRA-style) adapters.
    Requires `peft` to be installed. Keeps same forward API as `Bert` above.
    """
    def __init__(self, model_name='bert-base-uncased', hidden_dim=512,
                 lora_r=8, lora_alpha=16, lora_dropout=0.1, target_modules=None):
        super().__init__()
        if not PEFT_AVAILABLE:
            raise ImportError("peft is required for BertQLoRA. Install with `pip install peft bitsandbytes`")

        base = AutoModel.from_pretrained(model_name)
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules or ["query", "value"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
        )
        self.bert = get_peft_model(base, lora_cfg)

        bert_output_dim = self.bert.config.hidden_size
        self.seq_projection = nn.Linear(bert_output_dim, hidden_dim)
        self.global_projection = nn.Linear(bert_output_dim, hidden_dim)
        self.output_dim = hidden_dim

    def forward(self, input_ids, attention_mask):
        # When using PEFT wrappers, calling the wrapper's forward can sometimes
        # inject unexpected kwargs (e.g. 'labels') into the underlying model.
        # Calling the wrapped model's base_model.forward directly avoids that
        # while still using the adapter-modified modules.
        if hasattr(self.bert, 'base_model'):
            outputs = self.bert.base_model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooler_output = getattr(outputs, 'pooler_output', None)
        if pooler_output is None:
            # fallback: use CLS token embedding
            pooler_output = last_hidden_state[:, 0, :]

        q_seq = self.seq_projection(last_hidden_state)
        q_global = self.global_projection(pooler_output)
        return q_seq, q_global


class RoBertaQLoRA(nn.Module):
    """
    RoBERTa wrapped with PEFT LoRA adapters.
    """
    def __init__(self, model_name='roberta-base', hidden_dim=512,
                 lora_r=8, lora_alpha=16, lora_dropout=0.1, target_modules=None):
        super().__init__()
        if not PEFT_AVAILABLE:
            raise ImportError("peft is required for RoBertaQLoRA. Install with `pip install peft bitsandbytes`")

        base = AutoModel.from_pretrained(model_name)
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules or ["query", "value"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
        )
        self.bert = get_peft_model(base, lora_cfg)

        bert_output_dim = self.bert.config.hidden_size
        self.seq_projection = nn.Linear(bert_output_dim, hidden_dim)
        self.global_projection = nn.Linear(bert_output_dim, hidden_dim)
        self.output_dim = hidden_dim

    def forward(self, input_ids, attention_mask):
        if hasattr(self.bert, 'base_model'):
            outputs = self.bert.base_model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooler_output = getattr(outputs, 'pooler_output', None)
        if pooler_output is None:
            pooler_output = last_hidden_state[:, 0, :]

        q_seq = self.seq_projection(last_hidden_state)
        q_global = self.global_projection(pooler_output)
        return q_seq, q_global