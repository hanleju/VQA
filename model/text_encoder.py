import torch.nn as nn
from transformers import BertModel

class TextEncoder_Bert(nn.Module):
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