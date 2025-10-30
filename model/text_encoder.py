import torch.nn as nn
from transformers import BertModel

class TextEncoder_Bert(nn.Module):
    """
    사전 학습된 BERT 모델을 Text Encoder로 사용합니다.
    """
    def __init__(self, model_name='bert-base-uncased', hidden_dim=512):
        super().__init__()
        
        # 1. BERT 모델 로드
        self.bert = BertModel.from_pretrained(model_name)
        
        # 2. BERT의 기본 출력 차원 (bert-base-uncased는 768)
        bert_output_dim = self.bert.config.hidden_size # 768
        
        # 3. (선택 사항) BERT 출력을 VQA 모델의 공통 차원(hidden_dim)으로 매핑
        #    (VisionEncoder와 차원을 맞추기 위함)
        self.seq_projection = nn.Linear(bert_output_dim, hidden_dim)
        self.global_projection = nn.Linear(bert_output_dim, hidden_dim)
        
        self.output_dim = hidden_dim # 최종 출력 차원은 512

    def forward(self, input_ids, attention_mask):
        """
        데이터 로더에서 collate_fn을 통해 생성된
        input_ids와 attention_mask를 직접 받습니다.
        """
        
        # BERT 모델은 device 처리가 필요 없습니다.
        # 모델 자체가 .to(device)로 이동하면 내부 텐서도 모두 이동합니다.
        
        # 1. BERT 순전파
        outputs = self.bert(input_ids=input_ids, 
                            attention_mask=attention_mask)
        
        # 2. BERT 출력에서 시퀀스 특징과 전역 특징 추출
        last_hidden_state = outputs.last_hidden_state # (B, L, 768)
        pooler_output = outputs.pooler_output     # (B, 768) [CLS] 토큰의 특징
        
        # 3. (선택 사항) Projection Layer 통과
        q_seq = self.seq_projection(last_hidden_state)     # (B, L, 512)
        q_global = self.global_projection(pooler_output) # (B, 512)
        
        # (만약 hidden_dim=768로 모델 전체를 통일한다면 projection은 필요 없습니다)

        return q_seq, q_global