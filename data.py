import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class VQADataset(Dataset):
    """
    easyVQA 데이터셋을 위한 커스텀 PyTorch Dataset 클래스.
    
    데이터 구조:
    ./easyVQA/
    ├── train/
    │   ├── images/
    │   │   ├── 0.png
    │   │   └── ...
    │   └── questions.json  (내용: [[question, answer, image_id], ...])
    └── labels.txt           (내용: [answer_1, answer_2, ...])
    """
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): easyVQA 폴더의 경로 (예: './easyVQA')
            split (string): 'train' 또는 'test' 등 데이터 스플릿
            transform (callable, optional): 이미지에 적용할 transform
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # 1. 경로 설정
        self.data_dir = os.path.join(root_dir, split)
        self.json_path = os.path.join(self.data_dir, 'questions.json')
        self.image_dir = os.path.join(self.data_dir, 'images')
        self.label_path = os.path.join(root_dir, 'labels.txt')

        # 2. questions.json 로드
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.annotations = json.load(f)
        except FileNotFoundError:
            print(f"오류: {self.json_path} 파일을 찾을 수 없습니다.")
            self.annotations = []
        except json.JSONDecodeError:
            print(f"오류: {self.json_path} 파일이 올바른 JSON 형식이 아닙니다.")
            self.annotations = []

        # 3. labels.txt 로드 (답변 문자열 -> 정수 인덱스 매핑)
        self.answer_to_idx = {}
        self.idx_to_answer = []
        try:
            with open(self.label_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    answer = line.strip()
                    if answer:
                        self.answer_to_idx[answer] = idx
                        self.idx_to_answer.append(answer)
        except FileNotFoundError:
            print(f"경고: {self.label_path} 파일을 찾을 수 없습니다. 답변을 인덱스로 변환할 수 없습니다.")
        
        self.num_answers = len(self.idx_to_answer)
        if self.num_answers > 0:
             print(f"총 {len(self.annotations)}개의 샘플과 {self.num_answers}개의 고유한 답변을 로드했습니다.")


    def __len__(self):
        """데이터셋의 총 샘플 수를 반환합니다."""
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        주어진 인덱스(idx)에 해당하는 샘플 1개를 불러옵니다.
        
        반환값:
            dict: { 'image': 이미지 텐서, 
                    'question': 질문 텍스트(string), 
                    'answer': 답변 인덱스(tensor) }
        """
        
        # 1. 어노테이션 정보 추출 (question, answer_str, image_id)
        try:
            question_text, answer_text, image_id = self.annotations[idx]
        except (IndexError, ValueError) as e:
            print(f"오류: 인덱스 {idx}의 어노테이션을 파싱하는 데 실패했습니다. {e}")
            # 유효하지 않은 인덱스에 대해 빈 샘플이나 None을 반환할 수 있습니다.
            # 여기서는 첫 번째 샘플을 대신 반환하거나 예외를 발생시킬 수 있습니다.
            # 간단하게 None을 처리할 수 있도록 None을 반환합니다. (collate_fn에서 처리 필요)
            return None # 혹은 예외 발생

        # 2. 이미지 로드
        # 참고: 이미지 파일 확장자가 .png가 아닐 수 있습니다. 
        # 실제 파일 확장자(예: .jpg)에 맞게 수정해야 할 수 있습니다.
        image_filename = f"{image_id}.png" 
        image_path = os.path.join(self.image_dir, image_filename)
        
        try:
            # 
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"경고: 이미지 파일 {image_path}를 찾을 수 없습니다. 회색 이미지로 대체합니다.")
            image = Image.new('RGB', (224, 224), color='grey') # Placeholder

        # 3. 이미지 변환 (Transform) 적용
        if self.transform:
            image = self.transform(image)

        # 4. 답변(answer_text)을 정수 인덱스(tensor)로 변환
        answer_idx = self.answer_to_idx.get(answer_text, -1) # labels.txt에 없는 답변은 -1 처리
        
        if answer_idx == -1:
             print(f"경고: 답변 '{answer_text}'를 labels.txt에서 찾을 수 없습니다.")
             # VQA는 보통 분류 문제로 접근하므로, CrossEntropyLoss의 
             # ignore_index=-100을 사용하거나, 0번 인덱스(예: 'unknown')를 사용할 수 있습니다.
             # 여기서는 -1을 그대로 두고, 손실 함수에서 ignore_index=-1로 설정한다고 가정합니다.
        
        answer_tensor = torch.tensor(answer_idx, dtype=torch.long)

        # 5. 샘플을 딕셔너리 형태로 반환
        # 질문(question_text)은 collate_fn에서 리스트로 묶인 후, 
        # 모델의 forward pass 직전에 tokenizer로 처리됩니다.
        return {
            'image': image,
            'question': question_text,
            'answer': answer_tensor
        }

# --- DataLoader를 위한 Custom Collate Function ---
def collate_fn_with_tokenizer(batch, tokenizer):
    """
    Collate_fn 내부에서 토큰화까지 수행합니다.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    images = [item['image'] for item in batch]
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]

    # 이미지와 답변은 스택
    batch_images = torch.stack(images, dim=0)
    batch_answers = torch.stack(answers, dim=0)
    
    # 
    # 질문(텍스트 리스트)을 여기서 바로 토큰화 (동적 패딩)
    tokenized_inputs = tokenizer(
        questions, 
        padding='longest',  # 배치 내 최대 길이로 패딩
        truncation=True, 
        return_tensors="pt",
        max_length=77 # (예: CLIP의 경우 77)
    )
    
    return {
        'image': batch_images,
        'inputs': tokenized_inputs, # input_ids, attention_mask 등이 담긴 딕셔너리
        'answer': batch_answers
    }