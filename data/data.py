import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

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
        
        self.data_dir = os.path.join(root_dir, split)
        self.json_path = os.path.join(self.data_dir, 'questions.json')
        self.image_dir = os.path.join(self.data_dir, 'images')
        self.label_path = os.path.join(root_dir, 'labels.txt')
        self.sensitive_path_split = os.path.join(self.data_dir, 'sensitive.json')
        self.sensitive_path_root = os.path.join(root_dir, 'sensitive.json')

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
             print(f"Data sample: {len(self.annotations)}, Label: {self.num_answers}.")

        # 4. Optional 민감 레이블 로드 (존재 시 사용)
        # 우선순위: split 폴더의 sensitive.json -> root의 sensitive.json
        # 포맷 예시: {"0": 1, "1": 0, ...}  또는  {"0001": 2, ...}
        self.sensitive_map = None
        if os.path.exists(self.sensitive_path_split):
            try:
                with open(self.sensitive_path_split, 'r', encoding='utf-8') as f:
                    self.sensitive_map = json.load(f)
            except Exception as e:
                print(f"경고: {self.sensitive_path_split} 로부터 민감 레이블을 로드하지 못했습니다: {e}")
        elif os.path.exists(self.sensitive_path_root):
            try:
                with open(self.sensitive_path_root, 'r', encoding='utf-8') as f:
                    self.sensitive_map = json.load(f)
            except Exception as e:
                print(f"경고: {self.sensitive_path_root} 로부터 민감 레이블을 로드하지 못했습니다: {e}")


    def __len__(self):
        """데이터셋의 총 샘플 수를 반환합니다."""
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        주어진 인덱스(idx)에 해당하는 샘플 1개를 불러옵니다.
        
        반환값:
            dict: {
                'image': 이미지 텐서,
                'question': 질문 텍스트(string),
                'answer': 답변 인덱스(tensor),
                'sensitive': 민감 레이블 텐서(int) 또는 -1(없음)
            }
        """
        
        try:
            question_text, answer_text, image_id = self.annotations[idx]
        except (IndexError, ValueError) as e:
            print(f"오류: 인덱스 {idx}의 어노테이션을 파싱하는 데 실패했습니다. {e}")
            return None

        image = None
        for ext in ['.png', '.jpg', '.jpeg']:
            image_filename = f"{image_id}{ext}"
            image_path = os.path.join(self.image_dir, image_filename)
            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert('RGB')
                    break
                except Exception as e:
                    print(f"Warning: Failed to load {image_path}: {e}")
                    continue
        
        if image is None:
            print(f"Warning: No valid image found for ID {image_id}. Using grey placeholder.")
            image = Image.new('RGB', (224, 224), color='grey')

        if self.transform:
            image = self.transform(image)

        answer_idx = self.answer_to_idx.get(answer_text, -1) # labels.txt에 없는 답변은 -1 처리
        
        if answer_idx == -1:
             print(f"경고: 답변 '{answer_text}'를 labels.txt에서 찾을 수 없습니다.")

        answer_tensor = torch.tensor(answer_idx, dtype=torch.long)

        # 민감 레이블: 우선 어노테이션 4번째 필드, 없으면 sensitive_map에서 조회, 둘 다 없으면 -1
        sensitive_label = -1
        try:
            if isinstance(self.annotations[idx], (list, tuple)) and len(self.annotations[idx]) >= 4:
                # 4번째 필드 사용 (int로 기대)
                sensitive_label = int(self.annotations[idx][3])
            elif self.sensitive_map is not None:
                # image_id를 키로 조회 (문자열/정수 키 모두 처리)
                key_candidates = [str(image_id), int(image_id) if str(image_id).isdigit() else None]
                for k in key_candidates:
                    if k is None:
                        continue
                    if k in self.sensitive_map:
                        sensitive_label = int(self.sensitive_map[k])
                        break
        except Exception:
            # 어떤 이유로든 파싱 실패 시 -1 유지
            sensitive_label = -1
        sensitive_tensor = torch.tensor(sensitive_label, dtype=torch.long)

        return {
            'image': image,
            'question': question_text,
            'answer': answer_tensor,
            'sensitive': sensitive_tensor
        }

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
    sensitives = [item.get('sensitive', torch.tensor(-1, dtype=torch.long)) for item in batch]

    batch_images = torch.stack(images, dim=0)
    batch_answers = torch.stack(answers, dim=0)
    batch_sensitive = torch.stack(sensitives, dim=0)
    
    tokenized_inputs = tokenizer(
        questions, 
        padding='longest',  # 배치 내 최대 길이로 패딩
        truncation=True, 
        return_tensors="pt",
        max_length=77 # (예: CLIP의 경우 77)
    )
    
    return {
        'image': batch_images,
        'inputs': tokenized_inputs,
        'answer': batch_answers,
        'sensitive': batch_sensitive
    }