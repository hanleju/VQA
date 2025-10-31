@echo OFF
REM -----------------------------------------------------------------
REM 이 스크립트는 여러 VQA 모델을 순차적으로 학습시킵니다.
REM -----------------------------------------------------------------

echo [INFO] Activating Conda environment 'vqa'...
REM (vqa) 환경을 활성화합니다. Conda/Anaconda를 사용하지 않는다면 이 줄을 지우거나 수정하세요.
call conda activate vqa

echo [INFO] ---------------------------------------------
echo [INFO] 1. Starting Experiment: MFB
echo [INFO] ---------------------------------------------
python train.py -c ./cfg/ResNet50_BERT_coco/mfb.yaml

echo [INFO] ---------------------------------------------
echo [INFO] 2. Starting Experiment: Co-Attention
echo [INFO] ---------------------------------------------
python train.py -c ./cfg/ResNet50_BERT_coco/coattention.yaml

echo [INFO] ---------------------------------------------
echo [INFO] 3. Starting Experiment: Concat
echo [INFO] ---------------------------------------------
python train.py -c ./cfg/ResNet50_BERT_coco/concat.yaml

echo [INFO] ---------------------------------------------
echo [INFO] 3. Starting Experiment: attention
echo [INFO] ---------------------------------------------
python train.py -c ./cfg/ResNet50_BERT_coco/attention.yaml

echo [INFO] ---------------------------------------------
echo [INFO] 3. Starting Experiment: gated_fusion
echo [INFO] ---------------------------------------------
python train.py -c ./cfg/ResNet50_BERT_coco/gated_fusion.yaml

echo [INFO] ---------------------------------------------
echo [INFO] All experiments finished!
echo [INFO] ---------------------------------------------

REM 창이 바로 닫히지 않도록 잠시 대기
pause