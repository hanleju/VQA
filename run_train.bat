@echo OFF
REM -----------------------------------------------------------------
REM VQA 모델 학습
REM -----------------------------------------------------------------

echo [INFO] Activating Conda environment 'vqa'...

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

pause