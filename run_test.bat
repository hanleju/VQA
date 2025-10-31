@echo OFF
REM -----------------------------------------------------------------
REM 이 스크립트는 여러 VQA 모델을 순차적으로 학습시킵니다.
REM -----------------------------------------------------------------

echo [INFO] Activating Conda environment 'vqa'...
REM (vqa) 환경을 활성화합니다. Conda/Anaconda를 사용하지 않는다면 이 줄을 지우거나 수정하세요.
call conda activate vqa

@REM echo [INFO] ---------------------------------------------
@REM echo [INFO] 1. Starting Experiment: MFB
@REM echo [INFO] ---------------------------------------------
@REM python test.py -c ./cfg/ResNet50_BERT_coco/mfb.yaml -w ./checkpoints/ResNet50_BERT_coco/mfb/best_model_epoch_18_acc_64.98.pth

@REM echo [INFO] ---------------------------------------------
@REM echo [INFO] 2. Starting Experiment: Co-Attention
@REM echo [INFO] ---------------------------------------------
@REM python test.py -c ./cfg/ResNet50_BERT_coco/co_attention.yaml -w ./checkpoints/ResNet50_BERT_coco/

@REM echo [INFO] ---------------------------------------------
@REM echo [INFO] 3. Starting Experiment: Concat
@REM echo [INFO] ---------------------------------------------
@REM python test.py -c ./cfg/ResNet50_BERT_coco/concat.yaml -w ./checkpoints/ResNet50_BERT_coco/concat/best_model_epoch_14_acc_66.42.pth

@REM echo [INFO] ---------------------------------------------
@REM echo [INFO] 3. Starting Experiment: attention
@REM echo [INFO] ---------------------------------------------
@REM python test.py -c ./cfg/ResNet50_BERT_coco/attention.yaml -w ./checkpoints/ResNet50_BERT_coco/attention/best_model_epoch_13_acc_65.20

echo [INFO] ---------------------------------------------
echo [INFO] 3. Starting Experiment: gated_fusion
echo [INFO] ---------------------------------------------
python test.py -c ./cfg/ResNet50_BERT_coco/gated_fusion.yaml  -w ./checkpoints/ResNet50_BERT_coco/gated_fusion/best_model_epoch_9_acc_66.82.pth

echo [INFO] ---------------------------------------------
echo [INFO] All experiments finished!
echo [INFO] ---------------------------------------------

REM 창이 바로 닫히지 않도록 잠시 대기
pause