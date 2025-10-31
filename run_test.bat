@echo OFF
REM -----------------------------------------------------------------
REM VQA 모델 테스트
REM -----------------------------------------------------------------

echo [INFO] Activating Conda environment 'vqa'...

call conda activate vqa


@REM COCO Dataset
echo [INFO] ---------------------------------------------
echo [INFO] 1. Starting Experiment: MFB
echo [INFO] ---------------------------------------------
python test.py -c ./cfg/ResNet50_BERT_coco/mfb.yaml -w ./checkpoints/ResNet50_BERT_coco/mfb/best_model_epoch_18_acc_64.98.pth

echo [INFO] ---------------------------------------------
echo [INFO] 2. Starting Experiment: Co-Attention
echo [INFO] ---------------------------------------------
python test.py -c ./cfg/ResNet50_BERT_coco/coattention.yaml -w ./checkpoints/ResNet50_BERT_coco/coattention/best_model_epoch_16_acc_66.57.pth

echo [INFO] ---------------------------------------------
echo [INFO] 3. Starting Experiment: Concat
echo [INFO] ---------------------------------------------
python test.py -c ./cfg/ResNet50_BERT_coco/concat.yaml -w ./checkpoints/ResNet50_BERT_coco/concat/best_model_epoch_14_acc_66.42.pth

@REM echo [INFO] ---------------------------------------------
@REM echo [INFO] 4. Starting Experiment: attention
@REM echo [INFO] ---------------------------------------------
@REM python test.py -c ./cfg/ResNet50_BERT_coco/attention.yaml -w ./checkpoints/ResNet50_BERT_coco/attention/best_model_epoch_13_acc_65.20.pth

@REM echo [INFO] ---------------------------------------------
@REM echo [INFO] 5. Starting Experiment: gated_fusion
@REM echo [INFO] ---------------------------------------------
@REM python test.py -c ./cfg/ResNet50_BERT_coco/gated_fusion.yaml  -w ./checkpoints/ResNet50_BERT_coco/gated_fusion/best_model_epoch_9_acc_66.82.pth





@REM easyVQA Dataset
@REM echo [INFO] ---------------------------------------------
@REM echo [INFO] 1. Starting Experiment: attention
@REM echo [INFO] ---------------------------------------------
@REM python test.py -c ./cfg/ResNet50_BERT_easy/attention.yaml -w ./checkpoints/ResNet50_BERT_easy/attention/best_model_epoch_5_acc_100.00.pth

@REM echo [INFO] ---------------------------------------------
@REM echo [INFO] 2. Starting Experiment: Co-Attention
@REM echo [INFO] ---------------------------------------------
@REM python test.py -c ./cfg/ResNet50_BERT_easy/coattention.yaml -w ./checkpoints/ResNet50_BERT_easy/coattention/best_model_epoch_4_acc_100.00.pth

@REM echo [INFO] ---------------------------------------------
@REM echo [INFO] 3. Starting Experiment: Concat
@REM echo [INFO] ---------------------------------------------
@REM python test.py -c ./cfg/ResNet50_BERT_easy/concat.yaml -w ./checkpoints/ResNet50_BERT_easy/concat/best_model_epoch_4_acc_100.00.pth

@REM echo [INFO] ---------------------------------------------
@REM echo [INFO] 4. Starting Experiment: gated_fusion
@REM echo [INFO] ---------------------------------------------
@REM python test.py -c ./cfg/ResNet50_BERT_easy/gated_fusion.yaml  -w ./checkpoints/ResNet50_BERT_easy/gated_fusion/best_model_epoch_5_acc_94.89.pth

@REM echo [INFO] ---------------------------------------------
@REM echo [INFO] 5. Starting Experiment: MFB
@REM echo [INFO] ---------------------------------------------
@REM python test.py -c ./cfg/ResNet50_BERT_easy/mfb.yaml -w ./checkpoints/ResNet50_BERT_easy/mfb/best_model_epoch_5_acc_100.00.pth


echo [INFO] ---------------------------------------------
echo [INFO] All experiments finished!
echo [INFO] ---------------------------------------------

REM
pause