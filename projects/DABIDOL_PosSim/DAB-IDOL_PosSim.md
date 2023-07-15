## DAB-IDOL BASE 

* * *

### Training

To train DAB-IDOL_PosSim(ResNet50) on YouTube-VIS 2019 with 3 GPUs , run:

```
python3 projects/DABIDOL_PosSim/train_net.py --config-file projects/DABIDOL_PosSim/configs/ytvis19_r50.yaml --num-gpus 3 MODEL.WEIGHTS weights/cocopretrain_R50.pth SOLVER.IMS_PER_BATCH 12 
```
- 학습이 완료되면 Detectron2 코드에 의해 최종 checkpoint 파일을 이용해서 자동으로 inference를 진행하며, 최종적으로 output 디렉토리에 checkpoint file, log txt, inference(results.json) 등이 모두 저장됨



### Inference & Evaluation



Evaluating on YouTube-VIS 2019 or OVIS:

```
python3 projects/DABIDOL_PosSim/train_net.py --config-file projects/DABIDOL_PosSim/configs/XXX.yaml --num-gpus 3 --eval-only
```
- To get quantitative results, please zip the json file and upload to the [codalab server](https://codalab.lisn.upsaclay.fr/competitions/6064#participate-submit_results) for YouTube-VIS 2019 and ~~[server](https://google.com)~~ for OVIS.
- 위치 유사도 계산 시 사용 파라미터 반드시 확인한 후 진행할 것



