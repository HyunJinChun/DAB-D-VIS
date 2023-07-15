## Installation

#### 1. 가상환경 설정 :

```bash
conda create -n vnext python=3.9
conda activate vnext
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

#### 2. Clone the repository locally or Download:

```bash
git clone https://github.com/wjf5203/VNext.git (or Download this repository)
cd VNext
```

#### 3. Install dependencies and pycocotools for VIS:

```bash
pip install -r requirements.txt
pip install -e .
pip install shapely==1.7.1
pip install git+https://github.com/youtubevos/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
```

#### 4. Compiling Deformable DETR CUDA operators :

```bash
cd projects/IDOL/idol/models/ops/
sh make.sh
```

#### 5. Error : No module named 'detectron2.projects'
```bash
pip install -e .
pip install git+https://github.com/youtubevos/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
```
- projects에 새로운 모델(ex. DABIDOL_***)을 추가할 때마다 해줘야 해당 오류가 발생하지 않음
  - 가상환경의 detectron2에 수정된 구조를 반영해야 함


* * *


## Data Preparation



Download and extract 2019 version of YoutubeVIS train and val images with annotations from [CodeLab](https://competitions.codalab.org/competitions/20128#participate-get_data) or [YouTubeVIS](https://youtube-vos.org/dataset/vis/), download [OVIS](https://codalab.lisn.upsaclay.fr/competitions/4763#participate)  and COCO 2017 datasets. Then, link datasets:

```bash
cd datasets/
ln -s /path_to_coco_dataset coco
ln -s /path_to_YTVIS19_dataset ytvis_2019
ln -s /path_to_ovis_dataset ovis
```
- Example : ln -s ../../../seqformer_ws/coco/ coco


Extract YouTube-VIS 2019, OVIS, COCO 2017 datasets, we expect the directory structure to be the following:

```
VNext
├── datasets
│   ├──ytvis_2019
│   ├──ovis 
│   ├──coco 
...
ytvis_2019
├── train
├── val
├── annotations
│   ├── instances_train_sub.json
│   ├── instances_val_sub.json
...
ovis
├── train
├── valid
├── annotations_train.json
├── annotations_valid.jso
...
coco
├── train2017
├── val2017
├── annotations
│   ├── instances_train2017.json
│   ├── instances_val2017.json
```


