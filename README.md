# EditGuard Fine-tune

COCO2017 이미지에 대해 EditGuard 모델을 파인튜닝한 레포지토리이다.  
워터마크 삽입∙복원 성능을 유지하면서 **Cover PSNR 향상**을 목표로 한다.

---

## 주요 파일

- 파인튜닝용 옵션 파일  
  - `code/options/train_editguard_image.yml`  
  - `code/options/test_editguard_finetune.yml`
- 테스트 스크립트  
  - `code/test.py`

---

## 개발 환경

- Python 3.10 이상  
- CUDA 사용 가능 GPU (테스트 환경: CUDA 11.x)

---

## 디렉터리 구조

```text
EditGuard/
├─ code/
│  ├─ train.py
│  ├─ test.py
│  └─ models/, data/, ...
├─ options/
│  ├─ train_editguard_image.yml
│  ├─ test_editguard_finetune.yml
│  └─ test_editguard_visual.yml
├─ checkpoints/
│  ├─ clean.pth
│  └─ 5000_G.pth
├─ results/               # test.py 실행 결과 이미지/로그
├─ requirements.txt
└─ README.md

## 데이터셋
- 학습, 평가는 COCO2017 기준으로 함
- train2017: COCO2017 전체 dataset 사용
- val2017: COCO2017에서 500개의 sample shuffle
- sample2017: 이미지 복원 정도 확인용 mini sample

## Train
```
cd code
source ../venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

nohup python3 train.py \
  -opt ../options/train_editguard_image.yml \ 
  --launcher none \
  > ../logs/train_psnr.log 2>&1 
```

## Test 

```
cd code
source ../venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

python3 test.py \
  -opt ../options/test_editguard_finetune.yml \ 
  --ckpt ../checkpoints/5000_G.pth \ 
  --launcher none
```

## License
EditGuard 구현체를 기반으로 함


