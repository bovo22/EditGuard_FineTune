# EditGuard Fine-tune
COCO2017 이미지에 대해 EditGuard 모델을 파인튜닝한 레포지토리

워터마크 삽입∙복원 성능을 유지하면서 Coverr PSNR향상을 목표로 함
---
- 파인튜닝용 옵션 파일

/mnt/ssd/BG/EditGuard/EditGuard_git/code/options/train_editguard_image.yml

/mnt/ssd/BG/EditGuard/EditGuard_git/code/options/test_editguard_finetune.yml

- test파일

/mnt/ssd/BG/EditGuard/EditGuard_git/code/test.py
---

## 개발 환경 구성

+ Python 3.10 이상

+ CUDA 사용 가능 GPU (테스트 환경: CUDA 11.x)

