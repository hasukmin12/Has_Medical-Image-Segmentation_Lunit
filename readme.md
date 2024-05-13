# 간략한 사용법

1. Train
```bash
train.py
```
config 폴더에 새롭게 할당할 config 파일을 생성하고 train.py의 argument에 적용

이때 PAIP 대회에서 3등했던 config 파일은

```bash
config/PAIP_2023/bf/PAIP_2023_norm_main.py
```

&nbsp;
&nbsp;
&nbsp;


2. Test
```bash
predict_for_visualize_Test_StainNorm.py
```
argument로 train과 마찬가지로 목표하는 config를 넣고, 모델 경로, inference 할 이미지 경로 설정 후 run

&nbsp;
&nbsp;
&nbsp;


3. model 설명
```bash
core/core.py
```
여기에서 주요 코드가 돌아가며 call_model.py에서 원하는 모델을 불러올 수 있다.
PAIP 대회에서 3등했던 모델을 불러오려면 config의 "MODEL_NAME"에 "CacoX"를 넣어주면 자동으로 모델을 불러온다.