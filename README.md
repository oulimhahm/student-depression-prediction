# 대학생 우울증 발병 예측 모델

생활습관 및 SNS 텍스트 데이터 기반 XGBoost 우울증 예측 시스템

---

## 개요

최근 5년간 청년 우울증 발병률이 30% 이상 증가하였으나, 증상을 인지하지 못하는 대학생 비율이 60%에 달하며 전문가 상담을 기피하는 비율도 1/3에 이른다. 본 연구는 생활습관 정형 데이터와 SNS 비정형 텍스트 데이터를 융합하여 대학생 우울증 발병 여부를 예측하는 머신러닝 모델을 제안한다.

단순 예측 수치를 넘어 SHAP으로 피처 기여도를 정량화하고, ipywidgets 기반 실시간 시뮬레이터를 구현하여 실제 정신건강 개입에 활용 가능한 시스템을 구축하였다.

---

## 파이프라인

정형 데이터(Kaggle) + 비정형 데이터(Twitter) → EDA → VADER 감성 분석 → SMOTE 오버샘플링 → LR / RF / XGBoost → SHAP 해석 + 실시간 시뮬레이터

---

## 주요 특징

- 정형 생활습관 데이터(27,901명, 17개 피처)와 Twitter NLP 감성 분석 결과를 융합한 이중 데이터셋 구성
- VADER 감성 분석으로 감성 점수, 키워드 카운트 등 3개의 새로운 예측 피처 생성
- SMOTE를 활용한 클래스 불균형 처리
- Optuna 베이지안 최적화(30 trials)로 Recall 극대화
- SHAP으로 각 피처의 기여도 정량화 및 모델 해석가능성 확보
- ipywidgets 슬라이더 기반 실시간 What-If 시뮬레이터 (6개 변수)
- Recall을 핵심 지표로 설정 — 고위험군 미탐지(FN)가 오탐지(FP)보다 임상적으로 위험

---

## 실험 결과

최종 모델: XGBoost (Optuna 튜닝)

| 지표 | 값 |
|------|-----|
| AUC | 0.9183 |
| Recall | 0.8788 |

- 핵심 예측 요인 (SHAP 기준): 자살충동 경험 > 학업 스트레스 > 재정 스트레스
- 수면 5시간 미만 시 위험도 급증 (수면이 보호 요인으로 작용)
- 재정 스트레스 5단계 시 임계값 0.5 초과

---

## 데이터셋

본 저장소에는 데이터셋이 포함되지 않는다. 아래 공식 출처에서 직접 다운로드:

- 정형 데이터: Student Depression Dataset (Kaggle, adilshamim8)  
  https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset
- 비정형 데이터: Mental Health Social Media Dataset (Kaggle, infamouscoder)
  https://www.kaggle.com/datasets/infamouscoder/mental-health-social-media
---

## 저장소 구조

```
student-depression-prediction/
├── README.md
├── main.ipynb          # 전체 실험 코드 (ipywidgets 실시간 시뮬레이터 포함)
└── model.pkl    
```

---

## 실험 환경

- Google Colab
- Python 3.10
- scikit-learn, XGBoost, SHAP, Optuna
- VADER (vaderSentiment)
- imbalanced-learn (SMOTE)
- ipywidgets

---

## 활용 방안

- 대학 상담센터: 고위험군 학생 우선 상담 대상 조기 선별 도구
- 공공 정신건강 정책: 청년 우울증 예방 프로그램 설계 근거 자료
- 헬스케어 앱: 정신건강 자가진단 알고리즘 기반으로 발전 가능

한계: 자가보고 방식의 주관적 편향 가능성 존재. 실제 임상 진단의 대체가 아닌 보조 도구로 한정.

---

## 참고 문헌

- 국회 교육위원회 대학생 마음건강 실태조사 (2024)
- 한국대학교육협의회 대학별 실태 보고 (2025)
- Hutto, C., Gilbert, E.: VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. ICWSM (2014)
- Chen, T., Guestrin, C.: XGBoost: A Scalable Tree Boosting System. KDD (2016)
- Lundberg, S., Lee, S.: A Unified Approach to Interpreting Model Predictions. NeurIPS (2017)
- Akiba, T., et al.: Optuna: A Next-generation Hyperparameter Optimization Framework. KDD (2019)
