<p align="center">
  <img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white" />
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/scikitlearn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/matplotlib-3F4F75?style=for-the-badge&logo=matplotlib&logoColor=white" />
  <img src="https://img.shields.io/badge/seaborn-4EABE6?style=for-the-badge&logo=seaborn&logoColor=white" />
  <img src="https://img.shields.io/badge/streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/joblib-2E6BB1?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/RandomForest-47A248?style=for-the-badge&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/VSCode-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white" />
</p>



# ML_1st_Prj (머신러닝 첫 번째 프로젝트)

## 프로젝트 개요
이 레포지토리는 머신러닝 학습 및 실험을 위한 첫 번째 프로젝트입니다. 다양한 머신러닝 알고리즘과 기법을 실습하고 결과를 분석하는 과정을 담고 있습니다.

## 개발기간
2025/03/18 ~ 2025/03/19

## 목표
- 기본적인 머신러닝 개념 이해 및 적용
- 데이터 전처리, 모델 학습, 평가 과정 경험
- 실제 데이터셋을 활용한 문제 해결 능력 향상
- 머신러닝 프로젝트 개발 및 관리 방법 습득

## 사용 기술
- Python 3.x
- 주요 라이브러리:
  - scikit-learn
  - pandas
  - numpy
  - matplotlib / seaborn

## 프로젝트 구조
ML_1st_Prj/
│
├── dataset/               # 데이터셋 저장 디렉토리
│   └── E.csv              # E-commerce 고객 데이터
│
├── .venv/                 # 가상환경 디렉토리
│
├── E_commerce_page.py     # 메인 Streamlit 애플리케이션 파일
├── E_commerce_report.ipynb# 데이터 분석 및 모델 개발 노트북
│
├── preprocessor.pkl       # 전처리 모델
├── requirements.txt       # 필요 패키지 목록
│
├── customer_rating_model.pkl  # 고객 평점 예측 모델(회귀)
├── satisfaction_model.pkl     # 고객 만족도 예측 모델(분류)
├── scaler_reg.pkl             # 회귀 모델용 데이터 스케일러
├── scaler_clf.pkl             # 분류 모델용 데이터 스케일러
│
└── README.md              # 프로젝트 설명 문서


## 주요 기능
총 구매 금액, 구매한 아이템 수, 마지막 구매 이후 일수를 입력하여 고객 평점 예측
동일한 입력을 통해 고객 만족도 예측
모델 성능 지표 확인 (MSE, RMSE, R², Accuracy, Precision, Recall, F1 Score)
특성 중요도 시각화

## 시작하기

### 필수 요구사항
- Python 3.8 이상
- pip 또는 conda 환경 관리자

### 설치 및 실행 방법
```bash
# 레포지토리 클론
git clone https://github.com/ssuuoo12/ML_1st_Prj.git
cd ML_1st_Prj

# 가상환경 생성 및 활성화 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필요한 패키지 설치
pip install -r requirements.txt

# Streamlit 앱 실행
streamlit run E_commerce_page.py
