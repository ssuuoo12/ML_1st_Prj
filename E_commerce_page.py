import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


# 폰트 지정 (한글 지원)
plt.rcParams['font.family'] = 'Malgun Gothic'

# Streamlit 캐싱을 이용한 데이터 로딩
@st.cache_data
def load_data():
    df = pd.read_csv('dataset/E.csv')
    # df['Discount Applied'] = df['Discount Applied'].replace({'TRUE': 1, 'FALSE': 0})
    df = df.dropna()  # 결측값 제거
    return df

# 데이터 불러오기
data = load_data()

# 특성 선택
selected_features = ['Total Spend', 'Items Purchased', 'Days Since Last Purchase']
X = data[selected_features]

# 회귀 타겟: Average Rating
y_regression = data['Average Rating']

# 분류 타겟: Satisfaction Level
y_classification = data['Satisfaction Level']

# 데이터 분할 (회귀)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)

# 데이터 분할 (분류)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_classification, test_size=0.2, random_state=42)

# 데이터 스케일링 (회귀)
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# 데이터 스케일링 (분류)
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

# 모델 학습 및 저장
model_reg = RandomForestRegressor(n_estimators=25, random_state=42)
model_reg.fit(X_train_reg_scaled, y_train_reg)
joblib.dump(model_reg, 'customer_rating_model.pkl')
joblib.dump(scaler_reg, 'scaler_reg.pkl')

model_clf = RandomForestClassifier(n_estimators=25, random_state=42)
model_clf.fit(X_train_clf_scaled, y_train_clf)
joblib.dump(model_clf, 'satisfaction_model.pkl')
joblib.dump(scaler_clf, 'scaler_clf.pkl')

# Streamlit UI 구현
st.title('고객 평점 및 만족도 예측 시스템')
st.write('총 구매 금액, 구매한 아이템 수, 마지막 구매 이후 일수를 입력하여 고객 평점과 만족도를 예측해보세요.')

# 사용자 입력
total_spend = st.slider('총 지출액 (Total Spend)', min_value=400.0, max_value=1600.0, value=1000.0, step=10.0)
items_purchased = st.slider('구매한 아이템 수 (Items Purchased)', min_value=7, max_value=21, value=10, step=1)
days_since_last_purchase = st.slider('마지막 구매 이후 일수 (Days Since Last Purchase)', min_value=9, max_value=63, value=20, step=1)

# 예측 버튼
if st.button('예측하기'):
    # 모델 및 스케일러 로드
    model_reg = joblib.load('customer_rating_model.pkl')
    scaler_reg = joblib.load('scaler_reg.pkl')
    model_clf = joblib.load('satisfaction_model.pkl')
    scaler_clf = joblib.load('scaler_clf.pkl')

    # 입력값 변환
    input_data = np.array([[total_spend, items_purchased, days_since_last_purchase]])
    input_data_reg_scaled = scaler_reg.transform(input_data)
    input_data_clf_scaled = scaler_clf.transform(input_data)

    # 예측 수행
    prediction_reg = model_reg.predict(input_data_reg_scaled)[0]  # Average Rating
    prediction_clf = model_clf.predict(input_data_clf_scaled)[0]  # Satisfaction Level

    # 모델 평가 (회귀)
    y_pred_reg = model_reg.predict(X_test_reg_scaled)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_reg, y_pred_reg)

    # 모델 평가 (분류)
    y_pred_clf = model_clf.predict(X_test_clf_scaled)
    accuracy = accuracy_score(y_test_clf, y_pred_clf)
    precision = precision_score(y_test_clf, y_pred_clf, average='weighted', zero_division=1)
    recall = recall_score(y_test_clf, y_pred_clf, average='weighted', zero_division=1)
    f1 = f1_score(y_test_clf, y_pred_clf, average='weighted')
    # average='weighted': 분류시 가중 평균 처리 , zero_division=1 : 정밀도 계산시 예외 처리 
    
    # 결과 출력
    st.success(f'예측된 고객 평점: {prediction_reg:.2f} (0~5 사이) | 예측된 만족도: {prediction_clf}')

    st.subheader("회귀 모델 평가 지표")
    st.write(f'평균 제곱 오차 (MSE): {mse:.4f}')
    st.write(f'평균 제곱근 오차 (RMSE): {rmse:.2f}')
    st.write(f'결정 계수 (R²): {r2:.2f}')

    st.subheader("분류 모델 평가 지표")
    st.write(f'정확도 (Accuracy): {accuracy*100:.2f}%')
    st.write(f'정밀도 (Precision): {precision:.2f}')
    st.write(f'재현율 (Recall): {recall:.2f}')
    st.write(f'F1 Score: {f1:.2f}')

# 특성 중요도 시각화 (회귀 모델)
st.subheader('평점 , 만족도 예측 모델의 특성 중요도')
col1, col2 = st.columns(2)  # 두 열 생성

# 색상 팔레트 지정
palette = sns.color_palette("viridis")

# 평점 예측 모델 그래프
with col1:
    feature_importances_reg = model_reg.feature_importances_
    fig_reg, ax_reg = plt.subplots(figsize=(6, 4))  # 크기 조절
    ax_reg.bar(selected_features, feature_importances_reg , color=palette)
    ax_reg.set_xlabel('특성')
    ax_reg.set_ylabel('중요도')
    ax_reg.set_title('평점 예측 모델')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig_reg)

# 만족도 예측 모델 그래프
with col2:
    feature_importances_clf = model_clf.feature_importances_
    fig_clf, ax_clf = plt.subplots(figsize=(6, 4))  # 크기 조절
    ax_clf.bar(selected_features, feature_importances_clf, color=palette)
    ax_clf.set_xlabel('특성')
    ax_clf.set_ylabel('중요도')
    ax_clf.set_title('만족도 예측 모델')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig_clf)