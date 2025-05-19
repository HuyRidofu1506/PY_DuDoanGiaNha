import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import shutil

# Đặt tiêu đề cho ứng dụng
st.set_page_config(page_title="Ứng dụng dự đoán giá nhà", layout="wide")

# Tiêu đề ứng dụng
st.title("Ứng dụng dự đoán giá nhà sử dụng mô hình dự đoán tốt nhất")
st.markdown("---")

# Tạo layout với 2 cột
col1, col2 = st.columns([1, 2])

# Hàm tính MAPE accuracy
def mean_absolute_percentage_accuracy(y_true, y_pred):
    """
    Tính độ chính xác phần trăm dựa trên MAPE
    Công thức: 100% - (trung bình tỷ lệ sai số tuyệt đối)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Kiểm tra để tránh chia cho 0
    if np.any(y_true == 0):
        print("Cảnh báo: Giá trị thực tế có chứa 0, loại bỏ các mẫu này để tính toán")
        mask = y_true != 0
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if len(y_true) == 0:
            return 0.0
    
    # Tính MAPE và chuyển thành độ chính xác
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    accuracy = 100 - mape
    
    return accuracy

# Đào tạo các mô hình mới để tránh các vấn đề không tương thích phiên bản
def force_retrain_models():
    model_paths = ['linear_model.pkl', 'random_forest_model.pkl']
    export_dir = "exported_model"
    
    # Xóa các tập tin mô hình cũ để đảm bảo huấn luyện mới
    for path in model_paths:
        if os.path.exists(path):
            os.remove(path)
    
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)

# Kiểm tra và tải mô hình Linear Regression
def load_linear_model():
    model_path = 'linear_model.pkl'
    
    # Luôn đào tạo một mô hình mới để tránh các vấn đề về phiên bản
    try:
        df = pd.read_csv("data_gia.csv")
        X = df[['Diện tích (m2)', 'Số phòng ngủ', 'Số phòng tắm', 'Năm xây dựng', 'Vị trí (Mã vùng)']]
        y = df['Giá bán (VNĐ)']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Lưu mô hình
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return model
    except Exception as e:
        st.error(f"Lỗi khi huấn luyện mô hình Linear Regression: {str(e)}")
        # Trả về một mô hình đơn giản nếu có lỗi
        model = LinearRegression()
        return model

# Kiểm tra và tải mô hình XGBoost
def load_xgboost_model():
    model_path = "exported_model/xgboost_model.json"
    
    # Luôn đào tạo một mô hình mới để tránh các vấn đề về phiên bản
    try:
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        df = pd.read_csv("data_gia.csv")
        X = df[['Diện tích (m2)', 'Số phòng ngủ', 'Số phòng tắm', 'Năm xây dựng', 'Vị trí (Mã vùng)']]
        y = df['Giá bán (VNĐ)']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = xgb.XGBRegressor(
            objective='reg:squarederror', 
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=4, 
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Lưu mô hình
        model.save_model(model_path)
        
        return model
    except Exception as e:
        st.error(f"Lỗi khi huấn luyện mô hình XGBoost: {str(e)}")
        # Trả về một mô hình đơn giản nếu có lỗi
        model = xgb.XGBRegressor()
        return model

# Kiểm tra và tải mô hình Random Forest
def load_random_forest_model():
    model_path = 'random_forest_model.pkl'
    
    # Luôn đào tạo một mô hình mới để tránh các vấn đề về phiên bản
    try:
        df = pd.read_csv("data_gia.csv")
        X = df[['Diện tích (m2)', 'Số phòng ngủ', 'Số phòng tắm', 'Năm xây dựng', 'Vị trí (Mã vùng)']]
        y = df['Giá bán (VNĐ)']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Lưu mô hình
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return model
    except Exception as e:
        st.error(f"Lỗi khi huấn luyện mô hình Random Forest: {str(e)}")
        # Trả về một mô hình đơn giản nếu có lỗi
        model = RandomForestRegressor()
        return model

# Đánh giá và chọn mô hình tốt nhất
@st.cache_data
def evaluate_and_select_best_model():
    # Đào tạo lại các mô hình để tránh các vấn đề về khả năng tương thích phiên bản
    force_retrain_models()
    
    try:
        # Tải dữ liệu
        df = pd.read_csv("data_gia.csv")
        X = df[['Diện tích (m2)', 'Số phòng ngủ', 'Số phòng tắm', 'Năm xây dựng', 'Vị trí (Mã vùng)']]
        y = df['Giá bán (VNĐ)']
        
        # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Tải các mô hình
        linear_model = load_linear_model()
        xgboost_model = load_xgboost_model()
        random_forest_model = load_random_forest_model()
        
        # Dự đoán trên tập kiểm tra
        linear_pred = linear_model.predict(X_test)
        xgboost_pred = xgboost_model.predict(X_test)
        rf_pred = random_forest_model.predict(X_test)
        
        # Đánh giá độ chính xác của từng mô hình
        linear_accuracy = mean_absolute_percentage_accuracy(y_test, linear_pred)
        xgboost_accuracy = mean_absolute_percentage_accuracy(y_test, xgboost_pred)
        rf_accuracy = mean_absolute_percentage_accuracy(y_test, rf_pred)
        
        # Tạo từ điển chứa thông tin về mô hình và độ chính xác
        model_accuracies = {
            "Linear Regression": {"model": linear_model, "accuracy": linear_accuracy},
            "XGBoost": {"model": xgboost_model, "accuracy": xgboost_accuracy},
            "Random Forest": {"model": random_forest_model, "accuracy": rf_accuracy}
        }
        
        # Tìm mô hình có độ chính xác cao nhất
        best_model_name = max(model_accuracies, key=lambda x: model_accuracies[x]["accuracy"])
        best_model = model_accuracies[best_model_name]["model"]
        best_accuracy = model_accuracies[best_model_name]["accuracy"]
        
        return {
            "best_model_name": best_model_name,
            "best_model": best_model,
            "best_accuracy": best_accuracy,
            "all_accuracies": model_accuracies
        }
    except Exception as e:
        st.error(f"Lỗi khi đánh giá mô hình: {str(e)}")
        return {"best_model_name": "Linear Regression", "best_model": load_linear_model(), "best_accuracy": 0}

# Sidebar cho nhập thông tin nhà
with col1:
    st.header("Thông tin nhà")
    
    # Tạo form nhập liệu
    with st.form(key="house_form"):
        dien_tich = st.slider("Diện tích (m²)", min_value=30, max_value=500, value=120, step=10)
        so_phong_ngu = st.slider("Số phòng ngủ", min_value=1, max_value=10, value=3)
        so_phong_tam = st.slider("Số phòng tắm", min_value=1, max_value=10, value=2)
        nam_xay_dung = st.slider("Năm xây dựng", min_value=1950, max_value=2023, value=2000)
        vi_tri_ma_vung = st.selectbox("Vị trí (Mã vùng)", options=[55106, 55107, 55109, 55110, 55111, 55123, 55126, 55156, 55160, 55199])
        
        # Nút dự đoán
        predict_button = st.form_submit_button(label="Dự đoán giá")

# Hiển thị kết quả dự đoán
with col2:
    try:
        # Đánh giá và chọn mô hình tốt nhất
        with st.spinner("Đang huấn luyện và đánh giá mô hình..."):
            best_model_info = evaluate_and_select_best_model()
            best_model_name = best_model_info["best_model_name"]
            best_model = best_model_info["best_model"]
            best_accuracy = best_model_info["best_accuracy"]
            all_accuracies = best_model_info.get("all_accuracies", {})
        
        # Hiển thị thông tin về độ chính xác của các mô hình
        st.header("So sánh độ chính xác của các mô hình")
        
        # Tạo DataFrame để hiển thị độ chính xác
        accuracy_data = []
        for model_name, info in all_accuracies.items():
            accuracy_data.append({
                "Mô hình": model_name,
                "Độ chính xác (%)": round(info["accuracy"], 2)
            })
        
        accuracy_df = pd.DataFrame(accuracy_data)
        
        # Hiển thị bảng độ chính xác
        st.dataframe(accuracy_df, use_container_width=True)
        
        # Hiển thị thông tin về mô hình tốt nhất
        st.success(f"Mô hình tốt nhất: {best_model_name} với độ chính xác {best_accuracy:.2f}%")
        
        # Vẽ biểu đồ so sánh độ chính xác
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Lấy dữ liệu cho biểu đồ
        models = list(all_accuracies.keys())
        accuracies = [all_accuracies[model]["accuracy"] for model in models]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        
        # Tạo biểu đồ cột
        bars = ax.bar(models, accuracies, color=colors)
        
        # Thêm giá trị trên đỉnh cột
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', rotation=0)
        
        ax.set_ylabel("Độ chính xác (%)")
        ax.set_title("So sánh độ chính xác giữa ba mô hình")
        
        # Hiển thị biểu đồ
        st.pyplot(fig)
        
        if predict_button:
            st.header("Kết quả dự đoán")
            
            # Tạo DataFrame từ thông tin nhập
            new_house = pd.DataFrame({
                'Diện tích (m2)': [dien_tich],
                'Số phòng ngủ': [so_phong_ngu],
                'Số phòng tắm': [so_phong_tam],
                'Năm xây dựng': [nam_xay_dung],
                'Vị trí (Mã vùng)': [vi_tri_ma_vung]
            })
            
            # Dự đoán với mô hình tốt nhất
            best_pred = best_model.predict(new_house)[0]
            
            # Hiển thị kết quả
            st.metric(
                label=f"Giá dự đoán ({best_model_name})", 
                value=f"{best_pred:,.0f} VNĐ",
                delta=f"Độ chính xác: {best_accuracy:.2f}%"
            )
            
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {str(e)}")

# Footer
st.markdown("---")
st.caption("© 2023 Ứng dụng dự đoán giá nhà | Sử dụng mô hình Machine Learning tốt nhất") 