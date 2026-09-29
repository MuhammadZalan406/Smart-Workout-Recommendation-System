import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from PIL import Image
import io
import base64

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Smart Workout Recommender 🏋️",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }
    
    /* Cards */
    .css-1r6slb0 {
        background: rgba(255,255,255,0.05) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    }
    
    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
    }
    
    /* Metrics */
    .css-1xarl3l {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    }
    
    /* Dataframes */
    .dataframe {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: white !important;
    }
    
    .dataframe th {
        background: rgba(102, 126, 234, 0.2) !important;
        color: white !important;
    }
    
    .dataframe td {
        color: white !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(15, 12, 41, 0.9) !important;
        backdrop-filter: blur(10px);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.05);
        border-radius: 50px;
        padding: 0.5rem 1.5rem;
        color: white;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'df_encoded' not in st.session_state:
    st.session_state.df_encoded = None
if 'df_fe' not in st.session_state:
    st.session_state.df_fe = None
if 'df_selected' not in st.session_state:
    st.session_state.df_selected = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/dumbbell.png", width=80)
    st.title("🏋️ Workout AI")
    st.markdown("---")
    
    # Step Progress
    st.markdown("### 📊 Pipeline Progress")
    steps = [
        "📂 Data Loading",
        "📊 EDA",
        "🧹 Data Cleaning",
        "🔢 Encoding",
        "⚙️ Feature Engineering",
        "🎯 Feature Selection",
        "📊 Train/Test Split",
        "🤖 Model Training",
        "📈 Model Evaluation",
        "🏆 Model Comparison",
        "🔮 Prediction"
    ]
    
    for i, step in enumerate(steps):
        if i < st.session_state.current_step:
            st.success(f"✅ {step}")
        elif i == st.session_state.current_step:
            st.info(f"🔄 {step}")
        else:
            st.caption(f"⏳ {step}")
    
    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit")

# ==================== MAIN CONTENT ====================
st.title("💪 Smart Workout Recommendation System")
st.markdown("*End-to-End Machine Learning Pipeline*")

# ==================== STEP 1: DATA LOADING ====================
with st.expander("📂 Step 1: Data Loading", expanded=st.session_state.current_step == 0):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
        
        if uploaded_file is not None:
            try:
                st.session_state.df = pd.read_csv(uploaded_file)
                st.success("✅ Data Loaded Successfully!")
                
                col1_1, col1_2, col1_3 = st.columns(3)
                with col1_1:
                    st.metric("Total Rows", st.session_state.df.shape[0])
                with col1_2:
                    st.metric("Total Columns", st.session_state.df.shape[1])
                with col1_3:
                    st.metric("Features", len(st.session_state.df.columns))
                
                # Sample Data
                st.subheader("📋 Dataset Sample")
                st.dataframe(st.session_state.df.head(10), use_container_width=True)
                
                # Data Overview
                st.subheader("📊 Data Overview")
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].bar(['Rows', 'Columns'], [st.session_state.df.shape[0], st.session_state.df.shape[1]], 
                           color=['#FF6B6B', '#4ECDC4'])
                axes[0].set_title('Dataset Dimensions', fontsize=14, fontweight='bold')
                axes[0].set_facecolor('none')
                
                col_types = st.session_state.df.dtypes.value_counts()
                axes[1].bar(col_types.index.astype(str), col_types.values, 
                           color=['#45B7D1', '#96CEB4'])
                axes[1].set_title('Data Types Distribution', fontsize=14, fontweight='bold')
                axes[1].set_facecolor('none')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                if st.button("Proceed to Step 2 ➡️", key="step1_btn"):
                    st.session_state.current_step = 1
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.info("👆 Please upload the smart_workout_raw_dataset.csv file")
            # Try to load from local
            if os.path.exists('smart_workout_raw_dataset.csv'):
                st.session_state.df = pd.read_csv('smart_workout_raw_dataset.csv')
                st.success("✅ Auto-loaded from local directory!")
                st.session_state.current_step = 1
                st.rerun()

# ==================== STEP 2: EDA ====================
if st.session_state.df is not None:
    with st.expander("📊 Step 2: Exploratory Data Analysis", expanded=st.session_state.current_step == 1):
        df = st.session_state.df
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Duplicate Rows", df.duplicated().sum())
        with col2:
            st.metric("Columns with Missing Values", df.isnull().sum().sum())
        
        # Statistical Summary
        st.subheader("📈 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Visualizations
        st.subheader("📊 Visualizations")
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Age Distribution
        plt.subplot(3, 3, 1)
        sns.histplot(df['age'].dropna(), bins=30, kde=True, color='#FF6B6B')
        plt.title('Age Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Age')
        
        # 2. Rating Distribution
        plt.subplot(3, 3, 2)
        sns.histplot(df['rating'], bins=20, kde=True, color='#4ECDC4')
        plt.title('Rating Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Rating')
        
        # 3. Fitness Goal
        plt.subplot(3, 3, 3)
        df['fitness_goal'].value_counts().head(10).plot(kind='bar', color='#45B7D1')
        plt.title('Top 10 Fitness Goals', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        
        # 4. Experience Level
        plt.subplot(3, 3, 4)
        df['experience_level'].value_counts().plot(kind='bar', color='#96CEB4')
        plt.title('Experience Level', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        
        # 5. Persona
        plt.subplot(3, 3, 5)
        df['persona'].value_counts().plot(kind='pie', autopct='%1.1f%%', 
                                         colors=['#FFEAA7', '#DDA0DD', '#87CEEB'])
        plt.title('Persona Distribution', fontsize=12, fontweight='bold')
        plt.ylabel('')
        
        # 6. Injury Constraint
        plt.subplot(3, 3, 6)
        df['injury_constraint'].value_counts().plot(kind='bar', color='#FFB347')
        plt.title('Injury Constraints', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        
        # 7. Body Part
        plt.subplot(3, 3, 7)
        df['bodyPart'].value_counts().head(10).plot(kind='barh', color='#6C5B7B')
        plt.title('Top 10 Body Parts', fontsize=12, fontweight='bold')
        
        # 8. Equipment
        plt.subplot(3, 3, 8)
        df['equipment'].value_counts().head(10).plot(kind='barh', color='#355C7D')
        plt.title('Top 10 Equipment', fontsize=12, fontweight='bold')
        
        # 9. Target Muscle
        plt.subplot(3, 3, 9)
        df['target_muscle'].value_counts().head(10).plot(kind='barh', color='#6B5B7B')
        plt.title('Top 10 Target Muscles', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Correlation Matrix
        st.subheader("📊 Correlation Matrix")
        fig2, ax = plt.subplots(figsize=(8, 6))
        corr_matrix = df[['age', 'rating']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
        st.pyplot(fig2)
        
        if st.button("Proceed to Step 3 ➡️", key="step2_btn"):
            st.session_state.current_step = 2
            st.rerun()

# ==================== STEP 3: DATA CLEANING ====================
if st.session_state.df is not None and st.session_state.current_step >= 2:
    with st.expander("🧹 Step 3: Data Cleaning", expanded=st.session_state.current_step == 2):
        df = st.session_state.df.copy()
        
        initial_rows = len(df)
        df = df.drop_duplicates()
        dup_removed = initial_rows - len(df)
        
        df = df[(df['age'] >= 10) & (df['age'] <= 100)]
        invalid_age = initial_rows - len(df) - dup_removed
        
        median_age = df['age'].median()
        df['age'] = df['age'].fillna(median_age)
        
        df = df[df['rating'] >= 1.0]
        invalid_rating = initial_rows - len(df) - dup_removed - invalid_age
        
        df['experience_level'] = df['experience_level'].str.lower()
        df['injury_constraint'] = df['injury_constraint'].fillna('none')
        df['equipment'] = df['equipment'].fillna('body weight')
        df['target_muscle'] = df['target_muscle'].fillna('general')
        
        st.session_state.df_clean = df
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Before Cleaning", initial_rows)
        with col2:
            st.metric("After Cleaning", len(df), delta=len(df)-initial_rows)
        with col3:
            st.metric("Duplicates Removed", dup_removed, delta=f"-{dup_removed}")
        with col4:
            st.metric("Invalid Rows", invalid_age + invalid_rating, delta=f"-{invalid_age + invalid_rating}")
        
        # Cleaning Summary
        st.subheader("📋 Cleaning Summary")
        st.dataframe(pd.DataFrame({
            'Action': ['Duplicates Removed', 'Invalid Age', 'Invalid Rating', 'Missing Values Filled'],
            'Count': [dup_removed, invalid_age, invalid_rating, df.isnull().sum().sum()]
        }), use_container_width=True)
        
        # Visualization
        st.subheader("📊 Cleaning Visualization")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].bar(['Before', 'After'], [initial_rows, len(df)], 
                   color=['#FF6B6B', '#4ECDC4'])
        axes[0].set_title('Data Size Before vs After', fontsize=12, fontweight='bold')
        axes[0].set_facecolor('none')
        
        missing_after = df.isnull().sum()
        axes[1].bar(missing_after.index, missing_after.values, color='#45B7D1')
        axes[1].set_title('Missing Values After Cleaning', fontsize=12, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].set_facecolor('none')
        
        axes[2].bar(['Duplicates', 'Invalid Age', 'Invalid Rating'], 
                   [dup_removed, invalid_age, invalid_rating], 
                   color=['#96CEB4', '#FFEAA7', '#FF6B6B'])
        axes[2].set_title('Cleaning Summary', fontsize=12, fontweight='bold')
        axes[2].set_facecolor('none')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Cleaned Data Sample
        st.subheader("📋 Cleaned Data Sample")
        st.dataframe(df.head(10), use_container_width=True)
        
        if st.button("Proceed to Step 4 ➡️", key="step3_btn"):
            st.session_state.current_step = 3
            st.rerun()

# ==================== STEP 4: ENCODING ====================
if st.session_state.df_clean is not None and st.session_state.current_step >= 3:
    with st.expander("🔢 Step 4: Encoding", expanded=st.session_state.current_step == 3):
        df = st.session_state.df_clean.copy()
        
        categorical_cols = ['persona', 'fitness_goal', 'experience_level', 'injury_constraint', 
                           'bodyPart', 'equipment', 'target_muscle']
        
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        
        df = df.drop(columns=categorical_cols)
        
        columns_to_drop = ['user_id', 'user_name', 'exercise_name', 'bodyPart_icon', 
                          'equipment_icon', 'persona_icon', 'injury_icon']
        df = df.drop(columns=columns_to_drop)
        
        st.session_state.df_encoded = df
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Columns", len(st.session_state.df_clean.columns))
        with col2:
            st.metric("After Encoding", len(df))
        with col3:
            st.metric("Categorical Columns", len(categorical_cols))
        
        st.subheader("📋 Categorical Columns Encoded")
        for col in categorical_cols:
            st.markdown(f"✅ {col}")
        
        # Visualization
        st.subheader("📊 Encoding Visualization")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].bar(['Before', 'After'], [len(st.session_state.df_clean.columns), len(df.columns)], 
                   color=['#FF6B6B', '#4ECDC4'])
        axes[0].set_title('Columns Before vs After', fontsize=12, fontweight='bold')
        axes[0].set_facecolor('none')
        
        dtypes_after = df.dtypes.value_counts()
        axes[1].pie(dtypes_after.values, labels=dtypes_after.index.astype(str), 
                   autopct='%1.1f%%', colors=['#45B7D1', '#96CEB4', '#FFEAA7'])
        axes[1].set_title('Data Types After Encoding', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("📋 Encoded Data Sample")
        st.dataframe(df.head(10), use_container_width=True)
        
        if st.button("Proceed to Step 5 ➡️", key="step4_btn"):
            st.session_state.current_step = 4
            st.rerun()

# ==================== STEP 5: FEATURE ENGINEERING ====================
if st.session_state.df_encoded is not None and st.session_state.current_step >= 4:
    with st.expander("⚙️ Step 5: Feature Engineering", expanded=st.session_state.current_step == 4):
        df = st.session_state.df_encoded.copy()
        
        def age_group(age):
            if age < 25: return 'young'
            elif age < 40: return 'adult'
            elif age < 60: return 'middle_age'
            else: return 'senior'
        
        df['age_group'] = df['age'].apply(age_group)
        le_age = LabelEncoder()
        df['age_group_encoded'] = le_age.fit_transform(df['age_group'])
        
        def rating_category(rating):
            if rating <= 2.0: return 'low'
            elif rating <= 3.5: return 'medium'
            else: return 'high'
        
        df['rating_category'] = df['rating'].apply(rating_category)
        df = df.drop(columns=['age_group', 'rating_category'])
        
        st.session_state.df_fe = df
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Features", len(st.session_state.df_encoded.columns))
        with col2:
            st.metric("After Engineering", len(df))
        with col3:
            st.metric("New Features Added", len(df.columns) - len(st.session_state.df_encoded.columns))
        
        st.subheader("📋 New Features Created")
        st.success("✅ age_group_encoded")
        
        # Visualization
        st.subheader("📊 Feature Distributions")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].hist(df['age'], bins=30, color='#4ECDC4', edgecolor='black', alpha=0.7)
        axes[0].axvline(df['age'].mean(), color='red', linestyle='--', linewidth=2)
        axes[0].set_title('Age Distribution', fontsize=12, fontweight='bold')
        axes[0].set_facecolor('none')
        
        axes[1].hist(df['rating'], bins=20, color='#FF6B6B', edgecolor='black', alpha=0.7)
        axes[1].axvline(df['rating'].mean(), color='blue', linestyle='--', linewidth=2)
        axes[1].set_title('Rating Distribution', fontsize=12, fontweight='bold')
        axes[1].set_facecolor('none')
        
        axes[2].bar(['Original', 'After Eng.'], 
                   [len(st.session_state.df_encoded.columns), len(df.columns)], 
                   color=['#45B7D1', '#96CEB4'])
        axes[2].set_title('Feature Count', fontsize=12, fontweight='bold')
        axes[2].set_facecolor('none')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("📋 Engineered Data Sample")
        st.dataframe(df.head(10), use_container_width=True)
        
        if st.button("Proceed to Step 6 ➡️", key="step5_btn"):
            st.session_state.current_step = 5
            st.rerun()

# ==================== STEP 6: FEATURE SELECTION ====================
if st.session_state.df_fe is not None and st.session_state.current_step >= 5:
    with st.expander("🎯 Step 6: Feature Selection", expanded=st.session_state.current_step == 5):
        df = st.session_state.df_fe
        
        X = df.drop(columns=['rating'])
        y = df['rating']
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        k = min(8, len(X.columns))
        selected_features = feature_importance.head(k)['feature'].tolist()
        df_selected = df[selected_features + ['rating']]
        
        st.session_state.df_selected = df_selected
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Selected Features", len(selected_features))
        with col2:
            st.metric("Top Features", ", ".join(selected_features[:3]))
        
        st.subheader("📋 Selected Features")
        for feat in selected_features:
            st.markdown(f"✅ {feat}")
        
        # Visualization
        st.subheader("📊 Feature Importance")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        top_features = feature_importance.head(10)
        axes[0].barh(top_features['feature'], top_features['importance'], color='#4ECDC4')
        axes[0].set_title('Top 10 Feature Importance', fontsize=12, fontweight='bold')
        axes[0].set_facecolor('none')
        
        correlations = df_selected.corr()['rating'].drop('rating').sort_values(ascending=False)
        axes[1].bar(correlations.index, correlations.values, 
                   color=['#FF6B6B' if x < 0 else '#45B7D1' for x in correlations.values])
        axes[1].set_title('Correlation with Rating', fontsize=12, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].set_facecolor('none')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("📋 Feature Importance Details")
        st.dataframe(feature_importance, use_container_width=True)
        
        if st.button("Proceed to Step 7 ➡️", key="step6_btn"):
            st.session_state.current_step = 6
            st.rerun()

# ==================== STEP 7: TRAIN/TEST SPLIT ====================
if st.session_state.df_selected is not None and st.session_state.current_step >= 6:
    with st.expander("📊 Step 7: Train/Test Split", expanded=st.session_state.current_step == 6):
        df = st.session_state.df_selected
        
        X = df.drop(columns=['rating'])
        y = df['rating']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test = pd.DataFrame(X_test_scaled, columns=X.columns)
        
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.scaler = scaler
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Training Samples", len(X_train))
        with col2:
            st.metric("Testing Samples", len(X_test))
        with col3:
            st.metric("Training Mean", f"{y_train.mean():.3f}")
        with col4:
            st.metric("Testing Mean", f"{y_test.mean():.3f}")
        
        st.subheader("📋 Features Used")
        for feat in X.columns:
            st.markdown(f"✅ {feat}")
        
        # Visualization
        st.subheader("📊 Split Visualization")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].pie([len(X_train), len(X_test)], 
                   labels=['Training', 'Testing'], 
                   autopct='%1.1f%%',
                   colors=['#4ECDC4', '#FF6B6B'],
                   explode=[0.05, 0.05])
        axes[0].set_title('Train/Test Split', fontsize=12, fontweight='bold')
        axes[0].set_facecolor('none')
        
        axes[1].hist(y_train, bins=20, color='#45B7D1', alpha=0.7, label='Training')
        axes[1].set_title('Training Target', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].set_facecolor('none')
        
        axes[2].hist(y_test, bins=20, color='#FF6B6B', alpha=0.7, label='Testing')
        axes[2].set_title('Testing Target', fontsize=12, fontweight='bold')
        axes[2].legend()
        axes[2].set_facecolor('none')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        if st.button("Proceed to Step 8 ➡️", key="step7_btn"):
            st.session_state.current_step = 7
            st.rerun()

# ==================== STEP 8: MODEL TRAINING ====================
if st.session_state.X_train is not None and st.session_state.current_step >= 7:
    with st.expander("🤖 Step 8: Model Training", expanded=st.session_state.current_step == 7):
        with st.spinner("Training models... This may take a moment..."):
            X_train = st.session_state.X_train
            X_test = st.session_state.X_test
            y_train = st.session_state.y_train
            y_test = st.session_state.y_test
            
            model_dict = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
            }
            
            results = []
            trained_models = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (name, model) in enumerate(model_dict.items()):
                status_text.text(f"Training {name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results.append({
                    'Model': name,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R² Score': r2
                })
                trained_models[name] = model
                progress_bar.progress((idx + 1) / len(model_dict))
            
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('R² Score', ascending=False)
            
            st.session_state.results_df = results_df
            st.session_state.models = trained_models
            
            status_text.text("✅ All models trained successfully!")
            progress_bar.empty()
        
        st.success(f"🏆 Best Model: {results_df.iloc[0]['Model']}")
        
        st.subheader("📊 Model Comparison Results")
        st.dataframe(results_df, use_container_width=True)
        
        # Visualization
        st.subheader("📊 Model Performance Visualization")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].bar(results_df['Model'], results_df['R² Score'], color='#4ECDC4')
        axes[0].set_title('R² Score', fontsize=12, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].set_facecolor('none')
        for i, v in enumerate(results_df['R² Score']):
            axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
        
        axes[1].bar(results_df['Model'], results_df['RMSE'], color='#FF6B6B')
        axes[1].set_title('RMSE', fontsize=12, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].set_facecolor('none')
        
        axes[2].bar(results_df['Model'], results_df['MAE'], color='#45B7D1')
        axes[2].set_title('MAE', fontsize=12, fontweight='bold')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].set_facecolor('none')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        if st.button("Proceed to Step 9 ➡️", key="step8_btn"):
            st.session_state.current_step = 8
            st.rerun()

# ==================== STEP 9: MODEL EVALUATION ====================
if st.session_state.results_df is not None and st.session_state.current_step >= 8:
    with st.expander("📈 Step 9: Model Evaluation", expanded=st.session_state.current_step == 8):
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
        models = st.session_state.models
        
        best_model_name = 'XGBoost'
        best_model = models[best_model_name]
        st.session_state.best_model = best_model
        
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Training R²", f"{train_r2:.4f}")
        with col2:
            st.metric("Testing R²", f"{test_r2:.4f}")
        with col3:
            st.metric("Testing RMSE", f"{test_rmse:.4f}")
        with col4:
            st.metric("Testing MAE", f"{test_mae:.4f}")
        
        overfit_gap = train_r2 - test_r2
        if overfit_gap < 0.1:
            st.success(f"✅ No significant overfitting! (Gap: {overfit_gap:.4f})")
        else:
            st.warning(f"⚠️ Some overfitting detected! (Gap: {overfit_gap:.4f})")
        
        # Visualization
        st.subheader("📊 Evaluation Visualizations")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].scatter(y_test, y_pred_test, alpha=0.5, color='#4ECDC4')
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual')
        axes[0].set_ylabel('Predicted')
        axes[0].set_title(f'Actual vs Predicted\nR² = {test_r2:.4f}', fontsize=12, fontweight='bold')
        axes[0].set_facecolor('none')
        
        residuals = y_test - y_pred_test
        axes[1].hist(residuals, bins=30, color='#FF6B6B', alpha=0.7, edgecolor='black')
        axes[1].axvline(x=0, color='blue', linestyle='--', linewidth=2)
        axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
        axes[1].set_facecolor('none')
        
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        axes[2].barh(feature_importance['feature'], feature_importance['importance'], color='#45B7D1')
        axes[2].set_title('Feature Importance', fontsize=12, fontweight='bold')
        axes[2].set_facecolor('none')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        if st.button("Proceed to Step 10 ➡️", key="step9_btn"):
            st.session_state.current_step = 9
            st.rerun()

# ==================== STEP 10: MODEL COMPARISON ====================
if st.session_state.results_df is not None and st.session_state.current_step >= 9:
    with st.expander("🏆 Step 10: Model Comparison", expanded=st.session_state.current_step == 9):
        results_df = st.session_state.results_df
        
        st.success(f"🏆 Best Model: {results_df.iloc[0]['Model']}")
        
        st.subheader("📊 Complete Comparison Results")
        st.dataframe(results_df, use_container_width=True)
        
        # Radar Chart
        st.subheader("📊 Radar Chart - Overall Performance")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        categories = ['R² Score', 'RMSE (inv)', 'MAE (inv)']
        
        r2_scores = results_df['R² Score'].values
        rmse_scores = 1 - (results_df['RMSE'].values / results_df['RMSE'].max())
        mae_scores = 1 - (results_df['MAE'].values / results_df['MAE'].max())
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        colors = ['#4ECDC4', '#FF6B6B', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        for i, model in enumerate(results_df['Model']):
            values = [r2_scores[i], rmse_scores[i], mae_scores[i]]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        st.pyplot(fig)
        
        if st.button("Proceed to Step 11 ➡️", key="step10_btn"):
            st.session_state.current_step = 10
            st.rerun()

# ==================== STEP 11: PREDICTION ====================
if st.session_state.best_model is not None and st.session_state.current_step >= 10:
    with st.expander("🔮 Step 11: Prediction", expanded=st.session_state.current_step == 10):
        st.subheader("🔮 Make a Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            exercise_id = st.number_input("Exercise ID", value=1500, step=1)
            persona = st.slider("Persona (0-4)", 0.0, 4.0, 1.0, 0.1)
            fitness_goal = st.slider("Fitness Goal (0-13)", 0.0, 13.0, 7.0, 0.1)
            experience_level = st.slider("Experience Level (0-6)", 0.0, 6.0, 5.0, 0.1)
        
        with col2:
            injury_constraint = st.slider("Injury Constraint (0-4)", 0.0, 4.0, 3.0, 0.1)
            bodyPart = st.slider("Body Part (0-63)", 0.0, 63.0, 34.0, 0.1)
            equipment = st.slider("Equipment (0-82)", 0.0, 82.0, 58.0, 0.1)
            target_muscle = st.slider("Target Muscle (0-13)", 0.0, 13.0, 3.0, 0.1)
        
        if st.button("🚀 Predict Rating", key="predict_btn"):
            try:
                features = [exercise_id, persona, fitness_goal, experience_level,
                           injury_constraint, bodyPart, equipment, target_muscle]
                
                features_scaled = st.session_state.scaler.transform([features])
                prediction = st.session_state.best_model.predict(features_scaled)[0]
                
                st.markdown("---")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 2rem; background: rgba(255,255,255,0.05); border-radius: 20px; border: 1px solid rgba(255,255,255,0.1);">
                        <h1 style="font-size: 4rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                            {prediction:.2f} / 5.0
                        </h1>
                        <p style="font-size: 1.2rem; color: white;">
                            {'🌟 Excellent match! Highly recommended!' if prediction >= 4.0 else 
                             '👍 Good match! Could work well for you.' if prediction >= 3.0 else
                             '😐 Average match. Consider other options.' if prediction >= 2.0 else
                             '👎 Poor match. Not recommended.'}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        st.markdown("---")
        if st.button("🔄 Reset Pipeline", key="reset_btn"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()