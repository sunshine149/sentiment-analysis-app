#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情感分析应用 - 后端主程序
支持TF-IDF + SGDClassifier模型
适配中文微博数据集
"""

import os
import json
import time
import datetime
import pickle
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import jieba
import joblib
import warnings

warnings.filterwarnings('ignore')

# 导入配置
from config import *

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量
current_model = None
current_vectorizer = None
current_model_name = None
current_model_time = None
training_in_progress = False

# 新增：训练阈值控制相关变量
current_feedback_threshold = 0  # 记录上次训练时的反馈数量阈值
last_training_feedback_count = 0  # 记录上次训练时的反馈数量
TRAINING_TRIGGER_THRESHOLD = 10  # 触发训练的阈值（10条新反馈）

training_status = {
    'is_running': False,
    'progress': 0,
    'current_epoch': 0,
    'total_epochs': 5,
    'eta_seconds': 0,
    'message': '空闲',
    'current_acc': 0,
    'current_loss': 0,
    'use_gpu': False
}
training_thread = None
training_start_time = None

# 初始化模型目录
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FEEDBACK_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


# ========== 新增：统一去重函数 ==========
def remove_duplicate_feedback(df):
    """统一去除重复反馈，确保前后端数据一致"""
    if df.empty or 'text' not in df.columns:
        return df

    # 记录原始数量
    original_count = len(df)

    # 清理文本：去除前后空格，统一格式
    df['text_clean'] = df['text'].astype(str).str.strip()

    # 去除文本完全相同的重复（基于清理后的文本）
    df_clean = df.drop_duplicates(subset=['text_clean'], keep='first')

    # 移除临时列
    df_clean = df_clean.drop(columns=['text_clean'], errors='ignore')

    removed_count = original_count - len(df_clean)
    if removed_count > 0:
        print(f"去除了 {removed_count} 条重复反馈，保留 {len(df_clean)} 条唯一反馈")

    return df_clean


def load_latest_model():
    """加载最新的模型"""
    global current_model, current_vectorizer, current_model_name, current_model_time

    model_files = list(MODELS_DIR.glob('model_*.pkl'))
    vectorizer_files = list(MODELS_DIR.glob('vectorizer_*.pkl'))

    if not model_files:
        print("未找到模型，将训练初始模型...")
        train_initial_model()
        return

    # 按时间戳排序，获取最新的模型
    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_model = model_files[0]

    # 找到对应的向量器
    timestamp = latest_model.stem.split('_')[1] + '_' + latest_model.stem.split('_')[2]
    vectorizer_pattern = f'vectorizer_{timestamp}.pkl'
    vectorizer_files = list(MODELS_DIR.glob(vectorizer_pattern))

    if not vectorizer_files:
        print(f"警告: 找不到与模型匹配的向量器文件 {vectorizer_pattern}")
        return

    latest_vectorizer = vectorizer_files[0]

    try:
        current_model = joblib.load(latest_model)
        current_vectorizer = joblib.load(latest_vectorizer)
        current_model_name = latest_model.name
        current_model_time = datetime.datetime.fromtimestamp(latest_model.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"已加载模型: {current_model_name} (创建于: {current_model_time})")
    except Exception as e:
        print(f"加载模型失败: {e}")
        train_initial_model()


def train_initial_model():
    """训练初始模型"""
    global current_model, current_vectorizer, current_model_name, current_model_time
    global last_training_feedback_count  # 新增：记录训练时的反馈数量

    print("开始训练初始模型...")

    # 加载数据集
    try:
        # 优先使用ChnSentiCorp_htl_all.csv
        if (DATA_DIR / 'ChnSentiCorp_htl_all.csv').exists():
            df = pd.read_csv(DATA_DIR / 'ChnSentiCorp_htl_all.csv')
            print(f"加载微博数据集: {len(df)} 条数据")
        else:
            df = pd.read_csv(DATA_DIR / 'base_dataset.csv')
            print(f"加载基础数据集: {len(df)} 条数据")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return

    # 确保列名正确
    if 'text' not in df.columns or 'label' not in df.columns:
        print("数据集格式错误，需要 'text' 和 'label' 列")
        return

    # 文本预处理
    def preprocess_text(text):
        if isinstance(text, str):
            # 使用jieba分词
            words = jieba.lcut(text)
            return ' '.join(words)
        return ''

    df['processed_text'] = df['text'].apply(preprocess_text)

    # 准备数据
    X = df['processed_text'].values
    y = df['label'].values

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 创建TF-IDF向量器
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9
    )

    # 向量化
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 训练模型
    model = SGDClassifier(
        loss='log_loss',
        max_iter=1000,
        tol=1e-3,
        random_state=42,
        n_jobs=-1 if USE_GPU else 1
    )

    model.fit(X_train_vec, y_train)

    # 评估模型
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"初始模型训练完成 - 准确率: {accuracy:.4f}, F1分数: {f1:.4f}")

    # 保存模型
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = MODELS_DIR / f'model_{timestamp}.pkl'
    vectorizer_path = MODELS_DIR / f'vectorizer_{timestamp}.pkl'

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    # 更新当前模型
    current_model = model
    current_vectorizer = vectorizer
    current_model_name = model_path.name
    current_model_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 更新训练时的反馈数量
    feedback_count_after_training = count_valid_feedback(unique_only=True)
    last_training_feedback_count = feedback_count_after_training

    # 保存阈值信息
    threshold_file = LOGS_DIR / 'training_threshold.txt'
    threshold_data = {
        'threshold': current_feedback_threshold,
        'feedback_count': feedback_count_after_training,
        'training_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': current_model_name,
        'training_type': 'initial'
    }

    with open(threshold_file, 'w', encoding='utf-8') as f:
        json.dump(threshold_data, f, ensure_ascii=False, indent=2)

    print(f"初始训练完成时反馈数量: {feedback_count_after_training}")

    # 记录训练日志
    log_entry = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': current_model_name,
        'dataset': 'initial',
        'samples': len(df),
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'features': X_train_vec.shape[1],
        'use_gpu': USE_GPU,
        'feedback_count_at_training': feedback_count_after_training
    }

    log_file = LOGS_DIR / f'training_{timestamp}.txt'
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_entry, f, ensure_ascii=False, indent=2)

    print(f"模型已保存: {current_model_name}")


def reset_training_status():
    """重置训练状态"""
    global training_status
    training_status = {
        'is_running': False,
        'progress': 0,
        'current_epoch': 0,
        'total_epochs': 5,
        'eta_seconds': 0,
        'message': '空闲',
        'current_acc': 0,
        'current_loss': 0,
        'use_gpu': False
    }


def train_model_incremental(use_gpu=False, epochs=5, batch_size=32):
    """增量训练模型（使用基础数据 + 反馈数据）"""
    global training_in_progress, training_status, current_model, current_vectorizer
    global current_model_name, current_model_time, training_start_time
    global last_training_feedback_count  # 新增：记录训练时的反馈数量

    # 重置训练状态
    reset_training_status()

    training_in_progress = True
    training_status['is_running'] = True
    training_status['use_gpu'] = use_gpu
    training_status['total_epochs'] = epochs
    training_start_time = time.time()

    try:
        # 1. 备份当前模型
        if current_model is not None and current_vectorizer is not None:
            backup_model()

        # 2. 加载所有数据（基础数据 + 反馈数据）
        training_status['message'] = '正在加载数据...'

        # 加载基础数据
        if (DATA_DIR / 'ChnSentiCorp_htl_all.csv').exists():
            df_base = pd.read_csv(DATA_DIR / 'ChnSentiCorp_htl_all.csv')
        else:
            df_base = pd.read_csv(DATA_DIR / 'base_dataset.csv')

        # 加载反馈数据（使用统一去重）
        feedback_files = list(FEEDBACK_DIR.glob('*.csv'))
        feedback_dfs = []
        for f in feedback_files:
            try:
                fb_df = pd.read_csv(f)
                # 只使用"符合"和"不符合"的反馈
                if 'choice' in fb_df.columns:
                    fb_df = fb_df[fb_df['choice'].isin(['符合', '不符合'])]
                    # 将"符合"转为原始标签，"不符合"转为相反标签
                    fb_df['label'] = fb_df.apply(
                        lambda row: row['predicted_label'] if row['choice'] == '符合' else 1 - row['predicted_label'],
                        axis=1
                    )
                feedback_dfs.append(fb_df[['text', 'label']])
            except:
                continue

        if feedback_dfs:
            df_feedback = pd.concat(feedback_dfs, ignore_index=True)

            # ========== 重要修改：使用统一去重 ==========
            df_feedback = remove_duplicate_feedback(df_feedback)

            print(f"加载反馈数据: {len(df_feedback)} 条 (已去重)")
            training_status['message'] = f'已加载 {len(df_base)} 条基础数据和 {len(df_feedback)} 条反馈数据(已去重)'
        else:
            df_feedback = pd.DataFrame(columns=['text', 'label'])
            training_status['message'] = f'已加载 {len(df_base)} 条基础数据，无反馈数据'

        # 合并数据
        df_all = pd.concat([df_base, df_feedback], ignore_index=True)

        # 3. 文本预处理
        training_status['message'] = '正在预处理文本...'

        def preprocess_text(text):
            if isinstance(text, str):
                words = jieba.lcut(text)
                return ' '.join(words)
            return ''

        df_all['processed_text'] = df_all['text'].apply(preprocess_text)

        # 4. 准备训练数据
        X = df_all['processed_text'].values
        y = df_all['label'].values

        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )

        # 5. 创建新的向量器（适应新数据）
        training_status['message'] = '正在创建文本向量...'
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )

        X_train_vec = vectorizer.fit_transform(X_train)
        X_val_vec = vectorizer.transform(X_val)

        # 6. 训练模型（分轮次训练，使用用户指定的epochs）
        training_status['current_epoch'] = 0

        # 创建新模型
        model = SGDClassifier(
            loss='log_loss',
            max_iter=1,  # 每次只训练一个epoch
            tol=1e-3,
            random_state=42,
            warm_start=True,  # 允许增量训练
            n_jobs=-1 if use_gpu else 1
        )

        # 记录每轮的性能
        epoch_accuracies = []
        epoch_losses = []

        for epoch in range(epochs):
            training_status['current_epoch'] = epoch + 1
            training_status['message'] = f'正在训练第 {epoch + 1}/{epochs} 轮...'

            # 训练一个epoch
            if epoch == 0:
                model.fit(X_train_vec, y_train)
            else:
                model.partial_fit(X_train_vec, y_train, classes=np.unique(y_train))

            # 评估当前模型
            y_pred = model.predict(X_val_vec)
            accuracy = accuracy_score(y_val, y_pred)

            # 计算损失（对数损失）
            y_proba = model.predict_proba(X_val_vec)
            loss = -np.mean(np.log(y_proba[np.arange(len(y_val)), y_val] + 1e-10))

            epoch_accuracies.append(accuracy)
            epoch_losses.append(loss)

            # 更新训练状态
            elapsed_time = time.time() - training_start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            remaining_epochs = epochs - (epoch + 1)
            eta_seconds = avg_time_per_epoch * remaining_epochs

            training_status['progress'] = ((epoch + 1) / epochs) * 100
            training_status['eta_seconds'] = eta_seconds
            training_status['current_acc'] = accuracy
            training_status['current_loss'] = loss

            # 每轮之间稍作延迟，模拟训练过程
            time.sleep(1)

        # 7. 保存新模型
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = MODELS_DIR / f'model_{timestamp}.pkl'
        vectorizer_path = MODELS_DIR / f'vectorizer_{timestamp}.pkl'

        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)

        # 8. 更新当前模型
        current_model = model
        current_vectorizer = vectorizer
        current_model_name = model_path.name
        current_model_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 9. 更新训练时的反馈数量（新增）
        feedback_count_after_training = count_valid_feedback(unique_only=True)
        last_training_feedback_count = feedback_count_after_training

        # 更新阈值文件
        threshold_file = LOGS_DIR / 'training_threshold.txt'
        threshold_data = {
            'threshold': current_feedback_threshold,
            'feedback_count': feedback_count_after_training,
            'training_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_name': current_model_name,
            'training_type': 'incremental'
        }

        with open(threshold_file, 'w', encoding='utf-8') as f:
            json.dump(threshold_data, f, ensure_ascii=False, indent=2)

        print(f"训练完成时反馈数量: {feedback_count_after_training}")

        # 10. 记录训练日志
        log_entry = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_name': current_model_name,
            'dataset': 'incremental',
            'base_samples': len(df_base),
            'feedback_samples': len(df_feedback),
            'total_samples': len(df_all),
            'epochs': epochs,
            'batch_size': batch_size,
            'epoch_accuracies': [float(acc) for acc in epoch_accuracies],
            'epoch_losses': [float(loss) for loss in epoch_losses],
            'final_accuracy': float(accuracy),
            'final_loss': float(loss),
            'use_gpu': use_gpu,
            'training_time': time.time() - training_start_time,
            'feedback_count_at_training': feedback_count_after_training  # 新增
        }

        log_file = LOGS_DIR / f'training_{timestamp}.txt'
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)

        training_status['message'] = f'训练完成！最终准确率: {accuracy:.4f}'
        print(f"增量训练完成，新模型: {current_model_name}")

    except Exception as e:
        training_status['message'] = f'训练失败: {str(e)}'
        print(f"训练失败: {e}")
        # 训练失败时恢复备份
        restore_backup()

    finally:
        training_in_progress = False
        training_status['is_running'] = False


def backup_model():
    """备份当前模型"""
    if current_model is not None and current_vectorizer is not None:
        backup_dir = MODELS_DIR / 'backup'
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        joblib.dump(current_model, backup_dir / f'model_backup_{timestamp}.pkl')
        joblib.dump(current_vectorizer, backup_dir / f'vectorizer_backup_{timestamp}.pkl')
        print(f"已备份模型到 backup/ 目录")


def restore_backup():
    """恢复备份的模型"""
    backup_dir = MODELS_DIR / 'backup'
    if backup_dir.exists():
        model_files = list(backup_dir.glob('model_backup_*.pkl'))
        if model_files:
            model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_backup = model_files[0]

            # 找到对应的向量器
            timestamp = latest_backup.stem.split('_')[2]
            vectorizer_pattern = f'vectorizer_backup_{timestamp}.pkl'
            vectorizer_files = list(backup_dir.glob(vectorizer_pattern))

            if vectorizer_files:
                global current_model, current_vectorizer, current_model_name, current_model_time

                current_model = joblib.load(latest_backup)
                current_vectorizer = joblib.load(vectorizer_files[0])
                current_model_name = f"恢复的备份_{timestamp}"
                current_model_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                print(f"已从备份恢复模型")
                return True

    print("没有可用的备份模型")
    return False


def get_model_info():
    """获取当前模型信息"""
    if current_model is None:
        return {
            'model_name': '无模型',
            'model_time': 'N/A',
            'model_type': 'N/A',
            'features': 0,
            'is_trained': False
        }

    return {
        'model_name': current_model_name,
        'model_time': current_model_time,
        'model_type': 'SGDClassifier (log loss)',
        'features': current_vectorizer.get_feature_names_out().shape[0] if current_vectorizer else 0,
        'is_trained': True
    }


def list_available_models():
    """列出所有可用的模型"""
    model_files = list(MODELS_DIR.glob('model_*.pkl'))
    models = []

    for model_file in sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True):
        timestamp = model_file.stem.split('_')[1] + '_' + model_file.stem.split('_')[2]
        create_time = datetime.datetime.fromtimestamp(model_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')

        # 检查是否有对应的向量器
        vectorizer_pattern = f'vectorizer_{timestamp}.pkl'
        vectorizer_files = list(MODELS_DIR.glob(vectorizer_pattern))

        models.append({
            'name': model_file.name,
            'time': create_time,
            'has_vectorizer': len(vectorizer_files) > 0
        })

    return models


def switch_to_model(model_name):
    """切换到指定模型"""
    global current_model, current_vectorizer, current_model_name, current_model_time

    model_path = MODELS_DIR / model_name

    if not model_path.exists():
        return False, "模型文件不存在"

    # 提取时间戳
    try:
        timestamp = model_name.split('_')[1] + '_' + model_name.split('_')[2].replace('.pkl', '')
        vectorizer_name = f'vectorizer_{timestamp}.pkl'
        vectorizer_path = MODELS_DIR / vectorizer_name

        if not vectorizer_path.exists():
            return False, f"找不到对应的向量器文件: {vectorizer_name}"

        # 加载模型
        current_model = joblib.load(model_path)
        current_vectorizer = joblib.load(vectorizer_path)
        current_model_name = model_name
        current_model_time = datetime.datetime.fromtimestamp(model_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')

        # 备份当前模型（切换前的）
        backup_model()

        return True, f"已切换到模型: {model_name}"

    except Exception as e:
        return False, f"切换模型失败: {str(e)}"


# ========== 修改后的统计函数（统一去重） ==========
def count_all_feedback(unique_only=True):
    """统计所有反馈数据"""
    feedback_files = list(FEEDBACK_DIR.glob('*.csv'))

    if unique_only:
        # 去重统计（与前端保持一致）
        all_texts = set()
        for f in feedback_files:
            try:
                df = pd.read_csv(f)
                if 'text' in df.columns:
                    # 清理文本：去除空格，统一格式
                    texts = df['text'].dropna().astype(str).str.strip().tolist()
                    all_texts.update(texts)
            except:
                continue
        return len(all_texts)
    else:
        # 原始统计
        total_rows = 0
        for f in feedback_files:
            try:
                df = pd.read_csv(f)
                total_rows += len(df)
            except:
                continue
        return total_rows


def count_valid_feedback(unique_only=True):
    """统计有效反馈数据（仅符合和不符合）"""
    feedback_files = list(FEEDBACK_DIR.glob('*.csv'))

    if unique_only:
        # 去重统计（与前端保持一致）
        valid_texts = set()
        for f in feedback_files:
            try:
                df = pd.read_csv(f)
                if 'choice' in df.columns and 'text' in df.columns:
                    valid_df = df[df['choice'].isin(['符合', '不符合'])]
                    texts = valid_df['text'].dropna().astype(str).str.strip().tolist()
                    valid_texts.update(texts)
            except:
                continue
        return len(valid_texts)
    else:
        # 原始统计
        total_valid = 0
        for f in feedback_files:
            try:
                df = pd.read_csv(f)
                if 'choice' in df.columns:
                    valid_feedback = df[df['choice'].isin(['符合', '不符合'])]
                    total_valid += len(valid_feedback)
            except:
                continue
        return total_valid


# 初始化加载模型
load_latest_model()


# API路由
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """预测文本情感"""
    if current_model is None or current_vectorizer is None:
        return jsonify({'error': '模型未加载'}), 500

    try:
        data = request.json
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': '文本不能为空'}), 400

        # 预处理文本
        words = jieba.lcut(text)
        processed_text = ' '.join(words)

        # 向量化
        text_vec = current_vectorizer.transform([processed_text])

        # 预测
        prediction = current_model.predict(text_vec)[0]
        proba = current_model.predict_proba(text_vec)[0]

        # 获取置信度和标签文本
        confidence = float(proba[prediction])
        label_text = "正面" if prediction == 1 else "负面"

        return jsonify({
            'label': int(prediction),
            'label_text': label_text,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': f'预测失败: {str(e)}'}), 500


@app.route('/api/feedback', methods=['POST'])
def feedback():
    """保存用户反馈"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        choice = data.get('choice', '')
        predicted_label = data.get('predicted_label', -1)

        if not text or choice not in ['符合', '不符合', '无法判断']:
            return jsonify({'error': '参数无效'}), 400

        # 清理文本：去除前后空格
        cleaned_text = text.strip()

        # 创建反馈记录
        feedback_record = {
            'text': cleaned_text,
            'choice': choice,
            'predicted_label': predicted_label,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # 保存到CSV文件
        feedback_file = FEEDBACK_DIR / f'feedback_{datetime.datetime.now().strftime("%Y%m%d")}.csv'

        # 如果文件不存在，创建并写入表头
        if not feedback_file.exists():
            pd.DataFrame([feedback_record]).to_csv(feedback_file, index=False, encoding='utf-8-sig')
        else:
            # 追加到现有文件
            df = pd.read_csv(feedback_file)
            new_df = pd.DataFrame([feedback_record])
            pd.concat([df, new_df], ignore_index=True).to_csv(feedback_file, index=False, encoding='utf-8-sig')

        return jsonify({'message': '反馈已保存'})

    except Exception as e:
        return jsonify({'error': f'保存反馈失败: {str(e)}'}), 500


@app.route('/api/train', methods=['POST'])
def train():
    """开始训练模型（手动或自动）"""
    global training_thread, last_training_feedback_count

    if training_in_progress:
        return jsonify({'error': '训练已在进行中'}), 400

    try:
        data = request.json
        use_gpu = data.get('use_gpu', False)
        epochs = data.get('epochs', 5)
        batch_size = data.get('batch_size', 32)

        # 验证参数范围
        if epochs < 1 or epochs > 20:
            return jsonify({'error': '训练轮数应在1-20之间'}), 400

        if batch_size < 16 or batch_size > 128:
            return jsonify({'error': '批次大小应在16-128之间'}), 400

        # 检查GPU是否可用
        gpu_available = False
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except:
            pass

        if use_gpu and not gpu_available:
            return jsonify({
                'message': 'GPU不可用，将使用CPU进行训练',
                'use_gpu': False
            })

        # 手动训练时也更新阈值
        current_count = count_valid_feedback(unique_only=True)

        # 在后台线程中开始训练
        training_thread = threading.Thread(
            target=train_model_incremental,
            kwargs={'use_gpu': use_gpu, 'epochs': epochs, 'batch_size': batch_size}
        )
        training_thread.daemon = True
        training_thread.start()

        return jsonify({
            'message': '训练已开始',
            'use_gpu_actual': use_gpu and gpu_available,
            'epochs': epochs,
            'batch_size': batch_size,
            'feedback_count_at_start': current_count
        })

    except Exception as e:
        return jsonify({'error': f'启动训练失败: {str(e)}'}), 500


@app.route('/api/train_status', methods=['GET'])
def train_status():
    """获取训练状态"""
    return jsonify(training_status)


@app.route('/api/session_end', methods=['POST'])
def session_end():
    """页面关闭时触发训练（满足阈值条件才触发）"""
    global current_feedback_threshold, last_training_feedback_count

    try:
        # 获取当前有效的唯一反馈数量
        current_feedback_count = count_valid_feedback(unique_only=True)
        total_all = count_all_feedback(unique_only=True)

        # 读取上次训练的阈值（如果存在）
        threshold_file = LOGS_DIR / 'training_threshold.txt'
        if threshold_file.exists():
            try:
                with open(threshold_file, 'r', encoding='utf-8') as f:
                    threshold_data = json.load(f)
                    current_feedback_threshold = threshold_data.get('threshold', 0)
                    last_training_feedback_count = threshold_data.get('feedback_count', 0)
            except:
                # 如果文件损坏，重置阈值
                current_feedback_threshold = 0
                last_training_feedback_count = 0

        # 计算自上次训练以来的新增反馈数量
        new_feedback_since_last_training = current_feedback_count - last_training_feedback_count

        # 计算需要达到的阈值
        required_feedback_for_training = current_feedback_threshold + TRAINING_TRIGGER_THRESHOLD

        # 检查是否满足训练条件
        # 条件1: 当前反馈数量 >= 阈值 + 10
        # 条件2: 自上次训练以来新增了至少10条反馈
        # 条件3: 训练不在进行中
        should_trigger_training = (
                current_feedback_count >= required_feedback_for_training and
                new_feedback_since_last_training >= TRAINING_TRIGGER_THRESHOLD and
                not training_in_progress
        )

        response_data = {
            'message': f'当前反馈数量: {current_feedback_count} 条',
            'total_unique_feedback': total_all,
            'total_unique_valid_feedback': current_feedback_count,
            'last_training_feedback_count': last_training_feedback_count,
            'new_feedback_since_last_training': new_feedback_since_last_training,
            'current_threshold': current_feedback_threshold,
            'required_for_next_training': required_feedback_for_training,
            'triggered_training': False
        }

        if should_trigger_training:
            # 更新阈值和上次训练时的反馈数量
            current_feedback_threshold = required_feedback_for_training
            last_training_feedback_count = current_feedback_count

            # 保存阈值信息
            threshold_data = {
                'threshold': current_feedback_threshold,
                'feedback_count': last_training_feedback_count,
                'updated_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'trigger_type': 'auto'
            }

            with open(threshold_file, 'w', encoding='utf-8') as f:
                json.dump(threshold_data, f, ensure_ascii=False, indent=2)

            # 异步触发训练
            threading.Thread(
                target=train_model_incremental,
                kwargs={'use_gpu': False, 'epochs': 5, 'batch_size': 32}
            ).start()

            response_data.update({
                'triggered_training': True,
                'message': f'检测到 {new_feedback_since_last_training} 条新反馈（总计 {current_feedback_count} 条），已触发自动训练',
                'new_threshold_set': current_feedback_threshold
            })
            print(f"自动训练触发: {new_feedback_since_last_training} 条新反馈，设置新阈值: {current_feedback_threshold}")

        return jsonify(response_data)

    except Exception as e:
        print(f"session_end处理失败: {e}")
        return jsonify({
            'error': str(e),
            'triggered_training': False
        }), 500


@app.route('/api/list_models', methods=['GET'])
def api_list_models():
    """列出所有可用模型"""
    models = list_available_models()
    current_info = get_model_info()

    return jsonify({
        'models': models,
        'current_model': current_info
    })


@app.route('/api/switch_model', methods=['POST'])
def api_switch_model():
    """切换模型"""
    if training_in_progress:
        return jsonify({'error': '训练进行中，无法切换模型'}), 400

    try:
        data = request.json
        model_name = data.get('model_name', '')

        if not model_name:
            return jsonify({'error': '请指定模型名称'}), 400

        success, message = switch_to_model(model_name)

        if success:
            return jsonify({'message': message})
        else:
            return jsonify({'error': message}), 400

    except Exception as e:
        return jsonify({'error': f'切换模型失败: {str(e)}'}), 500


@app.route('/api/model_info', methods=['GET'])
def api_model_info():
    """获取当前模型信息"""
    info = get_model_info()
    return jsonify(info)


@app.route('/api/check_gpu', methods=['GET'])
def check_gpu():
    """检查GPU是否可用"""
    gpu_available = False
    gpu_info = {}

    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_info = {
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(0)
            }
    except ImportError:
        pass

    return jsonify({
        'gpu_available': gpu_available,
        'gpu_info': gpu_info
    })


@app.route('/api/rollback', methods=['POST'])
def rollback():
    """回滚到上一个模型"""
    if training_in_progress:
        return jsonify({'error': '训练进行中，无法回滚'}), 400

    success = restore_backup()

    if success:
        return jsonify({'message': '已回滚到上一个模型'})
    else:
        return jsonify({'error': '回滚失败，没有可用的备份'}), 400


@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': current_model is not None,
        'model_type': 'TF-IDF + SGDClassifier',
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })


@app.route('/api/train_options', methods=['GET'])
def train_options():
    """获取训练选项"""
    gpu_available = False
    gpu_info = {}

    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_info = {
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(0),
                'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9
            }
    except ImportError:
        pass

    return jsonify({
        'gpu_available': gpu_available,
        'gpu_info': gpu_info,
        'max_samples': 100000,
        'max_epochs': 20,
        'available_batch_sizes': [16, 32, 64, 128]
    })


@app.route('/api/verify_dataset', methods=['GET'])
def verify_dataset():
    """验证数据集"""
    try:
        # 检查数据文件是否存在
        if (DATA_DIR / 'weibo_senti_100k.csv').exists():
            df = pd.read_csv(DATA_DIR / 'weibo_senti_100k.csv', nrows=5)
            return jsonify({
                'success': True,
                'message': f'数据集验证通过，共 {len(pd.read_csv(DATA_DIR / "weibo_senti_100k.csv"))} 条数据',
                'sample_rows': df.to_dict('records')
            })
        elif (DATA_DIR / 'base_dataset.csv').exists():
            df = pd.read_csv(DATA_DIR / 'base_dataset.csv')
            return jsonify({
                'success': True,
                'message': f'基础数据集验证通过，共 {len(df)} 条数据',
                'sample_rows': df.to_dict('records')
            })
        else:
            return jsonify({
                'success': False,
                'error': '未找到数据集文件'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'数据集验证失败: {str(e)}'
        })


@app.route('/api/test', methods=['POST'])
def test_api():
    """测试API"""
    test_texts = [
        "这部电影真的太好看了，强烈推荐！",
        "服务态度太差了，非常不满意。",
        "产品质量很好，性价比很高。",
        "这次购物体验非常糟糕。",
        "学习机器学习真的很有趣。"
    ]

    results = []
    for text in test_texts:
        try:
            words = jieba.lcut(text)
            processed_text = ' '.join(words)
            text_vec = current_vectorizer.transform([processed_text])
            prediction = current_model.predict(text_vec)[0]
            proba = current_model.predict_proba(text_vec)[0]
            confidence = float(proba[prediction])
            label_text = "正面" if prediction == 1 else "负面"

            results.append({
                'text': text,
                'label': int(prediction),
                'label_text': label_text,
                'confidence': confidence
            })
        except:
            continue

    return jsonify({
        'success': True,
        'results': results
    })


@app.route('/api/feedback_stats', methods=['GET'])
def feedback_stats():
    """获取反馈数据统计（显示去重前后的对比）"""
    total_all_unique = count_all_feedback(unique_only=True)  # 去重后
    total_all_raw = count_all_feedback(unique_only=False)  # 原始
    total_valid_unique = count_valid_feedback(unique_only=True)  # 有效去重后
    total_valid_raw = count_valid_feedback(unique_only=False)  # 有效原始

    feedback_files = list(FEEDBACK_DIR.glob('*.csv'))

    # 文件详细信息
    file_details = []
    for file in feedback_files:
        try:
            df = pd.read_csv(file)
            file_details.append({
                'name': file.name,
                'raw_rows': len(df),
                'unique_rows': len(df['text'].dropna().astype(str).str.strip().unique()) if 'text' in df.columns else 0,
                'valid_rows': len(df[df['choice'].isin(['符合', '不符合'])]) if 'choice' in df.columns else 0
            })
        except:
            continue

    return jsonify({
        'total_all_raw': total_all_raw,  # 原始总数：66
        'total_all_unique': total_all_unique,  # 去重后总数：63
        'total_valid_raw': total_valid_raw,  # 原始有效数：66
        'total_valid_unique': total_valid_unique,  # 去重后有效数：63
        'duplicate_count': total_all_raw - total_all_unique,  # 重复数：3
        'unclassified_count': total_all_unique - total_valid_unique,  # 无法判断数：0
        'feedback_files': [f.name for f in feedback_files],
        'file_details': file_details,
        'message': f'共收到 {total_all_raw} 条反馈，去重后 {total_all_unique} 条唯一反馈，其中 {total_valid_unique} 条有效反馈'
    })


@app.route('/api/training_threshold_info', methods=['GET'])
def training_threshold_info():
    """获取训练阈值信息"""
    try:
        threshold_file = LOGS_DIR / 'training_threshold.txt'

        if threshold_file.exists():
            with open(threshold_file, 'r', encoding='utf-8') as f:
                threshold_data = json.load(f)
        else:
            threshold_data = {
                'threshold': 0,
                'feedback_count': 0,
                'updated_time': '从未训练'
            }

        current_count = count_valid_feedback(unique_only=True)
        new_feedback_since_last = current_count - threshold_data.get('feedback_count', 0)
        required_for_next = threshold_data.get('threshold', 0) + TRAINING_TRIGGER_THRESHOLD

        return jsonify({
            'success': True,
            'threshold_info': threshold_data,
            'current_feedback_count': current_count,
            'new_feedback_since_last_training': new_feedback_since_last,
            'required_feedback_for_next_training': required_for_next,
            'remaining_for_next_training': max(0, required_for_next - current_count),
            'trigger_threshold': TRAINING_TRIGGER_THRESHOLD
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("=" * 50)
    print("情感分析应用启动")
    print(f"模型目录: {MODELS_DIR}")
    print(f"数据目录: {DATA_DIR}")
    print(f"反馈目录: {FEEDBACK_DIR}")
    print("=" * 50)

    # 初始化阈值信息
    threshold_file = LOGS_DIR / 'training_threshold.txt'
    if threshold_file.exists():
        try:
            with open(threshold_file, 'r', encoding='utf-8') as f:
                threshold_data = json.load(f)
                current_feedback_threshold = threshold_data.get('threshold', 0)
                last_training_feedback_count = threshold_data.get('feedback_count', 0)
                print(
                    f"加载阈值信息: 阈值={current_feedback_threshold}, 上次训练时反馈数={last_training_feedback_count}")
        except Exception as e:
            print(f"加载阈值信息失败: {e}")

    # 显示当前反馈统计
    total_raw = count_all_feedback(unique_only=False)
    total_unique = count_all_feedback(unique_only=True)
    valid_unique = count_valid_feedback(unique_only=True)

    # 计算距离下次训练还需要多少反馈
    required_for_next = current_feedback_threshold + TRAINING_TRIGGER_THRESHOLD
    remaining = max(0, required_for_next - valid_unique)

    print(f"反馈数据统计:")
    print(f"  原始数据: {total_raw} 条")
    print(f"  去重后: {total_unique} 条")
    print(f"  有效反馈(去重后): {valid_unique} 条")
    print(f"  重复数据: {total_raw - total_unique} 条")
    print(f"  当前阈值: {current_feedback_threshold}")
    print(f"  距离下次自动训练还需: {remaining} 条新反馈")
    print("=" * 50)

    # 检查并创建必要目录
    for directory in [DATA_DIR, FEEDBACK_DIR, MODELS_DIR, LOGS_DIR]:
        directory.mkdir(exist_ok=True)

    # 启动Flask应用
    app.run(host='0.0.0.0', port=5000, debug=True)