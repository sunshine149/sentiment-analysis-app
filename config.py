#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件
"""

import os
from pathlib import Path

# 基础路径
BASE_DIR = Path(__file__).parent.absolute()

# 数据路径
DATA_DIR = BASE_DIR / 'data'
FEEDBACK_DIR = BASE_DIR / 'data' / 'feedback'
# FEEDBACK_DIR = BASE_DIR / 'feedback'
MODELS_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'

# 模型配置
MODEL_CONFIG = {
    'vectorizer_params': {
        'max_features': 5000,
        'ngram_range': (1, 2),
        'min_df': 3,
        'max_df': 0.9
    },
    'classifier_params': {
        'loss': 'log_loss',
        'max_iter': 1000,
        'tol': 1e-3,
        'random_state': 42
    }
}

# 是否使用GPU（如果可用）
USE_GPU = False  # 可以在运行时通过API更改

# 训练配置
TRAINING_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'epochs': 5
}

# 模型类型配置
MODEL_TYPES = {
    'sgd_tfidf': {
        'name': 'TF-IDF + SGD分类器',
        'description': '传统机器学习方法，速度快，适合小数据',
        'requires_vectorizer': True
    },
    'random_forest': {
        'name': 'TF-IDF + 随机森林',
        'description': '集成学习方法，抗过拟合能力强',
        'requires_vectorizer': True
    },
    'svm_tfidf': {
        'name': 'TF-IDF + SVM',
        'description': '支持向量机，适合高维特征',
        'requires_vectorizer': True
    },
    'logistic_regression': {
        'name': 'TF-IDF + 逻辑回归',
        'description': '简单高效的线性模型',
        'requires_vectorizer': True
    }
}

# 图表配置
CHART_CONFIG = {
    'colors': {
        'train': '#667eea',
        'valid': '#00b09b',
        'test': '#ff416c',
        'baseline': '#ffb347',
        'new_model': '#764ba2'
    },
    'max_points': 100
}