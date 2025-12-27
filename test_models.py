#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型功能测试脚本
测试模型训练、预测、切换等功能
"""

import requests
import json
import time
import unittest
import sys
import os
import threading
import pandas as pd
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BASE_DIR, FEEDBACK_DIR, MODELS_DIR


class ModelTestCase(unittest.TestCase):
    """模型测试用例类"""

    BASE_URL = "http://localhost:5000"

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        print("\n" + "=" * 60)
        print("开始模型功能测试")
        print("=" * 60)

        # 等待服务启动
        max_retries = 10
        for i in range(max_retries):
            try:
                response = requests.get(f"{cls.BASE_URL}/health", timeout=5)
                if response.status_code == 200:
                    print("✓ 服务已启动")
                    return
            except:
                if i == max_retries - 1:
                    print(f"✗ 服务未启动")
                    sys.exit(1)
                print(f"等待服务启动...({i + 1}/{max_retries})")
                time.sleep(3)

    def test_1_all_model_types_training(self):
        """测试所有模型类型的训练功能"""
        print("\n[测试1] 模型类型训练测试")

        # 获取支持的模型类型
        response = requests.get(f"{self.BASE_URL}/api/model_types")
        if response.status_code != 200:
            print("  ⚠ 无法获取模型类型，跳过此测试")
            return

        data = response.json()
        if not data.get('success'):
            print(f"  ⚠ 获取模型类型失败: {data.get('error', '未知错误')}")
            return

        model_types = data.get('model_types', {})

        if not model_types:
            print("  ⚠ 没有可用的模型类型")
            return

        print(f"  发现 {len(model_types)} 种模型类型")

        # 测试每种模型类型
        for model_id, model_info in list(model_types.items())[:2]:  # 只测试前两种，避免时间太长
            print(f"\n  测试模型类型: {model_info.get('name', model_id)}")

            # 检查是否已有训练进程
            status_response = requests.get(f"{self.BASE_URL}/api/train_status")
            if status_response.status_code == 200:
                status_data = status_response.json()
                if status_data.get('is_running'):
                    print("  ⚠ 已有训练在进行中，跳过此模型")
                    continue

            # 开始训练
            train_data = {
                "use_gpu": False,
                "epochs": 2,  # 减少轮数以加快测试
                "batch_size": 16,
                "model_type": model_id
            }

            print(f"    开始训练: {train_data}")

            try:
                response = requests.post(
                    f"{self.BASE_URL}/api/train",
                    json=train_data,
                    headers={"Content-Type": "application/json"},
                    timeout=30  # 增加超时时间
                )

                if response.status_code == 200:
                    data = response.json()
                    print(f"    训练已启动: {data.get('message', 'N/A')}")

                    # 等待训练完成（简化版，实际应该轮询状态）
                    time.sleep(10)

                    # 检查训练状态
                    status_response = requests.get(f"{self.BASE_URL}/api/train_status")
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        print(f"    训练状态: {status_data.get('message', 'N/A')}")
                        print(f"    进度: {status_data.get('progress', 0):.1f}%")

                        if not status_data.get('is_running'):
                            print(f"    ✓ {model_info.get('name', model_id)} 训练完成")
                        else:
                            print(f"    ⚠ {model_info.get('name', model_id)} 训练仍在进行")
                    else:
                        print(f"    ⚠ 无法获取训练状态")
                else:
                    print(f"    ⚠ 启动训练失败: HTTP {response.status_code}")
                    error_data = response.json() if response.text else {}
                    print(f"    错误信息: {error_data.get('error', '未知错误')}")

            except requests.exceptions.Timeout:
                print(f"    ⚠ 训练请求超时，可能训练正在进行")
            except Exception as e:
                print(f"    ⚠ 训练请求异常: {e}")

        print("\n  ✓ 模型类型训练测试完成")

    def test_2_model_switch_functionality(self):
        """测试模型切换功能"""
        print("\n[测试2] 模型切换功能测试")

        # 获取模型列表
        response = requests.get(f"{self.BASE_URL}/api/list_models")
        if response.status_code != 200:
            print("  ⚠ 无法获取模型列表，跳过此测试")
            return

        data = response.json()
        models = data.get('models', [])

        if len(models) < 2:
            print(f"  ⚠ 需要至少2个模型才能测试切换功能，当前只有 {len(models)} 个")
            return

        print(f"  发现 {len(models)} 个可用模型")

        # 获取当前模型信息
        current_model_info = data.get('current_model', {})
        current_model_name = current_model_info.get('model_name', 'N/A')
        print(f"  当前模型: {current_model_name}")

        # 找到另一个模型进行切换
        target_model = None
        for model in models:
            if model.get('name') != current_model_name and model.get('has_vectorizer'):
                target_model = model
                break

        if not target_model:
            print("  ⚠ 没有找到可切换的目标模型")
            return

        print(f"  目标切换模型: {target_model.get('name')}")

        # 执行模型切换
        try:
            response = requests.post(
                f"{self.BASE_URL}/api/switch_model",
                json={"model_name": target_model.get('name')},
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                data = response.json()
                print(f"  切换结果: {data.get('message', 'N/A')}")

                # 验证切换是否成功
                time.sleep(2)  # 等待切换完成

                verify_response = requests.get(f"{self.BASE_URL}/api/model_info")
                if verify_response.status_code == 200:
                    verify_data = verify_response.json()
                    new_model_name = verify_data.get('model_name', 'N/A')

                    if new_model_name == target_model.get('name'):
                        print(f"  ✓ 模型切换验证成功")
                        print(f"    新模型: {new_model_name}")
                    else:
                        print(f"  ⚠ 模型切换验证失败")
                        print(f"    期望: {target_model.get('name')}")
                        print(f"    实际: {new_model_name}")
                else:
                    print(f"  ⚠ 无法验证模型切换")
            else:
                print(f"  ⚠ 模型切换失败: HTTP {response.status_code}")
                error_data = response.json() if response.text else {}
                print(f"    错误信息: {error_data.get('error', '未知错误')}")

        except Exception as e:
            print(f"  ⚠ 模型切换请求异常: {e}")

        print("\n  ✓ 模型切换功能测试完成")

    def test_3_model_rollback_functionality(self):
        """测试模型回滚功能"""
        print("\n[测试3] 模型回滚功能测试")

        # 检查是否允许回滚
        response = requests.get(f"{self.BASE_URL}/api/list_models")
        if response.status_code != 200:
            print("  ⚠ 无法获取模型列表，跳过此测试")
            return

        data = response.json()
        models = data.get('models', [])

        if len(models) < 2:
            print(f"  ⚠ 需要至少2个模型才能测试回滚功能，当前只有 {len(models)} 个")
            return

        print(f"  发现 {len(models)} 个可用模型")

        # 获取当前模型信息
        current_model_info = data.get('current_model', {})
        current_model_name = current_model_info.get('model_name', 'N/A')
        print(f"  当前模型: {current_model_name}")

        # 执行模型回滚
        try:
            response = requests.post(
                f"{self.BASE_URL}/api/rollback",
                json={},
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                data = response.json()
                print(f"  回滚结果: {data.get('message', 'N/A')}")

                # 验证回滚是否成功
                time.sleep(2)  # 等待回滚完成

                verify_response = requests.get(f"{self.BASE_URL}/api/model_info")
                if verify_response.status_code == 200:
                    verify_data = verify_response.json()
                    new_model_name = verify_data.get('model_name', 'N/A')

                    if new_model_name != current_model_name:
                        print(f"  ✓ 模型回滚验证成功")
                        print(f"    新模型: {new_model_name}")
                        print(f"    旧模型: {current_model_name}")
                    else:
                        print(f"  ⚠ 模型回滚验证失败，模型未变化")
                        print(f"    模型名称: {new_model_name}")
                else:
                    print(f"  ⚠ 无法验证模型回滚")
            else:
                print(f"  ⚠ 模型回滚失败: HTTP {response.status_code}")
                error_data = response.json() if response.text else {}
                print(f"    错误信息: {error_data.get('error', '未知错误')}")

        except Exception as e:
            print(f"  ⚠ 模型回滚请求异常: {e}")

        print("\n  ✓ 模型回滚功能测试完成")

    def test_4_threshold_mechanism(self):
        """测试阈值触发机制"""
        print("\n[测试4] 阈值触发机制测试")

        # 获取当前阈值信息
        response = requests.get(f"{self.BASE_URL}/api/training_threshold_info")
        if response.status_code != 200:
            print("  ⚠ 无法获取阈值信息，跳过此测试")
            return

        data = response.json()
        if not data.get('success'):
            print(f"  ⚠ 获取阈值信息失败: {data.get('error', '未知错误')}")
            return

        threshold_info = data.get('threshold_info', {})
        current_feedback_count = data.get('current_feedback_count', 0)
        last_training_feedback_count = threshold_info.get('feedback_count', 0)
        remaining_for_next = data.get('remaining_for_next_training', 0)

        print(f"  当前反馈数量: {current_feedback_count}")
        print(f"  上次训练反馈数: {last_training_feedback_count}")
        print(f"  还需反馈数: {remaining_for_next}")
        print(f"  触发阈值: {data.get('trigger_threshold', 10)}")

        # 测试页面关闭时的训练触发
        print(f"\n  测试页面关闭触发训练...")

        try:
            response = requests.post(
                f"{self.BASE_URL}/api/session_end",
                json={},
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                session_data = response.json()

                print(f"    当前反馈数量: {session_data.get('total_unique_valid_feedback', 0)}")
                print(f"    上次训练反馈数: {session_data.get('last_training_feedback_count', 0)}")
                print(f"    自上次训练新增反馈: {session_data.get('new_feedback_since_last_training', 0)}")
                print(f"    下次训练目标: {session_data.get('next_training_target', 0)}")
                print(f"    还需反馈数: {session_data.get('remaining_for_next_training', 0)}")

                triggered = session_data.get('triggered_training', False)
                if triggered:
                    print(f"    自动训练已触发: {session_data.get('message', 'N/A')}")
                    print(f"    新阈值设置: {session_data.get('new_threshold_set', 'N/A')}")

                    # 检查训练是否开始
                    time.sleep(3)
                    status_response = requests.get(f"{self.BASE_URL}/api/train_status")
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        if status_data.get('is_running'):
                            print(f"    ✓ 训练已成功启动")
                        else:
                            print(f"    ⚠ 训练未启动")
                else:
                    print(f"    未触发自动训练: {session_data.get('message', 'N/A')}")
                    print(f"    原因: 新增反馈不足或已有训练在进行")
            else:
                print(f"  ⚠ 页面关闭接口调用失败: HTTP {response.status_code}")

        except Exception as e:
            print(f"  ⚠ 页面关闭接口请求异常: {e}")

        print("\n  ✓ 阈值触发机制测试完成")

    def test_5_feedback_duplicate_removal(self):
        """测试反馈去重功能"""
        print("\n[测试5] 反馈去重功能测试")

        # 创建测试反馈数据
        test_feedbacks = [
            {
                "text": "这是一条测试反馈，用于测试去重功能。",
                "choice": "符合",
                "predicted_label": 1
            },
            {
                "text": "这是一条测试反馈，用于测试去重功能。",  # 重复文本
                "choice": "不符合",
                "predicted_label": 1
            },
            {
                "text": "这是另一条不同的测试反馈。",
                "choice": "符合",
                "predicted_label": 0
            }
        ]

        print(f"  创建 {len(test_feedbacks)} 条测试反馈，其中包含1条重复")

        # 提交测试反馈
        for i, feedback in enumerate(test_feedbacks):
            try:
                response = requests.post(
                    f"{self.BASE_URL}/api/feedback",
                    json=feedback,
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 200:
                    print(f"    反馈 {i + 1} 提交成功")
                else:
                    print(f"    ⚠ 反馈 {i + 1} 提交失败: HTTP {response.status_code}")

            except Exception as e:
                print(f"    ⚠ 反馈 {i + 1} 提交异常: {e}")

        # 获取反馈统计，验证去重
        time.sleep(2)  # 等待数据写入

        stats_response = requests.get(f"{self.BASE_URL}/api/feedback_stats")
        if stats_response.status_code == 200:
            stats_data = stats_response.json()
            if stats_data.get('success'):
                total_raw = stats_data.get('total_all_raw', 0)
                total_unique = stats_data.get('total_all_unique', 0)
                duplicate_count = stats_data.get('duplicate_count', 0)

                print(f"\n  反馈统计:")
                print(f"    原始数据: {total_raw} 条")
                print(f"    去重后: {total_unique} 条")
                print(f"    重复数据: {duplicate_count} 条")

                if duplicate_count > 0:
                    print(f"  ✓ 去重功能正常工作")
                else:
                    print(f"  ⚠ 未检测到重复数据")
            else:
                print(f"  ⚠ 获取反馈统计失败: {stats_data.get('error', '未知错误')}")
        else:
            print(f"  ⚠ 无法获取反馈统计")

        print("\n  ✓ 反馈去重功能测试完成")

    def test_6_training_charts_generation(self):
        """测试训练图表生成功能"""
        print("\n[测试6] 训练图表生成功能测试")

        # 获取训练历史
        response = requests.get(f"{self.BASE_URL}/api/training_history")
        if response.status_code != 200:
            print("  ⚠ 无法获取训练历史，跳过此测试")
            return

        data = response.json()
        if not data.get('success'):
            print(f"  ⚠ 获取训练历史失败: {data.get('error', '未知错误')}")
            return

        history = data.get('history', [])
        chart_data = data.get('chart_data', {})

        print(f"  训练历史记录数: {len(history)}")

        if history:
            # 检查图表数据
            timeline = chart_data.get('timeline', [])
            val_accuracies = chart_data.get('val_accuracies', [])
            train_accuracies = chart_data.get('train_accuracies', [])
            losses = chart_data.get('losses', [])
            model_types = chart_data.get('model_types', [])

            print(f"  图表数据:")
            print(f"    时间线点数: {len(timeline)}")
            print(f"    验证准确率点数: {len(val_accuracies)}")
            print(f"    训练准确率点数: {len(train_accuracies)}")
            print(f"    损失值点数: {len(losses)}")
            print(f"    模型类型数: {len(model_types)}")

            # 检查数据一致性
            if (len(timeline) == len(val_accuracies) == len(train_accuracies) ==
                    len(losses) == len(model_types)):
                print(f"  ✓ 图表数据一致性验证通过")

                # 检查数据有效性
                if len(val_accuracies) > 0:
                    valid_accuracies = [acc for acc in val_accuracies if 0 <= acc <= 1]
                    if len(valid_accuracies) == len(val_accuracies):
                        print(f"  ✓ 准确率数据有效性验证通过")
                    else:
                        print(f"  ⚠ 准确率数据有效性验证失败")
            else:
                print(f"  ⚠ 图表数据一致性验证失败")

            # 检查模型类型信息
            model_types_info = data.get('model_types', {})
            if model_types_info:
                print(f"  支持的模型类型:")
                for model_id, model_info in model_types_info.items():
                    print(f"    - {model_info.get('name', model_id)}")
        else:
            print(f"  ⚠ 没有训练历史数据")

        # 测试图表生成接口
        print(f"\n  测试图表生成接口...")

        charts_response = requests.get(f"{self.BASE_URL}/api/training_charts")
        if charts_response.status_code == 200:
            charts_data = charts_response.json()
            if charts_data.get('success'):
                charts = charts_data.get('charts', {})
                has_charts = charts_data.get('has_charts', False)

                print(f"    图表生成状态: {'成功' if has_charts else '失败'}")
                print(f"    生成图表数: {len(charts)}")

                if charts:
                    for chart_name in charts.keys():
                        print(f"    生成图表: {chart_name}")
                    print(f"  ✓ 图表生成功能正常工作")
                else:
                    print(f"  ⚠ 未生成图表，可能没有足够的训练数据")
            else:
                print(f"  ⚠ 图表生成失败: {charts_data.get('error', '未知错误')}")
        else:
            print(f"  ⚠ 图表生成接口调用失败: HTTP {charts_response.status_code}")

        print("\n  ✓ 训练图表生成功能测试完成")

    @classmethod
    def tearDownClass(cls):
        """测试类清理"""
        print("\n" + "=" * 60)
        print("模型功能测试完成")
        print("=" * 60)


def run_model_tests():
    """运行模型测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(ModelTestCase)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 输出测试统计
    print("\n" + "=" * 60)
    print("模型测试统计:")
    print(f"  运行测试数: {result.testsRun}")
    print(f"  成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  失败: {len(result.failures)}")
    print(f"  错误: {len(result.errors)}")
    print("=" * 60)

    return result.wasSuccessful()


def create_test_feedback_data():
    """创建测试反馈数据文件"""
    print("\n创建测试反馈数据...")

    # 创建反馈目录
    FEEDBACK_DIR.mkdir(exist_ok=True)

    # 创建测试反馈数据
    test_data = [
        {
            'text': '酒店服务很好，房间干净整洁',
            'choice': '符合',
            'predicted_label': 1,
            'timestamp': '2024-01-01 10:00:00'
        },
        {
            'text': '位置偏僻，交通不方便',
            'choice': '不符合',
            'predicted_label': 1,
            'timestamp': '2024-01-01 11:00:00'
        },
        {
            'text': '早餐丰富，种类很多',
            'choice': '符合',
            'predicted_label': 1,
            'timestamp': '2024-01-01 12:00:00'
        }
    ]

    # 保存为CSV文件
    df = pd.DataFrame(test_data)
    test_file = FEEDBACK_DIR / 'test_feedback.csv'
    df.to_csv(test_file, index=False, encoding='utf-8-sig')

    print(f"测试反馈数据已创建: {test_file}")
    print(f"数据条数: {len(df)}")


if __name__ == '__main__':
    # 创建测试反馈数据
    create_test_feedback_data()

    # 运行模型测试
    success = run_model_tests()

    # 根据测试结果返回退出码
    sys.exit(0 if success else 1)