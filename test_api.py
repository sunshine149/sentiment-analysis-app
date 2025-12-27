#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API接口测试脚本
用于测试情感分析应用的所有API接口
"""

import requests
import json
import time
import unittest
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BASE_DIR


class APITestCase(unittest.TestCase):
    """API测试用例类"""

    BASE_URL = "http://localhost:5000"

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        print("\n" + "=" * 60)
        print("开始API接口测试")
        print("=" * 60)

        # 等待服务启动
        max_retries = 10
        for i in range(max_retries):
            try:
                response = requests.get(f"{cls.BASE_URL}/health", timeout=5)
                if response.status_code == 200:
                    print("✓ 服务已启动，可以开始测试")
                    return
            except:
                if i == max_retries - 1:
                    print(f"✗ 服务未启动，请在 {cls.BASE_URL} 启动服务后重试")
                    sys.exit(1)
                print(f"等待服务启动...({i + 1}/{max_retries})")
                time.sleep(3)

    def test_1_health_check(self):
        """测试健康检查接口"""
        print("\n[测试1] 健康检查接口")
        response = requests.get(f"{self.BASE_URL}/health")

        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn('status', data)
        self.assertIn('model_loaded', data)
        self.assertIn('timestamp', data)

        print(f"  状态: {data['status']}")
        print(f"  模型加载: {data['model_loaded']}")
        print(f"  时间戳: {data['timestamp']}")
        print("  ✓ 健康检查通过")

    def test_2_model_info(self):
        """测试模型信息接口"""
        print("\n[测试2] 模型信息接口")
        response = requests.get(f"{self.BASE_URL}/api/model_info")

        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn('model_name', data)
        self.assertIn('model_time', data)
        self.assertIn('features', data)
        self.assertIn('is_trained', data)

        print(f"  模型名称: {data['model_name']}")
        print(f"  模型时间: {data['model_time']}")
        print(f"  特征数量: {data['features']}")
        print(f"  是否训练: {data['is_trained']}")
        print("  ✓ 模型信息获取成功")

    def test_3_predict_api(self):
        """测试预测接口"""
        print("\n[测试3] 情感预测接口")

        test_cases = [
            {
                "text": "这家酒店的服务非常周到，房间干净整洁，入住体验很好！",
                "expected_label_text": "正面"
            },
            {
                "text": "房间太小了，设施陈旧，服务态度也不好，非常失望。",
                "expected_label_text": "负面"
            },
            {
                "text": "位置很方便，早餐丰富，就是隔音效果一般。",
                "expected_label_text": "正面"  # 虽然中性，但倾向于正面
            }
        ]

        for i, test_case in enumerate(test_cases):
            print(f"\n  测试用例 {i + 1}:")
            print(f"    文本: {test_case['text'][:30]}...")

            response = requests.post(
                f"{self.BASE_URL}/api/predict",
                json={"text": test_case["text"]},
                headers={"Content-Type": "application/json"}
            )

            self.assertEqual(response.status_code, 200)

            data = response.json()
            self.assertIn('label', data)
            self.assertIn('label_text', data)
            self.assertIn('confidence', data)

            confidence = data['confidence']
            label_text = data['label_text']

            print(f"    预测结果: {label_text}")
            print(f"    置信度: {confidence:.2%}")

            # 置信度应该大于0.5
            self.assertGreater(confidence, 0.5)

            # 对于明确的测试用例，验证标签
            if test_case.get('expected_label_text'):
                self.assertEqual(label_text, test_case['expected_label_text'])
                print(f"    ✓ 标签验证通过")

        print("\n  ✓ 所有预测测试通过")

    def test_4_feedback_api(self):
        """测试反馈接口"""
        print("\n[测试4] 用户反馈接口")

        feedback_cases = [
            {
                "text": "酒店服务很好，下次还会入住！",
                "choice": "符合",
                "predicted_label": 1
            },
            {
                "text": "房间太吵了，根本睡不好",
                "choice": "不符合",
                "predicted_label": 1
            },
            {
                "text": "这个评价很难判断好坏",
                "choice": "无法判断",
                "predicted_label": 0
            }
        ]

        for i, feedback in enumerate(feedback_cases):
            print(f"\n  反馈用例 {i + 1}:")
            print(f"    文本: {feedback['text'][:20]}...")
            print(f"    选择: {feedback['choice']}")

            response = requests.post(
                f"{self.BASE_URL}/api/feedback",
                json=feedback,
                headers={"Content-Type": "application/json"}
            )

            self.assertEqual(response.status_code, 200)

            data = response.json()
            self.assertIn('message', data)

            print(f"    响应: {data['message']}")
            print(f"    ✓ 反馈提交成功")

        print("\n  ✓ 所有反馈测试通过")

    def test_5_model_list_api(self):
        """测试模型列表接口"""
        print("\n[测试5] 模型列表接口")

        response = requests.get(f"{self.BASE_URL}/api/list_models")

        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn('models', data)
        self.assertIn('current_model', data)

        models = data['models']
        print(f"  发现 {len(models)} 个模型")

        if models:
            for i, model in enumerate(models[:3]):  # 只显示前3个
                print(f"    模型{i + 1}: {model['name']}")
                print(f"      时间: {model['time']}")
                print(f"      类型: {model.get('model_type', 'N/A')}")
                print(f"      有向量器: {model['has_vectorizer']}")

        print("  ✓ 模型列表获取成功")

    def test_6_check_gpu_api(self):
        """测试GPU检查接口"""
        print("\n[测试6] GPU检查接口")

        response = requests.get(f"{self.BASE_URL}/api/check_gpu")

        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn('gpu_available', data)
        self.assertIn('gpu_info', data)

        gpu_available = data['gpu_available']
        gpu_info = data['gpu_info']

        if gpu_available:
            print(f"  GPU可用:")
            print(f"    设备数量: {gpu_info.get('device_count', 'N/A')}")
            print(f"    当前设备: {gpu_info.get('current_device', 'N/A')}")
            print(f"    设备名称: {gpu_info.get('device_name', 'N/A')}")
        else:
            print("  GPU不可用")

        print("  ✓ GPU检查完成")

    def test_7_training_options_api(self):
        """测试训练选项接口"""
        print("\n[测试7] 训练选项接口")

        response = requests.get(f"{self.BASE_URL}/api/train_options")

        if response.status_code == 200:
            data = response.json()
            print(f"  GPU可用: {data.get('gpu_available', False)}")
            print(f"  最大轮次: {data.get('max_epochs', 'N/A')}")
            print("  ✓ 训练选项获取成功")
        else:
            print("  ⚠ 训练选项接口可能未实现")

    def test_8_feedback_stats_api(self):
        """测试反馈统计接口"""
        print("\n[测试8] 反馈统计接口")

        response = requests.get(f"{self.BASE_URL}/api/feedback_stats")

        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn('success', data)

        if data['success']:
            print(f"  原始反馈总数: {data.get('total_all_raw', 0)}")
            print(f"  去重后总数: {data.get('total_all_unique', 0)}")
            print(f"  有效反馈(去重): {data.get('total_valid_unique', 0)}")
            print(f"  重复数据: {data.get('duplicate_count', 0)}")
            print(f"  反馈文件数: {len(data.get('feedback_files', []))}")
            print("  ✓ 反馈统计获取成功")
        else:
            print(f"  ⚠ 获取反馈统计失败: {data.get('error', '未知错误')}")

    def test_9_model_types_api(self):
        """测试模型类型接口"""
        print("\n[测试9] 模型类型接口")

        response = requests.get(f"{self.BASE_URL}/api/model_types")

        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                model_types = data.get('model_types', {})
                print(f"  支持的模型类型: {len(model_types)} 种")

                for model_id, model_info in model_types.items():
                    print(f"    - {model_info.get('name', model_id)}")
                    print(f"      描述: {model_info.get('description', 'N/A')}")

                print("  ✓ 模型类型获取成功")
            else:
                print(f"  ⚠ 获取模型类型失败: {data.get('error', '未知错误')}")
        else:
            print("  ⚠ 模型类型接口可能未实现")

    def test_10_training_history_api(self):
        """测试训练历史接口"""
        print("\n[测试10] 训练历史接口")

        response = requests.get(f"{self.BASE_URL}/api/training_history")

        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                history = data.get('history', [])
                chart_data = data.get('chart_data', {})

                print(f"  训练历史记录数: {len(history)}")
                print(f"  图表数据点数: {len(chart_data.get('timeline', []))}")

                if history:
                    latest = history[-1]
                    print(f"  最近训练:")
                    print(f"    时间: {latest.get('timestamp', 'N/A')}")
                    print(f"    模型类型: {latest.get('model_type', 'N/A')}")
                    print(f"    最终准确率: {latest.get('final_val_accuracy', 0):.2%}")
                    print(f"    反馈数量: {latest.get('feedback_count', 0)}")

                print("  ✓ 训练历史获取成功")
            else:
                print(f"  ⚠ 获取训练历史失败: {data.get('error', '未知错误')}")
        else:
            print("  ⚠ 训练历史接口可能未实现")

    def test_11_verify_dataset_api(self):
        """测试数据集验证接口"""
        print("\n[测试11] 数据集验证接口")

        response = requests.get(f"{self.BASE_URL}/api/verify_dataset")

        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"  验证结果: {data.get('message', 'N/A')}")
                print(f"  样本行数: {len(data.get('sample_rows', []))}")
                print("  ✓ 数据集验证通过")
            else:
                print(f"  ⚠ 数据集验证失败: {data.get('error', '未知错误')}")
        else:
            print("  ⚠ 数据集验证接口可能未实现")

    def test_12_test_api_endpoint(self):
        """测试内置测试接口"""
        print("\n[测试12] 内置测试接口")

        response = requests.post(f"{self.BASE_URL}/api/test", json={})

        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                results = data.get('results', [])
                print(f"  测试文本数量: {len(results)}")

                for i, result in enumerate(results[:3]):  # 只显示前3个
                    print(f"    测试{i + 1}: {result.get('text', '')[:20]}...")
                    print(f"      预测: {result.get('label_text', 'N/A')}")
                    print(f"      置信度: {result.get('confidence', 0):.2%}")

                print("  ✓ 内置测试接口调用成功")
            else:
                print(f"  ⚠ 内置测试接口失败: {data.get('error', '未知错误')}")
        else:
            print("  ⚠ 内置测试接口可能未实现")

    def test_13_threshold_info_api(self):
        """测试训练阈值信息接口"""
        print("\n[测试13] 训练阈值信息接口")

        response = requests.get(f"{self.BASE_URL}/api/training_threshold_info")

        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                threshold_info = data.get('threshold_info', {})
                print(f"  当前阈值: {threshold_info.get('threshold', 0)}")
                print(f"  上次训练反馈数: {threshold_info.get('feedback_count', 0)}")
                print(f"  更新时间: {threshold_info.get('updated_time', 'N/A')}")
                print(f"  当前反馈数: {data.get('current_feedback_count', 0)}")
                print(f"  下次训练目标: {data.get('next_training_target', 0)}")
                print(f"  还需反馈数: {data.get('remaining_for_next_training', 0)}")
                print("  ✓ 阈值信息获取成功")
            else:
                print(f"  ⚠ 获取阈值信息失败: {data.get('error', '未知错误')}")
        else:
            print("  ⚠ 阈值信息接口可能未实现")

    @classmethod
    def tearDownClass(cls):
        """测试类清理"""
        print("\n" + "=" * 60)
        print("API接口测试完成")
        print("=" * 60)


def run_all_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(APITestCase)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 输出测试统计
    print("\n" + "=" * 60)
    print("测试统计:")
    print(f"  运行测试数: {result.testsRun}")
    print(f"  成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  失败: {len(result.failures)}")
    print(f"  错误: {len(result.errors)}")
    print("=" * 60)

    return result.wasSuccessful()


if __name__ == '__main__':
    # 运行所有测试
    success = run_all_tests()

    # 根据测试结果返回退出码
    sys.exit(0 if success else 1)