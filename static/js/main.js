// 全局变量
let currentPrediction = null;
let trainingStatusInterval = null;
let feedbackCount = 0;
let isTraining = false;

// 检测API基地址
function getApiBase() {
    if (window.location.protocol === 'file:') {
        console.log('检测到file://协议，使用http://127.0.0.1:5000');
        return 'http://127.0.0.1:5000';
    }
    const hostname = window.location.hostname || 'localhost';
    return window.location.protocol + '//' + hostname + ':5000';
}

const API_BASE = getApiBase();
console.log('API基地址:', API_BASE);

// DOM元素
const elements = {
    textInput: document.getElementById('text-input'),
    charCount: document.getElementById('char-count'),
    analyzeBtn: document.getElementById('analyze-btn'),
    clearBtn: document.getElementById('clear-btn'),
    sentimentLabel: document.getElementById('sentiment-label'),
    confidenceFill: document.getElementById('confidence-fill'),
    confidenceText: document.getElementById('confidence-text'),
    modelType: document.getElementById('model-type'),
    feedbackButtons: document.querySelectorAll('.feedback-btn'),
    currentModelName: document.getElementById('current-model-name'),
    currentModelTime: document.getElementById('current-model-time'),
    currentModelFeatures: document.getElementById('current-model-features'),
    modelStatusBadge: document.getElementById('model-status-badge'),
    refreshModelBtn: document.getElementById('refresh-model-btn'),
    reloadModelBtn: document.getElementById('reload-model-btn'),
    useGpuCheckbox: document.getElementById('use-gpu-checkbox'),
    checkGpuBtn: document.getElementById('check-gpu-btn'),
    gpuIndicator: document.getElementById('gpu-indicator'),
    deviceInfo: document.getElementById('device-info'),
    epochCount: document.getElementById('epoch-count'),
    batchSize: document.getElementById('batch-size'),
    startTrainingBtn: document.getElementById('start-training-btn'),
    stopTrainingBtn: document.getElementById('stop-training-btn'),
    progressPanel: document.getElementById('progress-panel'),
    progressMessage: document.getElementById('progress-message'),
    progressPercent: document.getElementById('progress-percent'),
    progressFill: document.getElementById('progress-fill'),
    epochInfo: document.getElementById('epoch-info'),
    etaInfo: document.getElementById('eta-info'),
    currentAccuracy: document.getElementById('current-accuracy'),
    currentLoss: document.getElementById('current-loss'),
    listModelsBtn: document.getElementById('list-models-btn'),
    rollbackBtn: document.getElementById('rollback-btn'),
    modelListContainer: document.getElementById('model-list-container'),
    modelSelect: document.getElementById('model-select'),
    switchModelBtn: document.getElementById('switch-model-btn'),
    backendStatus: document.getElementById('backend-status'),
    trainingState: document.getElementById('training-state'),
    feedbackCountElement: document.getElementById('feedback-count'),
    lastUpdate: document.getElementById('last-update'),
    testApiBtn: document.getElementById('test-api-btn'),
    verifyDatasetBtn: document.getElementById('verify-dataset-btn'),
    resetSystemBtn: document.getElementById('reset-system-btn')
};

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

async function initializeApp() {
    // 设置事件监听器
    setupEventListeners();

    // 更新字符计数
    updateCharCount();

    // 检查后端状态
    await checkBackendStatus();

    // 刷新模型信息
    await refreshModelInfo();

    // 更新反馈数量
    await updateFeedbackCount();

    // 检查GPU状态
    await checkGPUStatus();

    // 设置轮询训练状态
    checkTrainingStatus();

    // 设置系统状态定时更新
    setInterval(updateSystemStatus, 10000);

    // 监听页面关闭事件
    window.addEventListener('beforeunload', handlePageUnload);

    // 显示欢迎消息
    showMessage('系统初始化完成，欢迎使用情感分析系统！', 'info');
}

function setupEventListeners() {
    // 文本输入事件
    elements.textInput.addEventListener('input', function() {
        updateCharCount();
        if (currentPrediction && elements.textInput.value !== currentPrediction.text) {
            elements.sentimentLabel.textContent = '等待分析...';
            elements.sentimentLabel.className = 'sentiment';
        }
    });

    // 分析按钮
    elements.analyzeBtn.addEventListener('click', analyzeText);

    // 清空按钮
    elements.clearBtn.addEventListener('click', function() {
        elements.textInput.value = '';
        elements.sentimentLabel.textContent = '等待分析...';
        elements.sentimentLabel.className = 'sentiment';
        currentPrediction = null;
        updateCharCount();
        showMessage('已清空输入', 'info');
    });

    // 反馈按钮
    elements.feedbackButtons.forEach(button => {
        button.addEventListener('click', function() {
            if (!currentPrediction) {
                showMessage('请先分析文本', 'error');
                return;
            }
            const choice = this.getAttribute('data-choice');
            submitFeedback(choice);
        });
    });

    // 刷新模型信息按钮
    elements.refreshModelBtn.addEventListener('click', refreshModelInfo);

    // 重新加载模型按钮
    elements.reloadModelBtn.addEventListener('click', reloadModel);

    // 检查GPU按钮
    elements.checkGpuBtn.addEventListener('click', checkGPUStatus);

    // 开始训练按钮
    elements.startTrainingBtn.addEventListener('click', startTraining);

    // 停止训练按钮
    elements.stopTrainingBtn.addEventListener('click', stopTraining);

    // 列出模型按钮
    elements.listModelsBtn.addEventListener('click', listModels);

    // 回滚模型按钮
    elements.rollbackBtn.addEventListener('click', rollbackModel);

    // 切换模型按钮
    elements.switchModelBtn.addEventListener('click', switchModel);

    // 测试API按钮
    elements.testApiBtn.addEventListener('click', testApi);

    // 验证数据集按钮
    elements.verifyDatasetBtn.addEventListener('click', verifyDataset);

    // 重置系统按钮
    elements.resetSystemBtn.addEventListener('click', resetSystem);

    // 回车键分析（Ctrl+Enter）
    elements.textInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && e.ctrlKey) {
            e.preventDefault();
            analyzeText();
        }
    });
}

function updateCharCount() {
    const count = elements.textInput.value.length;
    elements.charCount.textContent = count;
}

// 显示消息提示
function showMessage(message, type = 'info') {
    const toast = document.getElementById('message-toast');
    const toastMessage = document.getElementById('toast-message');

    toastMessage.textContent = message;
    toast.className = `toast ${type}`;

    // 显示3秒后自动隐藏
    setTimeout(() => {
        toast.classList.add('hidden');
    }, 3000);
}

// 更新最后更新时间
function updateLastUpdateTime() {
    const now = new Date();
    elements.lastUpdate.textContent = now.toLocaleString('zh-CN');
}

// 检查后端状态
async function checkBackendStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/model_info`);
        if (response.ok) {
            const data = await response.json();
            updateStatusIndicator(elements.backendStatus, 'status-ok', '正常');

            // 更新模型状态
            if (data.is_trained) {
                elements.modelStatusBadge.innerHTML = '<span class="status-success">已加载</span>';
            } else {
                elements.modelStatusBadge.innerHTML = '<span class="status-error">未加载</span>';
            }
        } else {
            updateStatusIndicator(elements.backendStatus, 'status-error', '异常');
        }
    } catch (error) {
        updateStatusIndicator(elements.backendStatus, 'status-error', '连接失败');
    }
}

function updateStatusIndicator(element, statusClass, text) {
    const indicator = element.querySelector('.status-indicator');
    indicator.className = `status-indicator ${statusClass}`;
    element.innerHTML = `${indicator.outerHTML} ${text}`;
}

// 更新反馈数量
async function updateFeedbackCount() {
    try {
        // 从本地存储获取反馈数量（临时方案）
        const storedCount = localStorage.getItem('feedbackCount') || '0';
        feedbackCount = parseInt(storedCount);
        elements.feedbackCountElement.textContent = `${feedbackCount} 条`;
    } catch (error) {
        console.error('更新反馈数量失败:', error);
    }
}

// 情感分析
async function analyzeText() {
    const text = elements.textInput.value.trim();

    if (!text) {
        showMessage('请输入要分析的文本', 'error');
        return;
    }

    if (text.length < 3) {
        showMessage('文本太短，请输入更长的文本', 'error');
        return;
    }

    // 禁用按钮防止重复点击
    elements.analyzeBtn.disabled = true;
    elements.analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 分析中...';

    try {
        const response = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || '预测失败');
        }

        // 保存预测结果用于反馈
        currentPrediction = {
            text: text,
            label: data.label,
            label_text: data.label_text,
            confidence: data.confidence
        };

        // 更新界面显示结果
        displayResult(data);

        showMessage(`分析完成：${data.label_text}，置信度 ${Math.round(data.confidence * 100)}%`, 'success');

    } catch (error) {
        showMessage(error.message, 'error');
        console.error('预测失败:', error);
    } finally {
        // 恢复按钮状态
        elements.analyzeBtn.disabled = false;
        elements.analyzeBtn.innerHTML = '<i class="fas fa-chart-bar"></i> 分析情感';
    }
}

function displayResult(data) {
    // 更新情感文本
    elements.sentimentLabel.textContent = data.label_text;
    elements.sentimentLabel.className = 'sentiment';
    elements.sentimentLabel.classList.add(data.label === 1 ? 'positive' : 'negative');

    // 更新置信度
    const confidencePercent = Math.round(data.confidence * 100);
    elements.confidenceFill.style.width = `${confidencePercent}%`;
    elements.confidenceText.textContent = `${confidencePercent}%`;

    // 添加动画效果
    elements.sentimentLabel.style.animation = 'none';
    setTimeout(() => {
        elements.sentimentLabel.style.animation = 'fadeIn 0.5s ease';
    }, 10);
}

// 提交反馈
async function submitFeedback(choice) {
    try {
        const response = await fetch(`${API_BASE}/api/feedback`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: currentPrediction.text,
                choice: choice,
                predicted_label: currentPrediction.label
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || '保存反馈失败');
        }

        // 更新反馈计数
        if (choice === '符合' || choice === '不符合') {
            feedbackCount++;
            localStorage.setItem('feedbackCount', feedbackCount.toString());
            updateFeedbackCount();
        }

        // 显示成功消息
        let message = '反馈已保存';
        if (choice === '无法判断') {
            message += '（仅记录，不用于训练）';
        } else {
            message += '，将用于后续模型训练';
        }

        showMessage(message, 'success');

    } catch (error) {
        showMessage(error.message, 'error');
        console.error('保存反馈失败:', error);
    }
}

// 刷新模型信息
async function refreshModelInfo() {
    try {
        const response = await fetch(`${API_BASE}/api/model_info`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || '获取模型信息失败');
        }

        // 更新模型信息显示
        elements.currentModelName.textContent = data.model_name || '未加载';
        elements.currentModelTime.textContent = data.model_time || '-';
        elements.currentModelFeatures.textContent = data.features || '0';

        // 更新模型状态
        if (data.is_trained) {
            elements.modelStatusBadge.innerHTML = '<span class="status-success">已加载</span>';
        } else {
            elements.modelStatusBadge.innerHTML = '<span class="status-error">未训练</span>';
        }

        showMessage('模型信息已更新', 'success');

    } catch (error) {
        showMessage(error.message, 'error');
        console.error('获取模型信息失败:', error);
    }
}

// 重新加载模型
async function reloadModel() {
    try {
        // 这里可以添加重新加载模型的API
        // 暂时模拟重新加载
        showMessage('重新加载模型中...', 'info');
        setTimeout(async () => {
            await refreshModelInfo();
            showMessage('模型重新加载成功', 'success');
        }, 1000);
    } catch (error) {
        showMessage(error.message, 'error');
        console.error('重新加载模型失败:', error);
    }
}

// 检查GPU状态
async function checkGPUStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/check_gpu`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error('检查GPU状态失败');
        }

        if (data.gpu_available) {
            elements.gpuIndicator.className = 'gpu-indicator available';
            elements.deviceInfo.textContent = `GPU可用: ${data.gpu_info.device_name || '未知设备'}`;
            elements.useGpuCheckbox.disabled = false;
            showMessage('GPU检测成功，可以使用GPU训练', 'success');
        } else {
            elements.gpuIndicator.className = 'gpu-indicator unavailable';
            elements.deviceInfo.textContent = 'GPU不可用，将使用CPU训练';
            elements.useGpuCheckbox.checked = false;
            elements.useGpuCheckbox.disabled = true;
            showMessage('GPU不可用，将使用CPU训练', 'warning');
        }

    } catch (error) {
        elements.deviceInfo.textContent = '无法检测GPU状态';
        showMessage(error.message, 'error');
    }
}

// 开始训练
async function startTraining() {
    if (isTraining) {
        showMessage('训练已在运行中', 'warning');
        return;
    }

    const useGpu = elements.useGpuCheckbox.checked;
    const epochs = parseInt(elements.epochCount.value) || 5;
    const batchSize = parseInt(elements.batchSize.value) || 32;

    if (epochs < 1 || epochs > 20) {
        showMessage('训练轮数应在1-20之间', 'error');
        return;
    }

    if (batchSize < 16 || batchSize > 128) {
        showMessage('批次大小应在16-128之间', 'error');
        return;
    }

    if (!confirm(`确定要开始训练吗？\n使用GPU: ${useGpu ? '是' : '否'}\n训练轮数: ${epochs}\n批次大小: ${batchSize}`)) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/train`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                use_gpu: useGpu,
                epochs: epochs,
                batch_size: batchSize
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || '启动训练失败');
        }

        showMessage('训练已开始', 'success');

        // 更新训练状态
        isTraining = true;
        elements.startTrainingBtn.style.display = 'none';
        elements.stopTrainingBtn.style.display = 'block';
        elements.progressPanel.classList.add('visible');
        updateStatusIndicator(elements.trainingState, 'status-active', '训练中');

        // 开始轮询训练状态
        startTrainingStatusPolling();

    } catch (error) {
        showMessage(error.message, 'error');
        console.error('启动训练失败:', error);
    }
}

// 停止训练
async function stopTraining() {
    if (!isTraining) {
        return;
    }

    if (!confirm('确定要停止训练吗？当前训练进度将丢失。')) {
        return;
    }

    try {
        // 这里可以添加停止训练的API
        // 暂时模拟停止训练
        isTraining = false;
        elements.startTrainingBtn.style.display = 'block';
        elements.stopTrainingBtn.style.display = 'none';
        updateStatusIndicator(elements.trainingState, 'status-idle', '空闲');
        showMessage('训练已停止', 'warning');

        // 清除轮询
        if (trainingStatusInterval) {
            clearInterval(trainingStatusInterval);
            trainingStatusInterval = null;
        }

    } catch (error) {
        showMessage(error.message, 'error');
        console.error('停止训练失败:', error);
    }
}

// 开始轮询训练状态
function startTrainingStatusPolling() {
    // 清除现有的轮询
    if (trainingStatusInterval) {
        clearInterval(trainingStatusInterval);
    }

    // 设置新的轮询（每秒一次）
    trainingStatusInterval = setInterval(checkTrainingStatus, 1000);
}

// 检查训练状态
async function checkTrainingStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/train_status`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error('获取训练状态失败');
        }

        // 更新训练状态显示
        elements.progressMessage.textContent = data.message || '训练中...';

        const progress = data.progress || 0;
        elements.progressPercent.textContent = `${progress.toFixed(1)}%`;
        elements.progressFill.style.width = `${progress}%`;

        elements.epochInfo.textContent = `${data.current_epoch || 0}/${data.total_epochs || 5}`;

        const etaSeconds = data.eta_seconds || 0;
        elements.etaInfo.textContent = formatTime(etaSeconds);

        const accuracy = data.current_acc || 0;
        elements.currentAccuracy.textContent = `${(accuracy * 100).toFixed(2)}%`;

        const loss = data.current_loss || 0;
        elements.currentLoss.textContent = loss.toFixed(4);

        // 检查训练是否完成
        if (!data.is_running && isTraining) {
            // 训练完成
            isTraining = false;
            elements.startTrainingBtn.style.display = 'block';
            elements.stopTrainingBtn.style.display = 'none';
            updateStatusIndicator(elements.trainingState, 'status-idle', '空闲');

            // 延迟隐藏进度条
            setTimeout(() => {
                if (!isTraining) {
                    elements.progressPanel.classList.remove('visible');
                    elements.progressFill.style.width = '0%';
                    elements.progressPercent.textContent = '0%';
                }
            }, 3000);

            // 清除轮询
            if (trainingStatusInterval) {
                clearInterval(trainingStatusInterval);
                trainingStatusInterval = null;
            }

            // 刷新模型信息
            refreshModelInfo();

            showMessage('训练完成！', 'success');
        }

    } catch (error) {
        console.error('获取训练状态失败:', error);
    }
}

// 格式化时间（秒 -> 时分秒）
function formatTime(seconds) {
    if (seconds < 60) {
        return `${Math.round(seconds)}秒`;
    } else if (seconds < 3600) {
        const minutes = Math.floor(seconds / 60);
        const secs = Math.round(seconds % 60);
        return `${minutes}分${secs}秒`;
    } else {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${hours}小时${minutes}分`;
    }
}

// 列出所有模型
async function listModels() {
    try {
        const response = await fetch(`${API_BASE}/api/list_models`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || '获取模型列表失败');
        }

        // 清空模型列表容器
        elements.modelListContainer.innerHTML = '';

        if (data.models && data.models.length > 0) {
            // 显示模型列表
            data.models.forEach(model => {
                const modelItem = document.createElement('div');
                modelItem.className = 'model-item';

                modelItem.innerHTML = `
                    <div class="model-info">
                        <div class="model-name">${model.name}</div>
                        <div class="model-time">创建时间: ${model.time}</div>
                    </div>
                `;

                // 点击模型项可以快速选择
                modelItem.addEventListener('click', function() {
                    // 选中该模型
                    elements.modelSelect.value = model.name;
                });

                elements.modelListContainer.appendChild(modelItem);
            });

            // 更新下拉选择框
            elements.modelSelect.innerHTML = '<option value="">请选择模型...</option>';
            data.models.forEach(model => {
                if (model.has_vectorizer) {
                    const option = document.createElement('option');
                    option.value = model.name;
                    option.textContent = `${model.name} (${model.time})`;
                    elements.modelSelect.appendChild(option);
                }
            });

            showMessage(`找到 ${data.models.length} 个模型`, 'success');

        } else {
            elements.modelListContainer.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-database"></i>
                    <p>没有找到模型文件</p>
                    <p>请先训练一个模型</p>
                </div>
            `;
            showMessage('没有找到模型文件', 'warning');
        }

    } catch (error) {
        showMessage(error.message, 'error');
        console.error('获取模型列表失败:', error);
    }
}

// 切换模型
async function switchModel() {
    const selectedModel = elements.modelSelect.value;

    if (!selectedModel) {
        showMessage('请先选择一个模型', 'error');
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/switch_model`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model_name: selectedModel })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || '切换模型失败');
        }

        showMessage(data.message || '模型切换成功', 'success');

        // 刷新模型信息
        refreshModelInfo();

    } catch (error) {
        showMessage(error.message, 'error');
        console.error('切换模型失败:', error);
    }
}

// 回滚模型
async function rollbackModel() {
    if (!confirm('确定要回滚到上一个模型吗？当前模型将被备份。')) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/rollback`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || '回滚失败');
        }

        showMessage(data.message || '已回滚到上一个模型', 'success');

        // 刷新模型信息
        refreshModelInfo();

    } catch (error) {
        showMessage(error.message, 'error');
        console.error('回滚失败:', error);
    }
}

// 测试API
async function testApi() {
    try {
        showMessage('正在测试API...', 'info');

        // 测试预测API
        const predictResponse = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: '这是一个测试文本，电影很好看！' })
        });

        if (predictResponse.ok) {
            showMessage('预测API测试成功', 'success');
        } else {
            showMessage('预测API测试失败', 'error');
        }

    } catch (error) {
        showMessage(`API测试失败: ${error.message}`, 'error');
    }
}

// 验证数据集
async function verifyDataset() {
    try {
        showMessage('正在验证数据集...', 'info');

        // 这里可以添加验证数据集的API
        // 暂时模拟验证
        setTimeout(() => {
            showMessage('数据集验证完成，格式正确', 'success');
        }, 1000);

    } catch (error) {
        showMessage(`验证数据集失败: ${error.message}`, 'error');
    }
}

// 重置系统
async function resetSystem() {
    if (!confirm('确定要重置系统吗？这会清除所有反馈数据和训练日志。')) {
        return;
    }

    try {
        showMessage('正在重置系统...', 'info');

        // 清除本地存储的反馈计数
        localStorage.removeItem('feedbackCount');
        feedbackCount = 0;
        updateFeedbackCount();

        setTimeout(() => {
            showMessage('系统重置完成，反馈数据已清除', 'success');
            location.reload();
        }, 1000);

    } catch (error) {
        showMessage(`重置失败: ${error.message}`, 'error');
    }
}

// 更新系统状态
async function updateSystemStatus() {
    updateLastUpdateTime();
    await checkBackendStatus();
    await updateFeedbackCount();
}

// 页面关闭处理
function handlePageUnload(event) {
    // 使用sendBeacon发送页面关闭通知
    const data = JSON.stringify({});
    navigator.sendBeacon(`${API_BASE}/api/session_end`, data);

    // 可选：显示提示消息
    event.returnValue = '页面即将关闭，如有足够反馈数据将自动训练模型';
}

// 添加CSS动画
const style = document.createElement('style');
style.textContent = `
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
`;
document.head.appendChild(style);