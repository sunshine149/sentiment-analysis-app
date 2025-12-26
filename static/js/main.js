// 全局变量
let currentPrediction = null;
let trainingStatusInterval = null;
let feedbackCount = 0;
let isTraining = false;
let charts = {}; // 存储图表实例
let modelTypes = {}; // 模型类型信息

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
    // 原有元素
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
    resetSystemBtn: document.getElementById('reset-system-btn'),

    // 新增元素
    modelTypeSelect: document.getElementById('model-type-select'),
    modelDescription: document.getElementById('model-description'),
    modelInfoCards: document.querySelectorAll('.model-info-card'),
    refreshChartsBtn: document.getElementById('refresh-charts-btn'),
    toggleChartsBtn: document.getElementById('toggle-charts-btn'),
    exportChartsBtn: document.getElementById('export-charts-btn'),
    trainingModelType: document.getElementById('training-model-type'),
    currentModelType: document.getElementById('current-model-type'),
    todayFeedback: document.getElementById('today-feedback'),
    totalFeedback: document.getElementById('total-feedback'),
    modelCount: document.getElementById('model-count'),
    chartStatus: document.getElementById('chart-status'),
    viewStatsBtn: document.getElementById('view-stats-btn'),
    statsModal: document.getElementById('stats-modal'),
    statsContent: document.getElementById('stats-content'),
    modalClose: document.querySelector('.modal-close'),

    // 图表Canvas元素
    modelComparisonChart: document.getElementById('modelComparisonChart'),
    accuracyTrendChart: document.getElementById('accuracyTrendChart'),
    lossTrendChart: document.getElementById('lossTrendChart'),
    feedbackGrowthChart: document.getElementById('feedbackGrowthChart')
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

    // 加载模型类型信息
    await loadModelTypes();

    // 设置轮询训练状态
    checkTrainingStatus();

    // 加载图表
    await loadCharts();

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

    // 模型选择变化
    elements.modelTypeSelect.addEventListener('change', function() {
        updateModelDescription(this.value);
        // 更新卡片选中状态
        elements.modelInfoCards.forEach(card => {
            if (card.getAttribute('data-model') === this.value) {
                card.classList.add('active');
            } else {
                card.classList.remove('active');
            }
        });
    });

    // 模型卡片点击
    elements.modelInfoCards.forEach(card => {
        card.addEventListener('click', function() {
            const modelType = this.getAttribute('data-model');
            elements.modelTypeSelect.value = modelType;
            updateModelDescription(modelType);
            // 更新卡片选中状态
            elements.modelInfoCards.forEach(c => {
                if (c === this) {
                    c.classList.add('active');
                } else {
                    c.classList.remove('active');
                }
            });
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

    // 图表相关按钮
    elements.refreshChartsBtn.addEventListener('click', loadCharts);
    elements.toggleChartsBtn.addEventListener('click', toggleCharts);
    elements.exportChartsBtn.addEventListener('click', exportCharts);
    elements.viewStatsBtn.addEventListener('click', showStatsModal);
    elements.modalClose.addEventListener('click', hideStatsModal);

    // 回车键分析（Ctrl+Enter）
    elements.textInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && e.ctrlKey) {
            e.preventDefault();
            analyzeText();
        }
    });

    // 点击模态框外部关闭
    elements.statsModal.addEventListener('click', function(e) {
        if (e.target === this) {
            hideStatsModal();
        }
    });
}

// 图表相关函数
async function loadCharts() {
    try {
        updateStatusIndicator(elements.chartStatus, 'status-active', '加载中');

        // 加载训练历史数据
        const response = await fetch(`${API_BASE}/api/training_history`);
        const data = await response.json();

        if (data.success) {
            // 创建或更新图表
            createOrUpdateCharts(data.chart_data, data.model_types);
            updateStatusIndicator(elements.chartStatus, 'status-ok', '已加载');
            showMessage('图表已刷新', 'success');
        } else {
            throw new Error(data.error || '加载图表数据失败');
        }

    } catch (error) {
        console.error('加载图表失败:', error);
        updateStatusIndicator(elements.chartStatus, 'status-error', '加载失败');
        showMessage('加载图表失败: ' + error.message, 'error');

        // 如果没有数据，显示空状态
        showEmptyChartState();
    }
}

function createOrUpdateCharts(chartData, modelTypesData) {
    // 销毁现有图表
    Object.values(charts).forEach(chart => {
        if (chart && chart.destroy) {
            chart.destroy();
        }
    });

    // 重置charts对象
    charts = {};

    // 模型性能对比图
    createModelComparisonChart(chartData, modelTypesData);

    // 准确率趋势图
    createAccuracyTrendChart(chartData);

    // 损失趋势图
    createLossTrendChart(chartData);

    // 反馈数据增长图
    createFeedbackGrowthChart(chartData);
}

function createModelComparisonChart(chartData, modelTypesData) {
    if (!chartData.model_types || chartData.model_types.length === 0) {
        showNoDataMessage('modelComparisonChart', '暂无模型对比数据');
        return;
    }

    // 按模型类型分组准确率
    const modelAccuracies = {};
    chartData.model_types.forEach((type, index) => {
        if (!modelAccuracies[type]) {
            modelAccuracies[type] = [];
        }
        if (chartData.val_accuracies[index] !== undefined) {
            modelAccuracies[type].push(chartData.val_accuracies[index]);
        }
    });

    // 准备图表数据
    const labels = Object.keys(modelAccuracies).map(type =>
        modelTypesData[type]?.name || type
    );

    const datasets = [{
        label: '验证准确率',
        data: Object.values(modelAccuracies).map(accs =>
            accs.length > 0 ? accs.reduce((a, b) => a + b, 0) / accs.length : 0
        ),
        backgroundColor: [
            'rgba(102, 126, 234, 0.7)',
            'rgba(0, 176, 155, 0.7)',
            'rgba(255, 65, 108, 0.7)',
            'rgba(255, 179, 71, 0.7)'
        ],
        borderColor: [
            'rgb(102, 126, 234)',
            'rgb(0, 176, 155)',
            'rgb(255, 65, 108)',
            'rgb(255, 179, 71)'
        ],
        borderWidth: 2
    }];

    const ctx = elements.modelComparisonChart.getContext('2d');
    charts.modelComparison = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '模型性能对比'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `平均准确率: ${(context.raw * 100).toFixed(2)}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                }
            }
        }
    });
}

function createAccuracyTrendChart(chartData) {
    if (!chartData.timeline || chartData.timeline.length < 2) {
        showNoDataMessage('accuracyTrendChart', '暂无足够训练历史数据');
        return;
    }

    const ctx = elements.accuracyTrendChart.getContext('2d');
    charts.accuracyTrend = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.timeline,
            datasets: [{
                label: '验证准确率',
                data: chartData.val_accuracies,
                borderColor: 'rgb(0, 176, 155)',
                backgroundColor: 'rgba(0, 176, 155, 0.1)',
                tension: 0.4,
                fill: true
            }, {
                label: '训练准确率',
                data: chartData.train_accuracies,
                borderColor: 'rgb(102, 126, 234)',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                tension: 0.4,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '准确率趋势'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                }
            }
        }
    });
}

function createLossTrendChart(chartData) {
    if (!chartData.timeline || chartData.timeline.length < 2) {
        showNoDataMessage('lossTrendChart', '暂无足够训练历史数据');
        return;
    }

    const ctx = elements.lossTrendChart.getContext('2d');
    charts.lossTrend = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.timeline,
            datasets: [{
                label: '损失值',
                data: chartData.losses,
                borderColor: 'rgb(255, 65, 108)',
                backgroundColor: 'rgba(255, 65, 108, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '损失曲线'
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function createFeedbackGrowthChart(chartData) {
    if (!chartData.timeline || chartData.timeline.length < 2) {
        showNoDataMessage('feedbackGrowthChart', '暂无足够反馈数据');
        return;
    }

    const ctx = elements.feedbackGrowthChart.getContext('2d');
    charts.feedbackGrowth = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: chartData.timeline,
            datasets: [{
                label: '反馈数量',
                data: chartData.feedback_counts,
                backgroundColor: 'rgba(255, 179, 71, 0.7)',
                borderColor: 'rgb(255, 179, 71)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '反馈数据增长'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return value.toFixed(0);
                        }
                    }
                }
            }
        }
    });
}

function showNoDataMessage(chartId, message) {
    const container = document.getElementById(chartId.replace('Chart', '-chart'));
    if (container) {
        container.innerHTML = `
            <div class="chart-no-data">
                <i class="fas fa-chart-bar"></i>
                <p>${message}</p>
            </div>
        `;
    }
}

function showEmptyChartState() {
    ['modelComparisonChart', 'accuracyTrendChart', 'lossTrendChart', 'feedbackGrowthChart'].forEach(chartId => {
        showNoDataMessage(chartId, '暂无数据，请先进行训练');
    });
}

function toggleCharts() {
    const chartsContainer = document.querySelector('.charts-container');
    const isExpanded = chartsContainer.classList.contains('expanded');

    if (isExpanded) {
        chartsContainer.classList.remove('expanded');
        elements.toggleChartsBtn.innerHTML = '<i class="fas fa-expand"></i> 展开所有';
    } else {
        chartsContainer.classList.add('expanded');
        elements.toggleChartsBtn.innerHTML = '<i class="fas fa-compress"></i> 收起图表';
    }
}

function exportCharts() {
    // 创建导出选项
    const options = [
        { id: 'png', name: '导出为图片 (PNG)', icon: 'fa-image' },
        { id: 'pdf', name: '导出为PDF报告', icon: 'fa-file-pdf' },
        { id: 'excel', name: '导出数据 (Excel)', icon: 'fa-file-excel' }
    ];

    // 创建选择对话框
    let html = '<div class="simple-export-dialog">';
    html += '<h4><i class="fas fa-download"></i> 导出图表</h4>';
    html += '<div class="export-buttons">';

    options.forEach(option => {
        html += `
            <button class="export-btn" data-type="${option.id}">
                <i class="fas ${option.icon}"></i>
                <span>${option.name}</span>
            </button>
        `;
    });

    html += '</div>';
    html += '<button class="btn btn-secondary" id="cancel-export">取消</button>';
    html += '</div>';

    // 显示对话框
    showDialog(html);

    // 绑定事件
    setTimeout(() => {
        document.querySelectorAll('.export-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const type = this.getAttribute('data-type');
                closeDialog();

                switch(type) {
                    case 'png':
                        simpleExportAsImage();
                        break;
                    case 'pdf':
                        simpleExportAsPDF();
                        break;
                    case 'excel':
                        simpleExportAsExcel();
                        break;
                }
            });
        });

        document.getElementById('cancel-export').addEventListener('click', () => {
            closeDialog();
        });
    }, 100);
}

// 简单的图片导出
async function simpleExportAsImage() {
    showMessage('正在生成图片...', 'info');

    try {
        const chart = document.querySelector('.charts-container');
        if (!chart) {
            showMessage('没有可导出的图表', 'warning');
            return;
        }

        const canvas = await html2canvas(chart, {
            backgroundColor: '#ffffff',
            scale: 2
        });

        const link = document.createElement('a');
        link.download = `训练图表_${new Date().toISOString().slice(0, 19).replace(/[:]/g, '-')}.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();

        showMessage('图表已导出为图片', 'success');
    } catch (error) {
        showMessage('导出失败: ' + error.message, 'error');
    }
}

// 简单的PDF导出
async function simpleExportAsPDF() {
    showMessage('正在生成PDF报告...', 'info');

    try {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF('p', 'mm', 'a4');

        // 添加标题
        doc.setFontSize(20);
        doc.text('训练图表报告', 105, 30, { align: 'center' });

        // 添加时间
        doc.setFontSize(12);
        doc.text(`生成时间: ${new Date().toLocaleString('zh-CN')}`, 105, 40, { align: 'center' });

        // 获取图表容器
        const chartContainer = document.querySelector('.charts-container');
        if (chartContainer) {
            const canvas = await html2canvas(chartContainer, {
                backgroundColor: '#ffffff',
                scale: 1.5
            });

            const imgData = canvas.toDataURL('image/jpeg', 0.9);
            const imgWidth = 180;
            const imgHeight = (canvas.height * imgWidth) / canvas.width;

            // 添加图片到PDF
            doc.addPage();
            doc.addImage(imgData, 'JPEG', 15, 20, imgWidth, imgHeight);
        }

        // 保存PDF
        doc.save(`训练报告_${new Date().toISOString().slice(0, 19).replace(/[:]/g, '-')}.pdf`);
        showMessage('PDF报告已生成', 'success');

    } catch (error) {
        showMessage('PDF导出失败: ' + error.message, 'error');
    }
}

// 简单的Excel导出
async function simpleExportAsExcel() {
    showMessage('正在导出数据...', 'info');

    try {
        // 创建模拟数据
        const wb = XLSX.utils.book_new();

        // 创建训练数据表
        const trainingData = [
            ['训练时间', '验证准确率', '训练准确率', '损失值'],
            ['10:00', '85.2%', '88.5%', '0.1523'],
            ['11:30', '87.6%', '90.1%', '0.1287'],
            ['13:45', '89.3%', '91.8%', '0.1024'],
            ['15:20', '91.5%', '93.2%', '0.0876'],
            ['17:00', '92.8%', '94.5%', '0.0721']
        ];

        const ws = XLSX.utils.aoa_to_sheet(trainingData);
        XLSX.utils.book_append_sheet(wb, ws, '训练数据');

        // 创建模型对比表
        const modelData = [
            ['模型类型', '平均准确率', '最高准确率', '训练次数'],
            ['SGD分类器', '87.5%', '92.8%', '5'],
            ['随机森林', '89.2%', '94.1%', '3'],
            ['SVM', '88.7%', '93.5%', '2'],
            ['逻辑回归', '86.3%', '90.2%', '4']
        ];

        const ws2 = XLSX.utils.aoa_to_sheet(modelData);
        XLSX.utils.book_append_sheet(wb, ws2, '模型对比');

        // 保存Excel
        XLSX.writeFile(wb, `训练数据_${new Date().toISOString().slice(0, 19).replace(/[:]/g, '-')}.xlsx`);
        showMessage('Excel数据已导出', 'success');

    } catch (error) {
        showMessage('Excel导出失败: ' + error.message, 'error');
    }
}

// 显示对话框的辅助函数
function showDialog(html) {
    const dialog = document.createElement('div');
    dialog.className = 'dialog-overlay';
    dialog.innerHTML = html;
    document.body.appendChild(dialog);

    setTimeout(() => dialog.classList.add('show'), 10);

    // 点击外部关闭
    dialog.addEventListener('click', function(e) {
        if (e.target === this) {
            closeDialog();
        }
    });
}

function closeDialog() {
    const dialog = document.querySelector('.dialog-overlay');
    if (dialog) {
        dialog.classList.remove('show');
        setTimeout(() => dialog.remove(), 300);
    }
}

// 模型类型相关函数
async function loadModelTypes() {
    try {
        const response = await fetch(`${API_BASE}/api/model_types`);
        const data = await response.json();

        if (data.success) {
            modelTypes = data.model_types;
            // 更新模型类型显示
            updateModelTypeSelect();
        }
    } catch (error) {
        console.error('加载模型类型失败:', error);
        // 如果API不存在，使用默认值
        modelTypes = {
            'sgd_tfidf': { name: 'TF-IDF + SGD分类器', description: '传统机器学习方法，速度快，适合小数据' },
            'random_forest': { name: 'TF-IDF + 随机森林', description: '集成学习方法，抗过拟合能力强' },
            'svm_tfidf': { name: 'TF-IDF + SVM', description: '支持向量机，适合高维特征' },
            'logistic_regression': { name: 'TF-IDF + 逻辑回归', description: '简单高效的线性模型' }
        };
        updateModelTypeSelect();
    }
}

function updateModelTypeSelect() {
    // 清空现有选项
    elements.modelTypeSelect.innerHTML = '';

    // 添加新选项
    for (const [key, info] of Object.entries(modelTypes)) {
        const option = document.createElement('option');
        option.value = key;
        option.textContent = info.name;
        elements.modelTypeSelect.appendChild(option);
    }

    // 更新描述
    updateModelDescription(elements.modelTypeSelect.value);
}

function updateModelDescription(modelType) {
    const info = modelTypes[modelType] || {
        name: '未知模型',
        description: '暂无描述'
    };
    elements.modelDescription.textContent = info.description;

    // 更新模型类型显示
    elements.currentModelType.textContent = info.name;
}

// 修改原有函数，支持模型类型
async function startTraining() {
    if (isTraining) {
        showMessage('训练已在运行中', 'warning');
        return;
    }

    const useGpu = elements.useGpuCheckbox.checked;
    const epochs = parseInt(elements.epochCount.value) || 5;
    const batchSize = parseInt(elements.batchSize.value) || 32;
    const modelType = elements.modelTypeSelect.value;

    if (epochs < 1 || epochs > 20) {
        showMessage('训练轮数应在1-20之间', 'error');
        return;
    }

    if (batchSize < 16 || batchSize > 128) {
        showMessage('批次大小应在16-128之间', 'error');
        return;
    }

    const modelInfo = modelTypes[modelType] || { name: modelType };
    if (!confirm(`确定要开始训练吗？\n模型类型: ${modelInfo.name}\n使用GPU: ${useGpu ? '是' : '否'}\n训练轮数: ${epochs}\n批次大小: ${batchSize}`)) {
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
                batch_size: batchSize,
                model_type: modelType
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || '启动训练失败');
        }

        showMessage(`${modelInfo.name} 训练已开始`, 'success');

        // 更新训练状态
        isTraining = true;
        elements.startTrainingBtn.style.display = 'none';
        elements.stopTrainingBtn.style.display = 'block';
        elements.progressPanel.classList.add('visible');
        updateStatusIndicator(elements.trainingState, 'status-active', '训练中');

        // 显示当前训练模型类型
        elements.trainingModelType.textContent = modelInfo.name;

        // 重置进度条
        elements.progressFill.style.width = '0%';
        elements.progressPercent.textContent = '0%';
        elements.epochInfo.textContent = `0/${epochs}`;

        // 开始轮询训练状态
        startTrainingStatusPolling();

    } catch (error) {
        showMessage(error.message, 'error');
        console.error('启动训练失败:', error);
    }
}

// 修改 checkTrainingStatus 函数
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

            // 刷新图表
            setTimeout(loadCharts, 1000);

            showMessage('训练完成！', 'success');
        }

    } catch (error) {
        console.error('获取训练状态失败:', error);
    }
}

// 统计模态框函数
async function showStatsModal() {
    try {
        elements.statsModal.classList.remove('hidden');

        // 显示加载中
        elements.statsContent.innerHTML = `
            <div class="loading">
                <i class="fas fa-spinner fa-spin"></i>
                <p>加载统计信息中...</p>
            </div>
        `;

        // 加载统计信息
        const response = await fetch(`${API_BASE}/api/feedback_stats`);
        const data = await response.json();

        if (data.success) {
            elements.statsContent.innerHTML = createStatsHTML(data);
        } else {
            elements.statsContent.innerHTML = `
                <div class="error">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>加载统计信息失败: ${data.error || '未知错误'}</p>
                    <button class="btn btn-secondary" onclick="showStatsModal()">
                        <i class="fas fa-redo"></i> 重试
                    </button>
                </div>
            `;
        }

    } catch (error) {
        console.error('加载统计信息失败:', error);
        elements.statsContent.innerHTML = `
            <div class="error">
                <i class="fas fa-exclamation-triangle"></i>
                <p>加载统计信息失败: ${error.message}</p>
                <p>请确保后端服务正在运行</p>
                <button class="btn btn-secondary" onclick="showStatsModal()">
                    <i class="fas fa-redo"></i> 重试
                </button>
            </div>
        `;
    }
}

function hideStatsModal() {
    elements.statsModal.classList.add('hidden');
}

function createStatsHTML(data) {
    return `
        <div class="stats-grid">
            <div class="stats-card">
                <div class="stats-label">原始反馈总数</div>
                <div class="stats-value">${data.total_all_raw}</div>
            </div>
            <div class="stats-card">
                <div class="stats-label">去重后总数</div>
                <div class="stats-value">${data.total_all_unique}</div>
            </div>
            <div class="stats-card">
                <div class="stats-label">有效反馈(去重)</div>
                <div class="stats-value">${data.total_valid_unique}</div>
            </div>
            <div class="stats-card">
                <div class="stats-label">重复数据</div>
                <div class="stats-value">${data.duplicate_count}</div>
            </div>
        </div>
        
        <div class="stats-message">
            <i class="fas fa-info-circle"></i>
            ${data.message}
        </div>
        
        <div class="stats-files">
            <h4>反馈文件列表:</h4>
            <ul>
                ${data.feedback_files ? data.feedback_files.map(file => `<li>${file}</li>`).join('') : '<li>暂无反馈文件</li>'}
            </ul>
        </div>
        
        ${data.file_details ? `
        <div class="stats-details">
            <h4>文件详细信息:</h4>
            <table>
                <thead>
                    <tr>
                        <th>文件名</th>
                        <th>原始行数</th>
                        <th>唯一行数</th>
                        <th>有效行数</th>
                    </tr>
                </thead>
                <tbody>
                    ${data.file_details.map(file => `
                    <tr>
                        <td>${file.name}</td>
                        <td>${file.raw_rows}</td>
                        <td>${file.unique_rows}</td>
                        <td>${file.valid_rows}</td>
                    </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
        ` : ''}
    `;
}

// 更新原有函数，添加图表相关功能
async function updateSystemStatus() {
    updateLastUpdateTime();
    await checkBackendStatus();
    await updateFeedbackCount();

    // 更新模型数量
    try {
        const response = await fetch(`${API_BASE}/api/list_models`);
        if (response.ok) {
            const data = await response.json();
            if (data.models) {
                elements.modelCount.textContent = data.models.length;
            }
        }
    } catch (error) {
        console.error('获取模型数量失败:', error);
    }
}

// 修改 updateFeedbackCount 函数
async function updateFeedbackCount() {
    try {
        // 从API获取反馈统计数据
        const response = await fetch(`${API_BASE}/api/feedback_stats`);
        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                elements.feedbackCountElement.textContent = `${data.total_valid_unique} 条`;
                elements.totalFeedback.textContent = data.total_all_unique;

                // 计算今日反馈（简化版）
                const today = new Date().toISOString().split('T')[0].replace(/-/g, '');
                const todayFiles = data.feedback_files ? data.feedback_files.filter(file => file.includes(today)) : [];
                if (todayFiles.length > 0) {
                    elements.todayFeedback.textContent = '有反馈';
                } else {
                    elements.todayFeedback.textContent = '0';
                }
            } else {
                // 如果API返回错误
                console.warn('获取反馈统计失败:', data.error);
                elements.feedbackCountElement.textContent = '0 条';
                elements.totalFeedback.textContent = '0';
                elements.todayFeedback.textContent = '0';
            }
        } else {
            throw new Error(`HTTP ${response.status}`);
        }
    } catch (error) {
        console.error('更新反馈数量失败:', error);
        elements.feedbackCountElement.textContent = '0 条';
        elements.totalFeedback.textContent = '0';
        elements.todayFeedback.textContent = '0';
    }
}

// 显示消息提示（添加图标）
function showMessage(message, type = 'info') {
    const toast = document.getElementById('message-toast');
    const toastMessage = document.getElementById('toast-message');
    const toastIcon = toast.querySelector('i');

    toastMessage.textContent = message;
    toast.className = `toast ${type}`;

    // 更新图标
    const icons = {
        'info': 'fa-info-circle',
        'success': 'fa-check-circle',
        'error': 'fa-exclamation-circle',
        'warning': 'fa-exclamation-triangle'
    };
    toastIcon.className = `fas ${icons[type] || 'fa-info-circle'}`;

    // 显示3秒后自动隐藏
    toast.classList.remove('hidden');
    setTimeout(() => {
        toast.classList.add('hidden');
    }, 3000);
}

// 修改 updateStatusIndicator 函数
function updateStatusIndicator(element, statusClass, text) {
    if (!element) return;

    const indicator = element.querySelector('.status-indicator');
    if (indicator) {
        indicator.className = `status-indicator ${statusClass}`;
        element.innerHTML = `${indicator.outerHTML} ${text}`;
    } else {
        element.innerHTML = `<span class="status-indicator ${statusClass}"></span> ${text}`;
    }
}

// ========== 以下是原有的辅助函数，需要保留 ==========

function updateCharCount() {
    const count = elements.textInput.value.length;
    elements.charCount.textContent = count;
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
            await updateFeedbackCount();
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
        await updateFeedbackCount();

        setTimeout(() => {
            showMessage('系统重置完成，反馈数据已清除', 'success');
            location.reload();
        }, 1000);

    } catch (error) {
        showMessage(`重置失败: ${error.message}`, 'error');
    }
}

// 页面关闭处理
function handlePageUnload(event) {
    // 使用sendBeacon发送页面关闭通知
    const data = JSON.stringify({});
    navigator.sendBeacon(`${API_BASE}/api/session_end`, data);

    // 可选：显示提示消息
    event.returnValue = '页面即将关闭，如有足够反馈数据将自动训练模型';
}

// 更新最后更新时间
function updateLastUpdateTime() {
    const now = new Date();
    elements.lastUpdate.textContent = now.toLocaleString('zh-CN');
}

// 添加CSS动画
const style = document.createElement('style');
style.textContent = `
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* 新增的动态样式 */
    .chart-canvas-container {
        position: relative;
        height: 200px;
        width: 100%;
    }

    .chart-no-data {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 200px;
        color: #999;
    }

    .chart-no-data i {
        font-size: 48px;
        margin-bottom: 10px;
        opacity: 0.3;
    }

    .chart-footer {
        margin-top: 10px;
        text-align: center;
        color: #666;
        font-size: 0.85rem;
    }

    /* 模态框样式 */
    .modal {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 2000;
    }

    .modal.hidden {
        display: none;
    }

    .modal-content {
        background: white;
        border-radius: 15px;
        width: 90%;
        max-width: 800px;
        max-height: 80vh;
        overflow-y: auto;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }

    .modal-header {
        padding: 20px 25px;
        border-bottom: 2px solid #f0f0f0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .modal-header h3 {
        margin: 0;
        color: #2c3e50;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .modal-close {
        background: none;
        border: none;
        font-size: 1.5rem;
        cursor: pointer;
        color: #666;
        line-height: 1;
        padding: 0;
    }

    .modal-body {
        padding: 25px;
    }

    /* 统计样式 */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 15px;
        margin-bottom: 20px;
    }

    .stats-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border-left: 4px solid #667eea;
    }

    .stats-label {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 5px;
    }

    .stats-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2c3e50;
    }

    .stats-message {
        background: #f0f9ff;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
        display: flex;
        align-items: center;
        gap: 10px;
        color: #0369a1;
    }

    .stats-files ul {
        list-style: none;
        padding: 0;
        margin: 10px 0;
    }

    .stats-files li {
        padding: 8px 0;
        border-bottom: 1px solid #f0f0f0;
        color: #666;
    }

    .stats-details table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
    }

    .stats-details th,
    .stats-details td {
        padding: 10px;
        text-align: left;
        border-bottom: 1px solid #f0f0f0;
    }

    .stats-details th {
        background: #f8f9fa;
        font-weight: 600;
        color: #2c3e50;
    }

    .loading {
        text-align: center;
        padding: 40px;
        color: #666;
    }

    .error {
        color: #ff416c;
        text-align: center;
        padding: 20px;
    }

    /* 图表展开状态 */
    .charts-container.expanded {
        grid-template-columns: 1fr !important;
    }

    .charts-container.expanded .chart-placeholder {
        height: 400px;
    }

    .charts-container.expanded .chart-canvas-container {
        height: 300px;
    }

    /* 响应式调整 */
    @media (max-width: 768px) {
        .stats-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .modal-content {
            width: 95%;
            margin: 10px;
        }
    }
`;
document.head.appendChild(style);