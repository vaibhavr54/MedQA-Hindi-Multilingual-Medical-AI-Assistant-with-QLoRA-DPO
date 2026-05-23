/**
 * MedQA-Hindi Chart.js Visualizations
 * Renders metrics dashboard charts.
 */

let scoreChart = null;
let improvementChart = null;

function renderCharts(metrics) {
    renderScoreChart(metrics);
    renderImprovementChart(metrics);
}

function renderScoreChart(metrics) {
    const ctx = document.getElementById('score-chart');
    if (!ctx) return;

    // Destroy existing chart
    if (scoreChart) {
        scoreChart.destroy();
    }

    const labels = metrics.map(m => m.metric);
    const baseData = metrics.map(m => m.base_value);
    const qloraData = metrics.map(m => m.qlora_value);
    const dpoData = metrics.map(m => m.dpo_value);

    scoreChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Base Model',
                    data: baseData,
                    backgroundColor: 'rgba(148, 163, 184, 0.8)',
                    borderColor: 'rgba(148, 163, 184, 1)',
                    borderWidth: 1,
                    borderRadius: 6
                },
                {
                    label: 'QLoRA',
                    data: qloraData,
                    backgroundColor: 'rgba(94, 234, 212, 0.8)',
                    borderColor: 'rgba(94, 234, 212, 1)',
                    borderWidth: 1,
                    borderRadius: 6
                },
                {
                    label: 'DPO',
                    data: dpoData,
                    backgroundColor: 'rgba(56, 189, 248, 0.8)',
                    borderColor: 'rgba(56, 189, 248, 1)',
                    borderWidth: 1,
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#94a3b8',
                        font: { size: 12, family: 'Inter' },
                        padding: 16
                    }
                },
                tooltip: {
                    backgroundColor: '#1e293b',
                    titleColor: '#38bdf8',
                    bodyColor: '#e2e8f0',
                    borderColor: '#334155',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 8
                }
            },
            scales: {
                x: {
                    ticks: {
                        color: '#94a3b8',
                        font: { size: 11, family: 'Inter' }
                    },
                    grid: {
                        color: 'rgba(51, 65, 85, 0.3)'
                    }
                },
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    ticks: {
                        color: '#94a3b8',
                        font: { size: 11, family: 'Inter' },
                        callback: function(value) {
                            return value.toFixed(2);
                        }
                    },
                    grid: {
                        color: 'rgba(51, 65, 85, 0.3)'
                    }
                }
            }
        }
    });
}

function renderImprovementChart(metrics) {
    const ctx = document.getElementById('improvement-chart');
    if (!ctx) return;

    // Destroy existing chart
    if (improvementChart) {
        improvementChart.destroy();
    }

    const labels = metrics.map(m => m.metric);
    const qloraImprovement = metrics.map(m => m.qlora_improvement);
    const dpoImprovement = metrics.map(m => m.dpo_improvement);

    improvementChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'QLoRA Improvement %',
                    data: qloraImprovement,
                    backgroundColor: 'rgba(94, 234, 212, 0.8)',
                    borderColor: 'rgba(94, 234, 212, 1)',
                    borderWidth: 1,
                    borderRadius: 6
                },
                {
                    label: 'DPO Improvement %',
                    data: dpoImprovement,
                    backgroundColor: 'rgba(56, 189, 248, 0.8)',
                    borderColor: 'rgba(56, 189, 248, 1)',
                    borderWidth: 1,
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#94a3b8',
                        font: { size: 12, family: 'Inter' },
                        padding: 16
                    }
                },
                tooltip: {
                    backgroundColor: '#1e293b',
                    titleColor: '#38bdf8',
                    bodyColor: '#e2e8f0',
                    borderColor: '#334155',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 8,
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': +' + context.parsed.y.toFixed(1) + '%';
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: {
                        color: '#94a3b8',
                        font: { size: 11, family: 'Inter' }
                    },
                    grid: {
                        color: 'rgba(51, 65, 85, 0.3)'
                    }
                },
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: '#94a3b8',
                        font: { size: 11, family: 'Inter' },
                        callback: function(value) {
                            return '+' + value.toFixed(0) + '%';
                        }
                    },
                    grid: {
                        color: 'rgba(51, 65, 85, 0.3)'
                    }
                }
            }
        }
    });
}

// Export for app.js
window.renderCharts = renderCharts;
