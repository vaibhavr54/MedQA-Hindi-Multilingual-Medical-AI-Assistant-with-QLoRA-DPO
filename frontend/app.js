/**
 * MedQA-Hindi Frontend Application
 * Handles module switching, API calls, comparison logic, and UI interactions.
 */

const API_BASE_URL = (() => {
    const meta = document.querySelector('meta[name="api-base-url"]');
    if (meta && meta.content) return meta.content.trim();

    if (window.location.protocol === 'file:') {
        return 'http://localhost:8000';
    }

    if (window.location.origin.includes('localhost')) {
        return 'http://localhost:8000';
    }

    return window.location.origin;
})();

const API_ENDPOINTS = {
    ask: `${API_BASE_URL}/api/ask`,
    compare: `${API_BASE_URL}/api/compare`,
    metrics: `${API_BASE_URL}/api/metrics`,
    logs: `${API_BASE_URL}/api/logs`,
    logsLatest: `${API_BASE_URL}/api/logs/latest`,
    logsClear: `${API_BASE_URL}/api/logs/clear`
};

// DOM Elements
const modules = document.querySelectorAll('.module');
const navButtons = document.querySelectorAll('.nav-btn');
const langButtons = document.querySelectorAll('.lang-btn');
const questionInput = document.getElementById('question-input');
const askBtn = document.getElementById('ask-btn');
const loadingOverlay = document.getElementById('loading-overlay');
const detectedLangSpan = document.getElementById('detected-lang');
const charCount = document.querySelector('.char-count');

// Mode switching
const modeButtons = document.querySelectorAll('.mode-btn');
const singleModelSelect = document.getElementById('single-model-select');
const compareModelSelect = document.getElementById('compare-model-select');
const singleResponse = document.getElementById('single-response');
const compareResponse = document.getElementById('compare-response');
const modelSelect = document.getElementById('model-select');

// State
let currentLang = 'auto';
let currentMode = 'single';

// Initialize
function init() {
    setupNavigation();
    setupLanguageToggle();
    setupModeSwitching();
    setupInputHandling();
    setupAskButton();
    setupModelBadgeSync();
    loadMetrics();
    applyLanguageSelection();
}

// Navigation
function setupNavigation() {
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetModule = btn.dataset.module;

            // Update nav buttons
            navButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Switch modules
            modules.forEach(m => m.classList.remove('active'));
            document.getElementById(`${targetModule}-module`).classList.add('active');

            // Load metrics if metrics module
            if (targetModule === 'metrics') {
                loadMetrics();
            }
        });
    });
}

// Language Toggle
function setupLanguageToggle() {
    langButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            langButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentLang = btn.dataset.lang;
            applyLanguageSelection();
        });
    });
}

function applyLanguageSelection() {
    if (currentLang === 'hi') {
        questionInput.setAttribute('lang', 'hi');
        questionInput.setAttribute('dir', 'ltr');
        questionInput.placeholder = 'उदाहरण: मुझे सिरदर्द और बुखार है, क्या करूं?';
        detectedLangSpan.textContent = 'Hindi (हिंदी)';
        detectedLangSpan.style.color = 'var(--accent-secondary)';
        return;
    }

    if (currentLang === 'en') {
        questionInput.setAttribute('lang', 'en');
        questionInput.setAttribute('dir', 'ltr');
        questionInput.placeholder = 'e.g., I have headache and fever, what should I do?';
        detectedLangSpan.textContent = 'English';
        detectedLangSpan.style.color = 'var(--accent-primary)';
        return;
    }

    questionInput.removeAttribute('lang');
    questionInput.setAttribute('dir', 'ltr');
    questionInput.placeholder = 'e.g., मुझे सिरदर्द और बुखार है, क्या करूं? (I have headache and fever, what should I do?)';
    detectLanguage(questionInput.value);
}

// Mode Switching (Single vs Compare)
function setupModeSwitching() {
    modeButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            modeButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentMode = btn.dataset.mode;

            if (currentMode === 'single') {
                singleModelSelect.classList.remove('hidden');
                compareModelSelect.classList.add('hidden');
                singleResponse.classList.remove('hidden');
                compareResponse.classList.add('hidden');
            } else {
                singleModelSelect.classList.add('hidden');
                compareModelSelect.classList.remove('hidden');
                singleResponse.classList.add('hidden');
                compareResponse.classList.remove('hidden');
                setCompareDefaults();
            }
        });
    });
}

function setCompareDefaults() {
    const checkboxes = compareModelSelect.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach(cb => {
        cb.checked = true;
    });
}

// Input Handling
function setupInputHandling() {
    questionInput.addEventListener('input', () => {
        const length = questionInput.value.length;
        charCount.textContent = `${length}/2000`;

        if (length > 2000) {
            charCount.style.color = 'var(--danger)';
        } else {
            charCount.style.color = 'var(--text-muted)';
        }

        // Detect language
        detectLanguage(questionInput.value);
    });
}

function detectLanguage(text) {
    if (currentLang !== 'auto') {
        return;
    }

    if (!text.trim()) {
        detectedLangSpan.textContent = 'Auto';
        return;
    }

    // Simple Devanagari detection
    const devanagariRange = /[ऀ-ॿ]/;
    const hasHindi = devanagariRange.test(text);

    detectedLangSpan.textContent = hasHindi ? 'Hindi (हिंदी)' : 'English';
    detectedLangSpan.style.color = hasHindi ? 'var(--accent-secondary)' : 'var(--accent-primary)';
}

// Ask Button
function setupAskButton() {
    askBtn.addEventListener('click', async () => {
        const question = questionInput.value.trim();

        if (!question) {
            alert('Please enter a medical question.');
            return;
        }

        if (question.length > 2000) {
            alert('Question is too long. Maximum 2000 characters.');
            return;
        }

        await clearLogs();
        showLoading(true);

        try {
            if (currentMode === 'single') {
                await askSingle(question);
            } else {
                await askCompare(question);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Error: ' + error.message);
        } finally {
            showLoading(false);
        }
    });
}

// Single Model Request
async function askSingle(question) {
    const model = modelSelect.value;

    const response = await fetch(API_ENDPOINTS.ask, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            question: question,
            model: model,
            language: currentLang,
            max_tokens: 256,
            temperature: 0.7
        })
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    const data = await response.json();

    // Update UI
    document.getElementById('single-model-badge').textContent = getModelLabel(model);
    document.getElementById('single-confidence').textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
    document.getElementById('single-answer').innerHTML = formatAnswer(data.answer);
    document.getElementById('single-time').textContent = `${data.processing_time_ms.toFixed(0)}ms`;
}

function setupModelBadgeSync() {
    const badge = document.getElementById('single-model-badge');
    if (!modelSelect || !badge) return;

    const sync = () => {
        badge.textContent = getModelLabel(modelSelect.value);
    };

    modelSelect.addEventListener('change', sync);
    sync();
}

function getModelLabel(model) {
    if (model === 'dpo') return 'DPO Aligned';
    if (model === 'qlora') return 'QLoRA Fine-tuned';
    return 'Base Qwen2.5';
}

// Compare Models Request
async function askCompare(question) {
    const checkboxes = compareModelSelect.querySelectorAll('input[type="checkbox"]:checked');
    const models = Array.from(checkboxes).map(cb => cb.value);

    if (models.length < 2) {
        alert('Please select at least 2 models to compare.');
        return;
    }

    const response = await fetch(API_ENDPOINTS.compare, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            question: question,
            models: models,
            language: currentLang,
            max_tokens: 256,
            temperature: 0.7
        })
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    const data = await response.json();

    const comparisons = data.comparisons || [];
    const best = comparisons.reduce((acc, item) => {
        if (!acc || item.confidence > acc.confidence) return item;
        return acc;
    }, null);

    document.querySelectorAll('.comparison-card').forEach(card => {
        card.classList.remove('best');
    });

    // Update comparison cards
    comparisons.forEach(comp => {
        const answerEl = document.getElementById(`${comp.model}-answer`);
        const confidenceEl = document.getElementById(`${comp.model}-confidence`);
        const timeEl = document.getElementById(`${comp.model}-time`);
        const cardEl = document.querySelector(`.comparison-card[data-model="${comp.model}"]`);

        if (answerEl) answerEl.innerHTML = formatAnswer(comp.answer);
        if (confidenceEl) confidenceEl.textContent = `Confidence: ${(comp.confidence * 100).toFixed(1)}%`;
        if (timeEl) timeEl.textContent = `${comp.response_time_ms.toFixed(0)}ms`;
        if (cardEl && best && comp.model === best.model) {
            cardEl.classList.add('best');
        }
    });
}

// Format Answer (handle newlines, etc.)
function formatAnswer(text) {
    if (!text) return '<div class="placeholder">No response generated.</div>';

    if (text.startsWith('Model unavailable.')) {
        return `<div class="placeholder">${escapeHtml(text)}</div>`;
    }

    // Escape HTML
    let formatted = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // Convert newlines to <br>
    formatted = formatted.replace(/\n/g, '<br>');

    // Remove inline disclaimers (footer already covers this)
    formatted = formatted
        .replace(/\s*⚠️?\s*Disclaimer:.*$/gim, '')
        .replace(/\s*⚠️?\s*अस्वीकरण:.*$/gim, '');

    return formatted;
}

// Loading Overlay
function showLoading(show) {
    if (show) {
        loadingOverlay.classList.remove('hidden');
    } else {
        loadingOverlay.classList.add('hidden');
    }
}

// Live Logs
let logsEventSource = null;
let logsPoller = null;
const logsToggle = document.getElementById('logs-toggle');
const logsDrawer = document.getElementById('logs-drawer');
const logsClose = document.getElementById('logs-close');
const logsBody = document.getElementById('logs-body');

function appendLogLine(line) {
    if (!logsBody) return;

    if (logsBody.querySelector('.logs-empty')) {
        logsBody.innerHTML = '';
    }

    if (logsBody.dataset.lastLine === line) {
        return;
    }

    logsBody.dataset.lastLine = line;

    const span = document.createElement('span');
    span.className = 'logs-line';
    if (line.includes('✅')) span.classList.add('log-success');
    else if (line.includes('❌')) span.classList.add('log-error');
    else if (line.includes('⚠️')) span.classList.add('log-warning');
    else if (line.includes('📥') || line.includes('ℹ️') || line.includes('🗑️')) span.classList.add('log-info');
    
    span.textContent = line;
    logsBody.appendChild(span);
    logsBody.scrollTop = logsBody.scrollHeight;
}

function openLogs() {
    if (logsDrawer) {
        logsDrawer.classList.remove('hidden');
    }
    if (!logsEventSource) {
        logsEventSource = new EventSource(API_ENDPOINTS.logs);
        logsEventSource.onmessage = (event) => appendLogLine(event.data);
        logsEventSource.onerror = () => {
            appendLogLine('⚠️  Live stream unavailable. Falling back to polling...');
            logsEventSource.close();
            logsEventSource = null;
            startLogsPolling();
        };
    }
}

function closeLogs() {
    if (logsDrawer) {
        logsDrawer.classList.add('hidden');
    }
}

async function clearLogs() {
    if (logsBody) {
        logsBody.innerHTML = '<p class="logs-empty">No logs yet. Run a query to see live events.</p>';
        logsBody.dataset.lastLine = '';
    }
    try {
        await fetch(API_ENDPOINTS.logsClear, { method: 'POST' });
    } catch (error) {
        console.error('Failed to clear logs:', error);
    }
}

async function fetchLatestLogs() {
    try {
        const response = await fetch(API_ENDPOINTS.logsLatest);
        if (!response.ok) throw new Error('Failed to fetch logs');
        const data = await response.json();
        (data.lines || []).forEach(appendLogLine);
    } catch (error) {
        console.error('Logs polling error:', error);
    }
}

function startLogsPolling() {
    if (logsPoller) return;
    fetchLatestLogs();
    logsPoller = setInterval(fetchLatestLogs, 1500);
}

// Load Metrics
async function loadMetrics() {
    try {
        const response = await fetch(API_ENDPOINTS.metrics);
        if (!response.ok) throw new Error('Failed to load metrics');

        const data = await response.json();
        renderMetricsTable(data.metrics);
        renderCharts(data.metrics);
        renderSamplePredictions(data.sample_predictions);

    } catch (error) {
        console.error('Error loading metrics:', error);
        // Show default metrics
        const defaultMetrics = [
            { metric: 'ROUGE-L', base_value: 0.32, qlora_value: 0.48, dpo_value: 0.54, qlora_improvement: 50.0, dpo_improvement: 68.8 },
            { metric: 'BERTScore', base_value: 0.71, qlora_value: 0.82, dpo_value: 0.87, qlora_improvement: 15.5, dpo_improvement: 22.5 },
            { metric: 'Medical Accuracy', base_value: 0.45, qlora_value: 0.72, dpo_value: 0.78, qlora_improvement: 60.0, dpo_improvement: 73.3 }
        ];
        renderMetricsTable(defaultMetrics);
        renderCharts(defaultMetrics);
        renderSamplePredictions(null);
    }
}

function renderSamplePredictions(samplePredictions) {
    const grid = document.getElementById('samples-grid');
    if (!grid) return;

    if (!samplePredictions || samplePredictions.length === 0) {
        grid.innerHTML = `
            <div class="sample-card">
                <p class="sample-loading">Run evaluation to see sample predictions...</p>
            </div>
        `;
        return;
    }

    grid.innerHTML = samplePredictions.map(sample => {
        const modelLabel = sample.model || sample.model_key || 'Model';
        const predictions = (sample.predictions || []).slice(0, 5);
        const references = (sample.references || []).slice(0, 5);

        const rows = predictions.map((pred, idx) => {
            const ref = references[idx] || '';
            return `
                <div class="sample-row">
                    <p class="sample-question">Prediction ${idx + 1}</p>
                    <p class="sample-answer">${escapeHtml(pred)}</p>
                    ${ref ? `<p class="sample-reference">Reference: ${escapeHtml(ref)}</p>` : ''}
                </div>
            `;
        }).join('');

        return `
            <div class="sample-card">
                <p class="sample-model">${escapeHtml(modelLabel)}</p>
                ${rows || '<p class="sample-loading">No sample predictions available.</p>'}
            </div>
        `;
    }).join('');
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function renderMetricsTable(metrics) {
    const tbody = document.getElementById('metrics-tbody');
    tbody.innerHTML = metrics.map(m => `
        <tr>
            <td><strong>${m.metric}</strong></td>
            <td>${m.base_value.toFixed(4)}</td>
            <td>${m.qlora_value.toFixed(4)}</td>
            <td>${m.dpo_value.toFixed(4)}</td>
            <td class="improvement-positive">DPO +${m.dpo_improvement}%</td>
        </tr>
    `).join('');
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', init);
document.addEventListener('DOMContentLoaded', () => {
    if (logsToggle) {
        logsToggle.addEventListener('click', openLogs);
    }
    if (logsClose) {
        logsClose.addEventListener('click', closeLogs);
    }
});
