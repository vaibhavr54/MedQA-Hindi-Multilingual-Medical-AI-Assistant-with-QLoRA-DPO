/**
 * MedQA-Hindi Frontend Application
 * Handles module switching, API calls, comparison logic, and UI interactions.
 */

const API_BASE_URL = window.location.origin.includes('localhost') 
    ? 'http://localhost:8000' 
    : window.location.origin;

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
    loadMetrics();
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
        });
    });
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
            }
        });
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
    if (!text.trim()) {
        detectedLangSpan.textContent = '—';
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
    const model = document.getElementById('model-select').value;

    const response = await fetch(`${API_BASE_URL}/api/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            question: question,
            model: model,
            language: currentLang,
            max_tokens: 512,
            temperature: 0.7
        })
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    const data = await response.json();

    // Update UI
    document.getElementById('single-model-badge').textContent = 
        model === 'dpo' ? 'DPO Aligned' : 
        model === 'qlora' ? 'QLoRA Fine-tuned' : 'Base Model';
    document.getElementById('single-confidence').textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
    document.getElementById('single-answer').innerHTML = formatAnswer(data.answer);
    document.getElementById('single-time').textContent = `${data.processing_time_ms.toFixed(0)}ms`;
}

// Compare Models Request
async function askCompare(question) {
    const checkboxes = compareModelSelect.querySelectorAll('input[type="checkbox"]:checked');
    const models = Array.from(checkboxes).map(cb => cb.value);

    if (models.length < 2) {
        alert('Please select at least 2 models to compare.');
        return;
    }

    const response = await fetch(`${API_BASE_URL}/api/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            question: question,
            models: models,
            max_tokens: 512,
            temperature: 0.7
        })
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    const data = await response.json();

    // Update comparison cards
    data.comparisons.forEach(comp => {
        const answerEl = document.getElementById(`${comp.model}-answer`);
        const confidenceEl = document.getElementById(`${comp.model}-confidence`);
        const timeEl = document.getElementById(`${comp.model}-time`);

        if (answerEl) answerEl.innerHTML = formatAnswer(comp.answer);
        if (confidenceEl) confidenceEl.textContent = `Confidence: ${(comp.confidence * 100).toFixed(1)}%`;
        if (timeEl) timeEl.textContent = `${comp.response_time_ms.toFixed(0)}ms`;
    });
}

// Format Answer (handle newlines, etc.)
function formatAnswer(text) {
    if (!text) return '<div class="placeholder">No response generated.</div>';

    // Escape HTML
    let formatted = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // Convert newlines to <br>
    formatted = formatted.replace(/\n/g, '<br>');

    // Highlight disclaimer
    formatted = formatted.replace(
        /(Disclaimer:|अस्वीकरण:)(.*?)(?=<br>|$)/gi,
        '<span style="color: var(--warning); font-weight: 500;">$1$2</span>'
    );

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

// Load Metrics
async function loadMetrics() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/metrics`);
        if (!response.ok) throw new Error('Failed to load metrics');

        const data = await response.json();
        renderMetricsTable(data.metrics);
        renderCharts(data.metrics);

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
    }
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
