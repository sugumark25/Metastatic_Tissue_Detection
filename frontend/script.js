const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadContent = document.getElementById('upload-content');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');
const removeBtn = document.getElementById('remove-btn');
const actionArea = document.getElementById('action-area');
const analyzeBtn = document.getElementById('analyze-btn');
const resultsCard = document.getElementById('results-card');
const resultBadge = document.getElementById('result-badge');
const probNormal = document.getElementById('prob-normal');
const probCancer = document.getElementById('prob-cancer');
const barNormal = document.getElementById('bar-normal');
const barCancer = document.getElementById('bar-cancer');

let currentFile = null;

// Drag and Drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetState();
});

const selectBtn = document.getElementById('select-btn');

dropZone.addEventListener('click', (e) => {
    if (e.target !== selectBtn) {
        if (!currentFile) fileInput.click();
    }
});

if (selectBtn) {
    selectBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file (PNG, JPG, TIF).');
        return;
    }

    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadContent.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        dropZone.style.borderStyle = 'solid';
        actionArea.classList.remove('hidden');
        resultsCard.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

function resetState() {
    currentFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    uploadContent.classList.remove('hidden');
    previewContainer.classList.add('hidden');
    actionArea.classList.add('hidden');
    resultsCard.classList.add('hidden');
    document.getElementById('explanation-card').classList.add('hidden');
    dropZone.style.borderStyle = 'dashed';
}

analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    resultsCard.classList.remove('hidden');
    resultBadge.textContent = 'Processing...';

    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<span>Analyzing...</span><i class="fa-solid fa-spinner fa-spin"></i>';

    try {
        const formData = new FormData();
        formData.append('file', currentFile);

        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errText = await response.text();
            throw new Error(`Status ${response.status}: ${errText}`);
        }

        const data = await response.json();
        console.log("Backend Response:", data);

        updateResults(data);

    } catch (error) {
        console.error("❌ ERROR:", error);
        resultBadge.textContent = 'Error';
        alert('Failed to analyze image. Check console for details.');
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<span>Analyze Sample</span><i class="fa-solid fa-microscope"></i>';
    }
});

function updateResults(data) {
    const pNormal = (data.probabilities.Normal * 100).toFixed(1);
    const pCancer = (data.probabilities.Cancer * 100).toFixed(1);

    probNormal.textContent = `${pNormal}%`;
    probCancer.textContent = `${pCancer}%`;

    barNormal.style.width = `${pNormal}%`;
    barCancer.style.width = `${pCancer}%`;

    if (data.prediction === 'Cancer') {
        resultBadge.textContent = '⚠️ METASTATIC TISSUE DETECTED';
        resultBadge.className = 'status-badge badge-cancer';
        resultsCard.style.borderTopColor = 'var(--danger)';

        document.getElementById('explanation-text').innerHTML = `
            <p>This tissue sample exhibits characteristics consistent with <strong>metastatic cancer cells</strong>. Further clinical evaluation and confirmation is recommended.</p>
            
            <ul class="insight-list">
                <li>
                    <strong>What is Metastasis?</strong>
                    Metastasis occurs when cancer cells break away from the original (primary) tumor, travel through the bloodstream or lymphatic system, and establish new tumors in distant organs or tissues.
                </li>
                <li>
                    <strong>Cellular Appearance</strong>
                    Metastatic cells typically display irregular morphology, enlarged nuclei, increased nuclear-to-cytoplasmic ratio, and dense cellular crowding compared to normal tissue.
                </li>
                <li>
                    <strong>Clinical Significance</strong>
                    Detection of metastatic spread is critical for staging and treatment planning. Immediate consultation with a pathologist and oncologist is advised.
                </li>
            </ul>

            <div class="medical-alert alert-cancer">
                <i class="fa-solid fa-triangle-exclamation"></i>
                <div class="medical-alert-text">
                    <strong>Clinical Note:</strong> This AI prediction is for research and screening purposes only. A qualified pathologist must provide official diagnosis and interpretation.
                </div>
            </div>
        `;
    } else {
        resultBadge.textContent = '✓ NORMAL TISSUE';
        resultBadge.className = 'status-badge badge-normal';
        resultsCard.style.borderTopColor = 'var(--success)';

        document.getElementById('explanation-text').innerHTML = `
            <p>This tissue sample appears consistent with <strong>normal, healthy biological tissue</strong> without evidence of malignancy.</p>
            
            <ul class="insight-list">
                <li>
                    <strong>Normal Cellular Characteristics</strong>
                    Healthy cells typically display uniform morphology, regular spatial arrangement, well-defined cell boundaries, and appropriate nuclear size relative to cytoplasm.
                </li>
                <li>
                    <strong>Tissue Architecture</strong>
                    Normal tissue maintains preserved structural integrity with organized cellular layers, normal vasculature, and absence of abnormal cellular invasion or uncontrolled growth patterns.
                </li>
                <li>
                    <strong>Assessment Outcome</strong>
                    No histopathological evidence of malignancy or metastatic disease detected in this sample based on deep learning analysis.
                </li>
            </ul>

            <div class="medical-alert alert-normal">
                <i class="fa-solid fa-circle-check"></i>
                <div class="medical-alert-text">
                    <strong>Reassurance Note:</strong> Normal findings support continued routine monitoring. However, clinical context and additional diagnostic tools should inform final diagnosis.
                </div>
            </div>
        `;
    }

    document.getElementById('explanation-card').classList.remove('hidden');
}

