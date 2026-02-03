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
        resultBadge.textContent = 'METATASTATIC TISSUE DETECTED';
        resultBadge.className = 'status-badge badge-cancer';
        resultsCard.style.borderTopColor = 'var(--danger)';

        document.getElementById('explanation-text').innerHTML = `
            <p><strong>Metastatic Cancer Detected:</strong> This sample shows characteristics consistent with metastatic tissue spread.</p>
            <ul style="margin-top:10px; padding-left:20px;">
                <li><strong>Spread Mechanism:</strong> Metastasis occurs when cancer cells break away from the original (primary) tumor, travel through the blood or lymph system, and form new tumors in other organs or tissues.</li>
                <li><strong>Appearance:</strong> These cells often appear irregular in shape, have larger nuclei, and are densely packed compared to normal tissue.</li>
            </ul>
        `;
    } else {
        resultBadge.textContent = 'NORMAL TISSUE';
        resultBadge.className = 'status-badge badge-normal';
        resultsCard.style.borderTopColor = 'var(--success)';

        document.getElementById('explanation-text').innerHTML = `
            <p><strong>Normal Tissue Detected:</strong> This sample appears consistent with healthy biological tissue.</p>
            <ul style="margin-top:10px; padding-left:20px;">
                <li><strong>Characteristics:</strong> Normal cells typically have uniform shapes, regular arrangement, and distinct boundaries.</li>
                <li><strong>Structure:</strong> The tissue architecture is preserved without signs of uncontrolled growth or invasion.</li>
            </ul>
        `;
    }

    document.getElementById('explanation-card').classList.remove('hidden');
}

