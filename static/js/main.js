document.addEventListener('DOMContentLoaded', () => {
    const passwordInput = document.getElementById('password');
    const confirmPasswordInput = document.getElementById('confirm_password');
    const signupForm = document.getElementById('signupForm');
    const signupBtn = document.getElementById('signupBtn');
    const strengthFill = document.getElementById('strengthFill');
    const strengthText = document.getElementById('strengthText');
    const passwordMatchError = document.getElementById('passwordMatchError');

    if (passwordInput && strengthFill) {
        passwordInput.addEventListener('input', function() {
            const val = this.value;
            let strength = 0;

            if (val.length >= 8) strength += 25;
            if (val.match(/[A-Z]/)) strength += 25;
            if (val.match(/[0-9]/)) strength += 25;
            if (val.match(/[^a-zA-Z0-9]/)) strength += 25;

            strengthFill.style.width = strength + '%';

            if (strength === 0) {
                strengthFill.style.background = 'transparent';
                strengthText.textContent = 'Password Strength';
            } else if (strength <= 25) {
                strengthFill.style.background = '#dc2626';
                strengthText.textContent = 'Weak';
            } else if (strength <= 50) {
                strengthFill.style.background = '#eab308';
                strengthText.textContent = 'Fair';
            } else if (strength <= 75) {
                strengthFill.style.background = '#3b82f6';
                strengthText.textContent = 'Good';
            } else {
                strengthFill.style.background = '#16a34a';
                strengthText.textContent = 'Strong';
            }
            
            checkPasswordMatch();
        });
    }

    if (confirmPasswordInput) {
        confirmPasswordInput.addEventListener('input', checkPasswordMatch);
    }

    function checkPasswordMatch() {
        if (!passwordInput || !confirmPasswordInput) return;
        
        if (confirmPasswordInput.value === '') {
            passwordMatchError.style.display = 'none';
            if (signupBtn) signupBtn.disabled = false;
        } else if (passwordInput.value !== confirmPasswordInput.value) {
            passwordMatchError.style.display = 'block';
            if (signupBtn) signupBtn.disabled = true;
        } else {
            passwordMatchError.style.display = 'none';
            if (signupBtn) signupBtn.disabled = false;
        }
    }

    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('imageUpload');
    const imagePreview = document.getElementById('imagePreview');
    const previewContainer = document.getElementById('previewContainer');
    const uploadPrompt = document.getElementById('uploadPrompt');
    const predictionForm = document.getElementById('predictionForm');
    const predictBtn = document.getElementById('predictBtn');
    const btnText = document.getElementById('btnText');
    const btnLoader = document.getElementById('btnLoader');

    if (dropZone && fileInput) {
        dropZone.addEventListener('click', () => fileInput.click());

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.style.borderColor = 'var(--primary)';
            dropZone.style.background = 'rgba(25, 135, 84, 0.05)';
        });

        dropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dropZone.style.borderColor = 'rgba(0, 0, 0, 0.15)';
            dropZone.style.background = 'transparent';
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.style.borderColor = 'rgba(0, 0, 0, 0.15)';
            dropZone.style.background = 'transparent';
            
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                handleFileSelect(fileInput.files[0]);
            }
        });

        fileInput.addEventListener('change', function() {
            if (this.files.length) {
                handleFileSelect(this.files[0]);
            }
        });
    }

    function handleFileSelect(file) {
        if (!file.type.match('image.*')) {
            alert('Please select a valid image file (JPG, PNG).');
            return;
        }

        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            uploadPrompt.style.display = 'none';
            previewContainer.style.display = 'block';
            if (predictBtn) predictBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    if (predictionForm) {
        predictionForm.addEventListener('submit', function() {
            if (predictBtn && btnText && btnLoader) {
                predictBtn.disabled = true;
                btnText.textContent = 'Processing Image...';
                btnLoader.style.display = 'inline-block';
                dropZone.style.pointerEvents = 'none';
            }
        });
    }
});

function removeImage(event) {
    event.stopPropagation();
    const fileInput = document.getElementById('imageUpload');
    const imagePreview = document.getElementById('imagePreview');
    const previewContainer = document.getElementById('previewContainer');
    const uploadPrompt = document.getElementById('uploadPrompt');
    const predictBtn = document.getElementById('predictBtn');

    fileInput.value = '';
    imagePreview.src = '';
    previewContainer.style.display = 'none';
    uploadPrompt.style.display = 'block';
    
    if (predictBtn) {
        predictBtn.disabled = true;
    }
}