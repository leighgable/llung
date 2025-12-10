const video = document.getElementById('camera');
const captureBtn = document.getElementById('capture-btn');
const photoInput = document.getElementById('photo-input');

async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
    video.srcObject = stream;
  } catch (err) {
    console.error('Camera access denied:', err);
  }
}

captureBtn.addEventListener('click', () => {
  // On mobile Safari, input.capture opens camera
  photoInput.click();
});

photoInput.addEventListener('change', () => {
  document.getElementById('photo-form').requestSubmit();
});

initCamera();
