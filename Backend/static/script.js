document.getElementById('uploadForm').addEventListener('submit', function (event) {
    event.preventDefault();  // Prevents the form from reloading the page

    const videoFile = document.getElementById('videoFile').files[0];
    if (!videoFile) {
        alert('Please select a video file.');
        return;
    }

    const formData = new FormData();
    formData.append('file', videoFile);

    document.getElementById('submitBtn').disabled = true;
    document.getElementById('resultMessage').textContent = 'Processing...';

    // Asynchronous form submission via fetch
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('submitBtn').disabled = false;

        if (data.error) {
            document.getElementById('resultMessage').textContent = 'Error: ' + data.error;
        } else {
            const isDeepfake = data.deepfake ? 'Yes' : 'No';
            const probability = data.probability.toFixed(2);
            document.getElementById('resultMessage').textContent = `Deepfake: ${isDeepfake}, Probability: ${probability}`;
        }
    })
    .catch(error => {
        document.getElementById('submitBtn').disabled = false;
        document.getElementById('resultMessage').textContent = 'Error processing video.';
        console.error('Error:', error);
    });
});