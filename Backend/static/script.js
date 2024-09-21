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




function toggleMenu() {
    const navLinks = document.getElementById('nav-links');
    navLinks.classList.toggle('active'); // Toggle the 'active' class to show/hide
}



document.getElementById('tryNowButton').addEventListener('click', function() {
        window.location.href = '#uploadForm'
    });

// feedback form submission 
document.addEventListener("DOMContentLoaded", function() {
    const feedbackForm = document.getElementById("feedbackForm");

    if (feedbackForm) {
        feedbackForm.addEventListener("submit", function(event) {
            event.preventDefault();

            // Capture form input values
            let name = document.getElementById("name").value;
            let email = document.getElementById("email").value;
            let message = document.getElementById("message").value;

            // AJAX request to Flask backend
            let xhr = new XMLHttpRequest();
            xhr.open("POST", "/feedback", true);
            xxhr.setRequestHeader("Content-Type", "application/json");
            xhr.onreadystatechange = function () {
                if (xhr.readyState === 4) {
                    let responseMessage = "";
                    if (xhr.status === 200) {
                        responseMessage = "Thank you for your feedback!";
                        feedbackForm.reset();
                    } else {
                        responseMessage = "An error occurred. Please try again.";
                    }
                    document.getElementById("formFeedback").innerText = responseMessage;
                }
            };

            let data = JSON.stringify({
                "name": name,
                "email": email,
                "message": message
            });

            xhr.send(data);
        });
    } else {
        console.error("Feedback form not found");
    }
});




// function showFlashMessage() {
//     const flashMessage = '{{ get_flashed_messages()[0] }}'; // Get the flash message
//     if (flashMessage) {
//         alert(flashMessage); // Display it (can be customized to display in a better way)
//     }
// }