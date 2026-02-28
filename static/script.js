const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const emotionText = document.getElementById("emotion");

// Access webcam
navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => {
    video.srcObject = stream;
})
.catch(err => {
    console.error("Camera access denied:", err);
});

// Wait until video loads before capturing
video.addEventListener("loadeddata", () => {
    setInterval(captureFrame, 1000);
});

function captureFrame() {
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL("image/jpeg");

    fetch("/analyze", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ image: imageData })
    })
    .then(response => response.json())
    .then(data => {
        if (data.dominant_emotion) {
            emotionText.innerText = data.dominant_emotion.toUpperCase();
        } else if (data.error) {
            emotionText.innerText = data.error;
        }
    })
    .catch(error => {
        console.error("Error:", error);
    });
}