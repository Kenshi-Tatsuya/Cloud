document.addEventListener('DOMContentLoaded', () => {

    // --- Page Elements ---
    const authPage = document.getElementById('auth-page');
    const homePage = document.getElementById('home-page');

    // --- Auth Elements ---
    const loginTab = document.getElementById('login-tab');
    const registerTab = document.getElementById('register-tab');
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');

    // --- Home Elements ---
    const logoutBtn = document.getElementById('logout-btn');
    const startDetectBtn = document.getElementById("start-detect-btn");
    const video = document.getElementById("camera-feed");
    const outputBox = document.getElementById("translation-output");


    // --------------------------
    // A. View Switching Logic
    // --------------------------
    function switchAuthForm(type) {
        if (type === 'login') {
            loginTab.classList.add('active-tab');
            registerTab.classList.remove('active-tab');
            registerForm.classList.remove('active-form');
            loginForm.classList.add('active-form');
        } else {
            registerTab.classList.add('active-tab');
            loginTab.classList.remove('active-tab');
            loginForm.classList.remove('active-form');
            registerForm.classList.add('active-form');
        }
    }

    loginTab.addEventListener('click', () => switchAuthForm('login'));
    registerTab.addEventListener('click', () => switchAuthForm('register'));


    // --------------------------
    // B. Page Navigation Logic
    // --------------------------
    function goToHome() {
        authPage.classList.remove('active');
        homePage.classList.add('active');

        // Reset output box
        outputBox.innerHTML = '<p>No signs detected yet...</p>';
    }

    function goToAuth() {
        homePage.classList.remove('active');
        authPage.classList.add('active');
        switchAuthForm('login');
        alert('You have been logged out.');
    }

    logoutBtn.addEventListener('click', goToAuth);


    // --------------------------
    // C. Auth Form Submission
    // --------------------------
    loginForm.addEventListener('submit', (e) => {
        e.preventDefault();
        alert("✅ Simulated Login Successful!");
        goToHome();
    });

    registerForm.addEventListener('submit', (e) => {
        e.preventDefault();

        const password = document.getElementById('register-password').value;
        const confirmPassword = document.getElementById('register-confirm-password').value;

        if (password !== confirmPassword) {
            alert("⚠️ Passwords do not match!");
            return;
        }

        alert("✅ Simulated Registration Successful!");
        goToHome();
    });


    // --------------------------
    // D. LIVE REAL-TIME DETECTION
    // --------------------------

    let streaming = false;

    startDetectBtn.addEventListener("click", async () => {

        if (streaming) return;

        streaming = true;
        startDetectBtn.textContent = "Detecting...";

        // Start Webcam
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;

        // Send frames every 300ms
        setInterval(() => {
            sendFrameToBackend();
        }, 300);
    });


    function sendFrameToBackend() {

        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0);

        const base64 = canvas.toDataURL("image/jpeg");

        fetch("http://localhost:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: base64 })
        })
        .then(res => res.json())
        .then(data => {
            if (data.text) {
                outputBox.innerHTML = `<p>${data.text}</p>`;
            }
        })
        .catch(err => {
            console.error("Prediction error:", err);
        });
    }

}); // DOMContentLoaded end
