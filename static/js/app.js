let mediaRecorder;
let audioChunks = [];

$(document).ready(function() {

    // ---------------- Text Chat ----------------
    $("#sendTextBtn").click(async function() {
        const query = $("#textQuery").val().trim();
        if (!query) return;

        $("#chatOutput").append(`You: ${query}\n`);

        const response = await fetch("/chat", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({query})
        });
        const data = await response.json();
        $("#chatOutput").append(`AI: ${data.answer}\n\n`);
        $("#textQuery").val("");
    });

    // ---------------- Voice Chat ----------------
    $("#startRecording").click(async function() {
        if (!navigator.mediaDevices) {
            alert("Microphone not supported!");
            return;
        }

        audioChunks = [];
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
        mediaRecorder.onstop = async function() {
            const blob = new Blob(audioChunks, { type: "audio/wav" });
            const formData = new FormData();
            formData.append("file", blob, "input.wav");

            const response = await fetch("/voice", { method: "POST", body: formData });
            const data = await response.json();

            $("#chatOutput").append(`You (voice): ${data.query}\n`);
            $("#chatOutput").append(`AI: ${data.answer}\n\n`);

            // Play TTS audio
            const ttsAudio = document.getElementById("ttsAudio");
            ttsAudio.src = data.tts_audio_path + "?cache=" + new Date().getTime();
            ttsAudio.style.display = "block";
            ttsAudio.play();
        };

        mediaRecorder.start();
        $("#startRecording").prop("disabled", true);
        $("#stopRecording").prop("disabled", false);
    });

    $("#stopRecording").click(function() {
        mediaRecorder.stop();
        $("#startRecording").prop("disabled", false);
        $("#stopRecording").prop("disabled", true);
    });

});
