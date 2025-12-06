# Clara Multimodal Assistant

Clara is an always-on **multimodal assistant** for the lab:

- **Wake word**: Porcupine (Picovoice)
- **Offline STT**: Vosk (EN & DE)
- **LLM / RAG**: `LabAssistRAG` (OpenAI + LangChain, FAISS index over PDFs)
- **TTS**: Piper (offline TTS, English & German voices)
- **Vision**: InsightFace + `FaceRecognitionLogger` for face recognition & enrollment
- **Architecture**: Multiple threaded workers (wake word, STT, LLM, TTS, vision), coordinated by `AssistantApp`.

The assistant is **always running**, but:
- Only “wakes up” after a **wake word**.
- Opens a **conversation session** for a limited time (`CONVERSATION_WINDOW_SEC`).
- STT + Vision only run while the session is active.
- Saying **“stop / goodbye / exit / quit”** ends the **current session only** and returns to wake-word listening.

---

## 1. Project Structure

LabBot_root/
│
├─ main_app.py                  # Orchestrator (AssistantApp, wake-word loop)
├─ config.py                    # Global constants, signals, RunningFlag, beep
├─ stt_worker.py                # STTWorker (Vosk via ClaraWakeApp)
├─ llm_worker.py                # LLMWorker (LabAssistRAG, language-aware)
├─ tts_worker.py                # TTSWorker + AudioPlaybackManager
├─ vision_worker.py             # VisionDynamicWorker (InsightFace + DB + enrollment)
│
├─ ClaraWakeApp.py              # Wake word + Vosk STT + Piper TTS helpers
├─ FaceRecognitionlogger.py     # Face DB management + recognition helper
├─ LLM_engine.py                # LabAssistRAG: OpenAI + LangChain RAG pipeline
├─ piper_tts.py                 # OfflineTTS wrapper around Piper
│
├─ settings_secrets_example.py  # Template for secrets (API keys, model paths)
├─ .env                         # environment-based secrets
├─ requirements.txt             # Python deps
│
├─ face_db/                     # Embeddings & meta.json for known faces
├─ recognition_logs/            # Vision recognition logs
├─ models/
│   ├─ vosk-model-small-en-us-0.15/
│   └─ vosk-model-small-de-0.15/
└─ clara_en_windows_v3_0_0/
    └─ clara_en_windows_v3_0_0.ppn  # Custom Picovoice wake-word model (if used)


## Features & Technology Stack
Multimodal Core: Seamlessly integrates Speech, Vision, and AI for natural interaction.

Offline Operation: Uses Vosk (STT) and Piper (TTS) for fast, privacy-preserving, local processing.

Contextual AI: LangChain/OpenAI RAG retrieves information from indexed PDF documents.

Concierge Mode: InsightFace provides real-time face recognition and enrollment management.

State Machine: Coordinated by the AssistantApp to enforce wake-word gating and session timeouts (CONVERSATION_WINDOW_SEC).

## Architecture Overview
Clara operates on a decoupled, threaded pipeline:
Idle: WakeWordWorker listens for "Clara" (using Porcupine).

Wake: WakeWordWorker starts the session, sets session_active_event.

Turn-Taking: The STT -> LLM -> TTS pipeline runs in a strict loop, cued by the READY_SIGNAL.

Vision: VisionDynamicWorker runs in parallel, sending structured reports to the LLM upon entry/exit events.

Sleep: The session ends after a verbal command or timeout, returning the assistant to idle mode.

## Requirements & Installation
This project is optimized for Raspberry Pi OS (64-bit).

Prerequisites:
Hardware: Raspberry Pi 4/5, Raspberry Pi Camera Module, Microphone Array (USB or HAT).

System: Must have ALSA/PulseAudio configured for sound device access and Picamera2 working.

System Tools: aplay (for fallback beep sound).

# Step-by-Step Setup:
Clone the Repository:


git clone https://github.com/your-username/LabBot_root.git
cd labassist

Set Up Virtual Environment:
python3 -m venv venv
source venv/bin/activate
Install Dependencies: 
pip install -r requirements.txt

# NOTE: PyAudio often requires system dependencies (e.g., sudo apt install portaudio19-dev)
Configure Secrets:

settings_secrets.py.

Create a .env file for your OpenAI API Key and other configuration parameters.

Download Models:

Place Vosk models (EN/DE) into the models/ directory.

Place your Piper voice models (and ONNX files) into a designated TTS model directory.

## Usage
To start the assistant:

# Must be run with the virtual environment activated
python3 main_app.py
The system will start the EmotionPlayer on the main thread and initialize all background workers.

Wait for the "bootup" screen and after 10 seconds
Say the wake word ("Clara").

The assistant will greet you and play a ready beep, signaling it is listening.
when its blinking it is listening.

## Team and Acknowledgments
This project was developed as a Industrial project for mechatronics at Technische Hochschule Würzburg-Schweinfurt.

Supervisor: Prof. Rainer Herrler., Mr Fabian schmidt

Development Team:

Moreen Fahim, System integrator
Gauruv , 3D printing Specialist
Alex, Vision Specialist
Bhuvan, STT_TTS Specialist
Varun, LLM Specialist

Credits: We acknowledge Picovoice (Porcupine), Kaldi/Vosk, Mycroft AI (Piper), and the InsightFace project for their foundational work.

## License and Credits

Credits: [Acknowledge Picovoice (Porcupine),Kaldi/Vosk, Mycroft AI (Piper), and InsightFace project]


## For emotions
[02:30, 12/6/2025] Bhuvan P: # 🧠 Emotion Image Dataset (Non-Commercial Use)

This repository contains categorized image folders representing different emotional states. The dataset is intended for *non-commercial educational and research purposes only*.

## 📁 Folder Structure

Each folder includes images corresponding to a specific emotion:

├── blink/
├── bootup/
├── dizzy/
├── happy/
├── neutral/
├── sad/
└── sleep/
[02:30, 12/6/2025] Bhuvan P: ## 📸 Source & License

All emotion images used in this project are sourced from the open-source project:

🔗 **[CodersCafeTech / Emo GitHub Repository](https://github.com/CodersCafeTech/Emo/tree/main/Code/emotions)**

These assets are licensed under:

📝 *Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)*  
[Read the Full License](https://creativecommons.org/licenses/by-nc/4.0/legalcode)
