---

## 🧾 Credits and Licensing

**Project:** CLARA – Cognitive Lab Assistant for Robotics & Automation  
**Institution:** Technische Hochschule Würzburg-Schweinfurt (THWS)  
**Developer:** Bhuvan Puttaswamy  
**Supervisor:** Prof. Hellerich  
**Team Members:** Aleksandr Markhakshinov, Gaurav Jogi, Moreen Fahim, Varun Girish Katti Project Group – Lab Assist Bot  

---

### 🧠 Technologies and Frameworks

| Component | Description | License |
|------------|--------------|----------|
| **Vosk** | Offline speech recognition engine for STT (Speech-to-Text) | Apache 2.0 |
| **Piper TTS** | Neural Text-to-Speech system for offline voice generation | MIT |
| **Voice Model: en_GB-cori-medium** | Natural British female voice for CLARA | CC-BY-4.0 |
| **Python 3.12** | Core programming language used for the project | PSF License |
| **ReSpeaker Mic Array** | Microphone array for multi-directional voice input | Open hardware |
| **Raspberry Pi 5** | Embedded hardware platform for deployment | Open hardware |

---

### 🔊 Voice Attribution (Required by CC-BY-4.0)

> Voice model **“Cori (Medium)”** by *Rhasspy Piper Voices*  
> Licensed under [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/).

---

### 🧩 Acknowledgements

- **AlphaCephei Vosk** for providing robust offline speech recognition.  
- **Rhasspy Piper** project for high-quality neural text-to-speech voices.  
- **THWS Faculty of Electrical Engineering / Mechatronics** for academic support.  
- **Lab Assist Bot Team** for collaboration in integrating speech and robotics.

---

### Wake Word Detection
This project uses the **Porcupine Wake Word Engine** by [Picovoice](https://picovoice.ai/),  
under a free educational-use license.  
Wake words such as *“Bumblebee”* or *“Grapefruit”* are part of Picovoice’s official keyword set.

---

**© 2025 CLARA – Lab Assist Bot Project, THWS**  
*Developed for academic project and intelligent automation in mechatronics systems.*

---

## Quick Start (Windows, PowerShell)
`powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.lock.txt
copy .env.example .env   # then edit values
python main.py           # your repo has main.py
`"
Add-Content -Encoding utf8 README.md "
Add-Content -Encoding utf8 README.md 
`powershell
pip install <pkg>
pip freeze > requirements.lock.txt


## Privacy Notice ##
Privacy Notice (Voice Demo)

This demo listens for a wake word (“hey clara”) and processes speech locally on this device to respond.

No audio is sent to external servers.

We do not keep recordings by default. If logging is enabled for debugging, short audio snippets/recognized text may be stored locally and deleted after the event.

By speaking to the demo, you consent to this processing.

Data Controller: [ Embodied Lab Assistant Project Team] — Contact: [your.email@…]

Hinweis zum Datenschutz (Sprach-Demo)

Diese Demo reagiert auf ein Aktivierungswort („hey clara“) und verarbeitet Sprache lokal auf diesem Gerät.

Es werden keine Audiodaten an externe Server gesendet.

Standardmäßig werden keine Aufnahmen gespeichert. Falls Protokolle für Debugging aktiv sind, können kurze Audioausschnitte/Erkennungstexte lokal gespeichert und nach der Veranstaltung gelöscht werden.

Mit dem Sprechen in die Demo stimmen Sie dieser Verarbeitung zu.

Verantwortliche Stelle: [Ihr Name / THWS-Projektteam] – Kontakt: [Ihre.E-Mail@…]
