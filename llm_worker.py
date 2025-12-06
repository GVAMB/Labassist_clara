""""
llm_worker.py

Contains the LLMWorker class.
Uses LabAssistRAG to answer questions, manage enrollment, and:
- respects Clara's current language (EN/DE)
- formats Vision System reports nicely
- 'stop/goodbye/exit/quit' ends ONLY the current session (back to wake word).
"""

import time
import threading
from queue import Queue, Empty   
from emotion_animator import set_emotion_from_outside
from config import (
    POISON_PILL,
    READY_SIGNAL,
    ENROLL_SIGNAL,
    RunningFlag,
)
from LLM.LLM_engine import LabAssistRAG
from STT_TTS.ClaraWakeApp import ClaraWakeApp


class LLMWorker(threading.Thread if False else object):
    # little trick to keep type-hints ok in some editors; we'll override below
    pass


class LLMWorker(threading.Thread):
    """
    Uses LabAssistRAG as the LLM engine.
    - Handles enroll trigger
    - Handles stop/exit
    - Streams RAG answer to TTS in small chunks
    - If Clara language == 'de', instructs LLM to answer in German.
    - On 'stop/goodbye/exit/quit': ends the current session only.
    """

    def __init__(
        self,
        rag_engine: LabAssistRAG,
        input_queue: Queue,
        output_queue: Queue,
        running_flag: RunningFlag,
        enroll_q: Queue,
        is_enrolling_event: threading.Event,
        clara: ClaraWakeApp,
        session_active_event: threading.Event,
        stt_listening_event: threading.Event,   
    ):
        super().__init__(daemon=True)
        self.rag = rag_engine
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running_flag = running_flag
        self.enroll_q = enroll_q
        self.is_enrolling_event = is_enrolling_event
        self.clara = clara
        self.session_active_event = session_active_event
        self.stt_listening_event = stt_listening_event 

        # Track previous listening state so we only send blink on rising edge
        self._was_listening = False

        # Known face/emotion bases from your EmotionPlayer (safe default set)
        self._face_bases = {
            "neutral",
            "happy",
            "angry",
            "sad",
            "sleep",
            "sleepy",
            "dizzy",
            "excited",
            "blink",
            "bootup",
            "crying",
            "thinking",
        }

    def _make_speech(self, text: str, emotion: str = "neutral"):
        """Wrap TTS text with an emotion tag for TTS + eyes."""
        return {
            "type": "speech",
            "text": text,
            "emotion": emotion,
        }

    def _decide_emotion(self, user_query: str, is_vision_report: bool) -> str:
        """
        Very simple emotion routing based on what kind of message this is.
        You can tweak this anytime.
        """
        t = (user_query or "").lower()

        if is_vision_report:
            # Someone entered / presence update -> cheerful greeting
            return "happy"

        # User asked something confusing / you couldn't answer last time
        if any(kw in t for kw in ["i cant help you","i don't know", "you don't know", "error", "problem", "issue"]):
            return "dizzy" if "dizzy" in self._face_bases else "neutral"

        # Greetings / nice interactions -> happy
        if any(
            kw in t
            for kw in [
                "hello",
                "hi",
                "hey",
                "good morning",
                "good afternoon",
                "good evening",
                "thank you",
                "thanks",
            ]
        ):
            return "happy"

        # Very negative / sad content -> sad
        if any(kw in t for kw in ["sad", "upset", "angry", "depressed"]):
            return "sad"

        # Goodbye / ending interaction: run a special crying -> sleepy sequence
        if any(kw in t for kw in ["goodbye", "bye", "stop", "see you", "exit"]):
            # Trigger the sequence in a background thread so we don't block the main logic
            threading.Thread(
                target=self._goodbye_emotion_sequence,
                daemon=True,
            ).start()

            # Immediate "primary" emotion for this message (e.g. for logging or fallback)
            return "crying" if "crying" in self._face_bases else "sad"

        # Default for normal questions = thinking
        return "neutral"

    def _goodbye_emotion_sequence(self) -> None:
        """
        Play a short goodbye sequence:
        crying for 3 seconds, then sleepy.
        Runs in a background thread.
        """
        # Show crying
        self.set_emotion("crying" if "crying" in self._face_bases else "sad")
        time.sleep(3)  # adjust as needed
        # Then sleepy
        self.set_emotion("sleepy" if "sleepy" in self._face_bases else "sleep")

    def set_emotion(self, emotion: str) -> None:
        """
        Set the robot's emotion on the face display by sending it to the EmotionPlayer.
        This uses the global queue from emotion_animator (Option 1 design).
        """
        emotion = (emotion or "").strip().lower()
        if not emotion:
            return

        print(f"[EmotionRouter] Setting emotion to: {emotion}")

        try:
            # This enqueues the emotion; the main-thread EmotionPlayer
            # picks it up in run_robot_face().
            set_emotion_from_outside(emotion)
        except Exception as e:
            print(f"[EmotionRouter] Failed to set emotion '{emotion}': {e}")

    def run(self):
        print("🧠 LLM Worker (RAG) started.")
        punctuation = [".", "?", "!", "\n", ")", "("]
        MIN_TTS_WORD_CHUNK = 4  # streaming chunk size in words

        while self.running_flag():
            # If enrollment dialog is active, wait until it's done
            while self.is_enrolling_event.is_set() and self.running_flag():
                time.sleep(0.1)
            try:
                user_text = self.input_queue.get(timeout=0.1)
            except Empty:
                # While we're idle, check if STT started listening
                if self.stt_listening_event is not None:
                    is_listening = self.stt_listening_event.is_set()
                    # Rising edge: just went from not listening -> listening
                    if is_listening and not self._was_listening:
                        # when its listening flag i want the emotions to go to blink
                        if "blink" in self._face_bases:
                            self.set_emotion("blink")
                    self._was_listening = is_listening
                continue

            try:
                if user_text is POISON_PILL:
                    self.output_queue.put(POISON_PILL)
                    self.input_queue.task_done()
                    break

                # Normalize the raw text
                raw_user_query = str(user_text)
                user_query = raw_user_query

                if user_query.startswith("Initial Vision Report:"):
                    user_query = user_query[len("Initial Vision Report:") :].strip()

                # --- Enrollment and Stop Logic ---
                if "enroll" in user_query.lower() or "register" in user_query.lower():
                    self.output_queue.put(
                        self._make_speech("Sure! To enroll, I need a few details.", "happy")
                    )
                    time.sleep(3)
                    self.is_enrolling_event.set()
                    self.enroll_q.put(ENROLL_SIGNAL)
                    self.input_queue.task_done()
                    continue

                if any(w in user_query.lower() for w in ["stop", "goodbye", "exit", "quit"]):
                    # End current session only; robot stays alive and returns to wake word.
                    self.output_queue.put(self._make_speech("Goodbye.", "sad"))
                    self.session_active_event.clear()
                    # Do NOT emit READY_SIGNAL (no more turns in this session)
                    self.input_queue.task_done()
                    continue

                # --- Prompt Construction (language-aware) ---

                cur_lang = getattr(self.clara, "current_lang", "en")
                is_vision_report = user_query.strip().startswith("Vision System Report:")
                emotion_for_answer = self._decide_emotion(user_query, is_vision_report)

                if is_vision_report:
                    # Vision report mode
                    if cur_lang == "en":
                        style_prompt = (
                            "You have received a Vision System Report. Respond in a friendly, excited, concierge-like tone. "
                            "When greeting known individuals, use their full title and last name "
                            "(e.g., 'Professor Smith' if role is professor). "
                            "Your response must be a single, combined greeting followed immediately by your offer of assistance. "
                            "The number following 'I see' is the TOTAL number of people present. "
                            "The count following 'Unidentified people' represents individuals not in your database. "
                            "If the report contains the tag 'INITIAL_COUNT', this is a session status update, NOT a change event. "
                            "Acknowledge the total number of people present and greet the listed known individuals. "
                            "For the initial report, treat the listed known faces as the identified person(s) and call them "
                            "by their name and include the role. "
                            "Greet the person(s) who entered, using the information provided. "
                            "Acknowledge and greet any unknown person(s) present in a nice way. "
                            "When you mention unknown people, NEVER use phrasing like "
                            "'including X unidentified individuals', because it can sound like the total is larger. "
                            "Instead, use clear sentences such as "
                            "'There are 2 people present. 2 of them are unidentified.' or "
                            "'There are N people in the room. X of them are unknown to me.' "
                            "Do not use the exact phrase 'Vision System Report'. "
                            "Respond only in English and use short sentences."
                        )
                    else:
                        style_prompt = (
                            "Sie haben einen visuellen Systembericht erhalten. Antworten Sie in einem freundlichen, "
                            "begeisterten und zuvorkommenden Ton. "
                            "Verwenden Sie bei der Begrüßung von bekannten Personen deren vollständigen Titel und Nachnamen "
                            "(z.B. 'Professor Müller' für Professoren). "
                            "Ihre Antwort muss eine einzige, kombinierte Begrüßung sein, gefolgt unmittelbar von Ihrem Hilfsangebot. "
                            "Wenn der Bericht das Tag 'INITIAL_COUNT' enthält, ist die Zahl nach 'Ich sehe' die GESAMTZAHL der anwesenden Personen, "
                            "und die Zahl nach 'Unidentified people' ist die Anzahl der unbekannten Personen in diesem Total. "
                            "Begrüßen Sie die eingetretene(n) Person(en) mit großer Freude und bestätigen Sie die abwesende(n) Person(en). "
                            "Erwähnen Sie die Gesamtzahl der erkannten Personen. "
                            "Sprechen Sie die bekannten Personen (namentlich) und die unbekannten Personen (durch Zählung) getrennt an. "
                            "Vermeiden Sie Formulierungen wie 'einschließlich X unbekannter Personen', da dies so klingen kann, "
                            "als ob die Gesamtzahl größer ist. "
                            "Formulieren Sie stattdessen klar, zum Beispiel: "
                            "'Es sind 2 Personen anwesend. 2 davon sind mir unbekannt.' oder "
                            "'Es sind N Personen im Raum. X davon kenne ich nicht.' "
                            "Verwenden Sie nicht den genauen Satz 'Vision System Report'. "
                            "Antworten Sie nur auf Deutsch und kurz."
                        )

                    llm_query = f"{style_prompt}\nReport Content: {user_query}"
                else:
                    # Normal query
                    lang_instruction = (
                        "Respond only in English. "
                        if cur_lang == "en"
                        else "Antworten Sie ausschließlich auf Deutsch. "
                    )
                    llm_query = f"{lang_instruction}{user_query}"

                # --- Streaming RetrievalQA from LabAssistRAG ---
                try:
                    stream_or_str = self.rag.answer_query(llm_query, max_web=2)
                except Exception as e:
                    print(" RAG Error:", e)
                    self.output_queue.put(
                        self._make_speech(
                            "Sorry, I had a problem accessing my knowledge. Could you please repeat or try again?",
                            "sad",
                        )
                    )
                    self.output_queue.put(READY_SIGNAL)
                    self.input_queue.task_done()
                    continue

                # --- Handle stream vs. string result ---
                if isinstance(stream_or_str, str):
                    # If it's a string (non-streaming fallback or error message), put the whole thing
                    self.output_queue.put(self._make_speech(stream_or_str, emotion_for_answer))
                    self.output_queue.put(READY_SIGNAL)
                    self.input_queue.task_done()
                    continue

                # If it's a stream (the expected generator)
                stream = stream_or_str

                # --- Manual chunking for TTS ---
                buf = ""
                while self.running_flag():
                    try:
                        chunk = next(stream)
                        # OpenAI-style stream: delta.content holds incremental text
                        text_chunk = getattr(chunk.choices[0].delta, "content", "") or ""
                    except StopIteration:
                        break
                    except Exception as e:
                        print(" Streaming Error:", e)
                        break

                    if not text_chunk:
                        continue

                    buf += text_chunk

                    # Decide if we flush based on punctuation in the current buffer
                    should_flush_by_punct = any(p in buf for p in punctuation)

                    if should_flush_by_punct:
                        msg = buf.strip()
                        if msg:
                            self.output_queue.put(self._make_speech(msg, emotion_for_answer))
                        buf = ""
                    elif len(buf.split()) >= MIN_TTS_WORD_CHUNK:
                        # Flush at last space to avoid cutting a word in half
                        last_space = buf.rfind(" ")
                        if last_space != -1:
                            msg = buf[:last_space].strip()
                            if msg:
                                self.output_queue.put(self._make_speech(msg, emotion_for_answer))
                            buf = buf[last_space:].lstrip()

                # Flush any leftovers
                if buf.strip():
                    self.output_queue.put(self._make_speech(buf.strip(), emotion_for_answer))

                # Cue beep for next turn
                self.output_queue.put(READY_SIGNAL)
                self.input_queue.task_done()

            except Exception as e:
                print(" LLM Worker Error:", e)
                self.input_queue.task_done()
                time.sleep(0.5)

        print("LLM Worker finished.")
