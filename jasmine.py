import pyaudio
import wave
import whisper
import webrtcvad
import numpy as np
import os
import tempfile
import warnings
import time
import pvporcupine
from pvrecorder import PvRecorder
from scipy.signal import butter, lfilter
from chat import send_command_to_ai, stream_voice_response
from pydub import AudioSegment
from pydub.playback import play
from dotenv import load_dotenv
from intent import classify_intent
from agents.agent_manager import execute_agent

load_dotenv()

AI_ASSISTANT_NAME = os.getenv("AI_ASSISTANT_NAME")
PORCUPINE_ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY")
PORCUPINE_WAKE_WORD = os.getenv("PORCUPINE_WAKE_WORD")
WHISPER_MODEL = os.getenv("WHISPER_MODEL")

# Suppress warnings 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

# Load Whisper model for transcription
model = whisper.load_model(WHISPER_MODEL)

# Load Porcupine wake word
porcupine = pvporcupine.create(access_key=PORCUPINE_ACCESS_KEY, keyword_paths=[PORCUPINE_WAKE_WORD])
try:
    recorder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)
except RuntimeError as e:
    print("Error initializing PvRecorder:")
    if "Failed to initialize" in str(e):
        print("The microphone might not be found or accessible.")
    else:
        print(f"Unexpected error: {e}")

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 512
VAD_MODE = 2
TIMEOUT_SECONDS = 15
SILENCE_FRAMES = 20

# Initialize PyAudio and VAD
p = pyaudio.PyAudio()

for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    print(f"Device {i}: {device_info['name']} (Input: {device_info['maxInputChannels']}, Output: {device_info['maxOutputChannels']})")

vad = webrtcvad.Vad()
vad.set_mode(VAD_MODE)

# Noise filtering
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut=300.0, highcut=3400.0, fs=16000, order=6):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def is_speech(frame, rate):
    pcm_data = np.frombuffer(frame, dtype=np.int16)
    filtered_audio = apply_bandpass_filter(pcm_data)

    frame_length = int(rate * 0.01)
    if len(filtered_audio) < frame_length:
        return False

    frame_10ms = filtered_audio[:frame_length].astype(np.int16).tobytes()
    return vad.is_speech(frame_10ms, rate)

def play_beep(beep_type="wake"):
    """Plays a beep sound with corrected format settings."""
    beep_file = "sounds/beep_wake.wav" if beep_type == "wake" else "sounds/beep_timeout.wav"

    # Force format conversion to 16-bit PCM
    sound = AudioSegment.from_file(beep_file, format="wav")
    sound = sound.set_channels(1).set_frame_rate(16000).set_sample_width(2)

    play(sound)

def play_error_beep():
    """Plays an error beep sound with corrected format settings."""
    sound = AudioSegment.from_file("sounds/buzz.wav", format="wav")
    sound = sound.set_channels(1).set_frame_rate(16000).set_sample_width(2)

    play(sound)

def listen_for_wake_word():
    print(f"👂 Waiting for wake word '{AI_ASSISTANT_NAME}'...")
    recorder.start()

    try:
        while True:
            pcm = recorder.read()
            keyword_index = porcupine.process(pcm)

            if keyword_index >= 0:
                print(f"✅ Wake word detected: '{AI_ASSISTANT_NAME}'")
                recorder.stop()
                
                # Play wake-up beep sound (distinct)
                play_beep(beep_type="wake")

                return True

    except KeyboardInterrupt:
        print("\n🛑 Exiting... Stopping wake-word listener.")
        recorder.stop()
        cleanup()
        exit(0)

def record_audio():
    """Continuously listens, transcribes, and pauses while AI speaks."""
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    try:
        while True:
            if listen_for_wake_word():
                print("🎤 Listening for speech...")

                frames = []
                silence_frames = 0
                recording = False
                last_speech_time = time.time()
                in_ai_processing = False

                try:
                    while True:
                        audio_chunk = stream.read(CHUNK, exception_on_overflow=False)
                        is_speaking = is_speech(audio_chunk, RATE)

                        if is_speaking:
                            if not recording:
                                print("🎙️ Speech detected, recording...")
                                recording = True
                                frames = []

                            frames.append(audio_chunk)
                            silence_frames = 0
                            last_speech_time = time.time()

                        elif recording:
                            silence_frames += 1
                            if silence_frames > SILENCE_FRAMES:
                                print("⏸️ Silence detected, stopping recording.")
                                
                                # Pause timeout while processing AI command
                                in_ai_processing = True
                                ai_processing_start = time.time()
                                save_and_transcribe(frames)
                                ai_processing_end = time.time()
                                in_ai_processing = False

                                # Adjust timeout calculation to exclude AI processing time
                                last_speech_time += (ai_processing_end - ai_processing_start)

                                frames = []
                                recording = False
                                silence_frames = 0

                        # Only track timeout if AI is not processing
                        if not in_ai_processing and (time.time() - last_speech_time > TIMEOUT_SECONDS):
                            print("⏳ Timeout: No speech detected after wake word.")
                            
                            # Play timeout beep sound (distinct)
                            play_beep(beep_type="timeout")

                            break

                except KeyboardInterrupt:
                    print("\n🛑 Exiting... Stopping speech recording.")
                    stream.stop_stream()
                    stream.close()
                    cleanup()
                    exit(0)

    except KeyboardInterrupt:
        print("\n🛑 Exiting... Stopping audio recording.")
        stream.stop_stream()
        stream.close()
        cleanup()
        exit(0)

def save_and_transcribe(frames):
    """Saves audio, transcribes it, and sends to AI."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        tmp_filename = tmp_audio.name

        with wave.open(tmp_filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

        print("🔄 Transcribing...")
        start_time = time.time()
        command = model.transcribe(tmp_filename, no_speech_threshold=0.1)["text"].strip()
        end_time = time.time()
        transcription_time = end_time - start_time

        print(f"⏱️ Transcription took {transcription_time:.2f} seconds.")

        if command:
            print(f"📝 Transcription: {command}")
            os.remove(tmp_filename)

            # Classify intent (LLM chat or agent)
            intent = classify_intent(command)

            print(f"🤖 Intent: {intent}")

            if intent == "llm" or intent == "other":
                response = send_command_to_ai(command)
            else:
                response = execute_agent(intent, command)

            if response:
                stream_voice_response(response)
        else:
            print("❌ Transcription failed: Empty result.")
            play_error_beep()
            os.remove(tmp_filename)

def cleanup():
    """Gracefully stops all resources."""
    print("🛑 Cleaning up resources...")
    try:
        recorder.stop()
    except:
        pass
    try:
        porcupine.delete()
    except:
        pass
    try:
        p.terminate()
    except:
        pass


if __name__ == "__main__":
    try:
        record_audio()
    except KeyboardInterrupt:
        print("\n🛑 Exiting...")
        cleanup()
        exit(0)