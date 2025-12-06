import pyaudio
pa = pyaudio.PyAudio()
print("Input devices:")
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if info.get("maxInputChannels", 0) > 0:
        print(f"[{i}] {info['name']}  ({int(info['defaultSampleRate'])} Hz)")
pa.terminate()
