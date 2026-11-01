"""Exercise actual ffmpeg MP3 encoding/decoding with generated test audio only."""

import array
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import tempfile
import wave

rate = 16000
frames = 4000
samples = array.array(
    "h",
    (
        round(6000 * math.sin(2 * math.pi * 440 * index / rate))
        for index in range(frames)
    ),
)
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    source, mp3, decoded = [
        root / name for name in ("generated.wav", "encoded.mp3", "decoded.wav")
    ]
    with wave.open(str(source), "wb") as output:
        output.setparams((1, 2, rate, frames, "NONE", "not compressed"))
        output.writeframes(samples.tobytes())
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-c:a",
            "libmp3lame",
            "-b:a",
            "32k",
            str(mp3),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(mp3),
            "-ac",
            "1",
            "-ar",
            str(rate),
            "-c:a",
            "pcm_s16le",
            str(decoded),
        ],
        check=True,
    )
    stream = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(mp3)],
            text=True,
        )
    )["streams"][0]
    assert stream["codec_name"] == "mp3" and stream["channels"] == 1
    with wave.open(str(decoded), "rb") as output:
        assert (
            output.getnchannels(),
            output.getsampwidth(),
            output.getframerate(),
        ) == (1, 2, rate)
        output_frames = output.getnframes()
        payload = output.readframes(output_frames)
    assert frames * 0.9 <= output_frames <= frames + rate / 5
    assert any(array.array("h", payload))
    result = {
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "ffmpeg": subprocess.check_output(
            ["ffmpeg", "-version"], text=True
        ).splitlines()[0],
        "encoder": "libmp3lame",
        "encoded_codec": stream["codec_name"],
        "generated_wav_frames": frames,
        "decoded_wav_frames": output_frames,
        "sample_rate": rate,
        "channels": 1,
        "generated_wav_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "mp3_bytes": mp3.stat().st_size,
        "decoded_wav_bytes": decoded.stat().st_size,
        "round_trip_passed": True,
    }
    print(json.dumps(result, indent=2))
