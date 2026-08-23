import io

import av
import numpy as np

CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}

CODEC_MAP = {
    "wav": ("pcm_s16le", "wav"),
    "mp3": ("mp3", "mp3"),
    "opus": ("libopus", "ogg"),
    "flac": ("flac", "flac"),
    "aac": ("aac", "adts"),
}

BITRATE_FORMATS = {"mp3", "aac", "opus"}


class StreamingAudioWriter:
    def __init__(self, output_format: str, sample_rate: int = 24000) -> None:
        self.output_format = output_format
        self.sample_rate = sample_rate
        self.pts = 0
        self._read_pos = 0

        if output_format == "pcm":
            self.container = None
            self.stream = None
            return

        codec, container_format = CODEC_MAP[output_format]
        self._buffer = io.BytesIO()
        options = {"write_xing": "0"} if output_format == "mp3" else {}
        self.container = av.open(
            self._buffer, mode="w", format=container_format, options=options
        )
        self.stream = self.container.add_stream(codec, rate=sample_rate)
        self.stream.layout = "mono"

        if output_format in BITRATE_FORMATS:
            self.stream.bit_rate = 128000
        if output_format == "opus":
            self.stream.rate = sample_rate

    def _read_new_bytes(self) -> bytes:
        self._buffer.seek(0, 2)
        end_pos = self._buffer.tell()
        if self._read_pos >= end_pos:
            return b""
        self._buffer.seek(self._read_pos)
        new_bytes = self._buffer.read(end_pos - self._read_pos)
        self._read_pos = end_pos
        return new_bytes

    def write_chunk(self, audio_float32: np.ndarray) -> bytes:
        audio_int16 = (audio_float32 * 32767).clip(-32768, 32767).astype(np.int16)

        if self.output_format == "pcm":
            return audio_int16.tobytes()

        frame = av.AudioFrame.from_ndarray(
            audio_int16.reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate = self.sample_rate
        frame.pts = self.pts
        self.pts += audio_int16.shape[0]

        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        return self._read_new_bytes()

    def finalize(self) -> bytes:
        if self.output_format == "pcm":
            return b""

        for packet in self.stream.encode(None):
            self.container.mux(packet)
        self.container.close()
        return self._read_new_bytes()

    def close(self) -> None:
        if self.output_format == "pcm":
            return
        if self.container is not None:
            try:
                self.container.close()
            except Exception:
                pass
            finally:
                self.container = None
                self.stream = None
        self._buffer = io.BytesIO()
        self._read_pos = 0
