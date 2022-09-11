import av
import numpy as np
import torch

audio_format_dtypes = {
    "dbl": "<f8",
    "dblp": "<f8",
    "flt": "<f4",
    "fltp": "<f4",
    "s16": "<i2",
    "s16p": "<i2",
    "s32": "<i4",
    "s32p": "<i4",
    "u8": "u1",
    "u8p": "u1",
}
import subprocess


def write_video(dest_dir, filename, aframes, metadata, audio=True, video_codec="h264"):
    audio_file = "audio.aac"
    num_channels = aframes.shape[0]
    with av.open(audio_file, "w") as container:
        if audio:
            a_stream = container.add_stream("aac", rate=metadata["audio_fps"])
            audio_layout = "stereo" if num_channels > 1 else "mono"
            audio_sample_fmt = container.streams.audio[0].format.name
            format_dtype = np.dtype(audio_format_dtypes[audio_sample_fmt])
            audio_array = torch.as_tensor(aframes).numpy().astype(format_dtype)
            frame = av.AudioFrame.from_ndarray(
                audio_array, format=audio_sample_fmt, layout=audio_layout
            )
            frame.sample_rate = metadata["audio_fps"]
            container.mux(a_stream.encode(frame))
    cmd = f"ffmpeg -y -hide_banner -framerate {metadata['video_fps']} -i {str(dest_dir)}/predicted_flow_%d.jpg -i {audio_file} -loop -1 -c:a aac {filename}"
    subprocess.run(cmd, shell=True)
