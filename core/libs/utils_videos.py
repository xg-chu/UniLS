import av
import numpy as np
import torch


def write_video(video_frames, output_path, fps, audio_samples=None, sample_rate=None, acodec="aac"):
    assert video_frames.ndim == 4, "Input frames should be a 4D array."
    assert video_frames.shape[1] == 3, "Input frames should have 3 channels (RGB)."
    if isinstance(video_frames, torch.Tensor):
        video_frames = video_frames.cpu().numpy()
    if video_frames.dtype != np.uint8:
        video_frames = video_frames.astype(np.uint8)
    _, _, height, width = video_frames.shape
    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18"}
    if audio_samples is not None:
        if acodec == "aac":
            audio_stream = container.add_stream("aac", rate=sample_rate)
            audio_stream.format = "fltp"
        elif acodec == "vs_preview" or acodec == "debug":
            audio_stream = container.add_stream("mp3", rate=sample_rate)
            audio_stream.format = "fltp"
        else:
            raise ValueError("Unsupported audio codec.")

    for frame in video_frames:
        frame = frame.transpose(1, 2, 0)
        video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(video_frame):
            container.mux(packet)

    if audio_samples is not None:
        if isinstance(audio_samples, torch.Tensor):
            audio_samples = audio_samples.cpu().numpy()
        assert audio_samples.ndim == 1, "Input audio samples should be a 1D array."
        num_samples_per_frame = int(sample_rate // fps)
        for i in range(0, audio_samples.shape[0], num_samples_per_frame):
            # audio_frame = av.AudioFrame.from_ndarray(audio_samples[:, i:i + num_samples_per_frame], format="fltp", layout="mono")
            chunk = audio_samples[i : i + num_samples_per_frame]
            if chunk.shape[0] < num_samples_per_frame:
                chunk = np.pad(chunk, (0, num_samples_per_frame - chunk.shape[0]), mode="constant")
            audio_frame = av.AudioFrame.from_ndarray(chunk[None], format="fltp", layout="mono")
            audio_frame.rate = sample_rate
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    if audio_samples is not None:
        for packet in audio_stream.encode():
            container.mux(packet)

    container.close()


def read_video_frames(video_path):
    container = av.open(video_path)
    for frame in container.decode(video=0):
        yield torch.tensor(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)


def get_video_info(video_path):
    info_dict = {}
    container = av.open(video_path)
    video_stream = next((s for s in container.streams if s.type == "video"), None)
    if video_stream is None:
        info_dict["video"] = None
    else:
        info_dict["video"] = {
            "width": video_stream.width,
            "height": video_stream.height,
            "frame_rate": float(video_stream.average_rate),
            "num_frames": video_stream.frames,
        }
    audio_stream = next((s for s in container.streams if s.type == "audio"), None)
    if audio_stream is None:
        info_dict["audio"] = None
    else:
        info_dict["audio"] = {
            "channels": audio_stream.channels,
            "sample_rate": audio_stream.rate,
            "duration": audio_stream.duration,
        }
    return info_dict


def read_all_video_frames(video_path):
    container = av.open(video_path)
    video_stream = next((s for s in container.streams if s.type == "video"), None)
    if video_stream is None:
        print("No video stream found in the file.")
        return np.zeros((0), dtype=np.uint8), 0
    frames = []
    for frame in container.decode(video=0):  # Decode only video stream
        if frame.pts is None:  # Ignore invalid frames
            continue
        frames.append(frame.to_ndarray(format="rgb24"))
        # frame_id = int(frame.pts * video_stream.time_base * float(video_stream.average_rate))
    frames = torch.tensor(np.stack(frames, axis=0)).permute(0, 3, 1, 2)
    return frames, float(video_stream.average_rate)


def read_audio(audio_path, target_sr=96000):
    container = av.open(audio_path)
    audio_stream = container.streams.get(audio=0)
    if not audio_stream:
        container.close()
        return None, None
    audio_stream = audio_stream[0]
    # resampler = av.AudioResampler(format="fltp", layout=audio_stream.layout.name, rate=target_sr)
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)
    audio_data = []
    for frame in container.decode(audio_stream):
        resampled_frames = resampler.resample(frame)
        for r_frame in resampled_frames:
            audio_data.append(r_frame.to_ndarray())
    resampled_frames = resampler.resample(None)
    for r_frame in resampled_frames:
        audio_data.append(r_frame.to_ndarray())
    container.close()
    if not audio_data:
        return torch.tensor([]), target_sr
    full_audio_array = torch.tensor(np.concatenate(audio_data, axis=1))
    assert full_audio_array.dim() == 2, f"Invalid audio array shape: {full_audio_array.shape}."
    assert full_audio_array.shape[0] == 1, f"Invalid audio array shape: {full_audio_array.shape}."
    return full_audio_array[0], target_sr


if __name__ == "__main__":
    from tqdm import tqdm

    # Example Usage
    video_path = "../MultiTalk_dataset/multitalk_dataset/english/-OknSRRyFJE_0.mp4"
    vres, fps = read_all_video_frames(video_path)
    print(vres.shape, fps)

    video_length = get_video_info(video_path)["video"]["num_frames"]
    print(get_video_info(video_path))

    for frame in tqdm(read_video_frames(video_path), total=video_length):
        pass

    # write_video(vres, "output_debug.mp4", fps)
