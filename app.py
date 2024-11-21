import concurrent.futures
import os
import shutil
import zipfile
import sieve
import subprocess

from utils import write_video_futures


@sieve.function(
    name="speaker-enhance",
    metadata=sieve.Metadata(
        description="Enhance a talking head video",
    ),
    python_packages=["opencv-python-headless", "numpy"],
    system_packages=["ffmpeg"],
)
def enhance_speaker(video: sieve.File) -> sieve.File:
    import cv2
    import numpy as np

    zeros = 8
    audio_enhance_fn = sieve.function.get("sieve/audio-enhance")
    bgr_function = sieve.function.get("sieve/background-removal")
    ecc_function = sieve.function.get("sieve/eye-contact-correction")

    # push jobs in parallel

    # background removal takes the longest usually so push it first
    bgr_out = bgr_function.push(
        video, backend="parallax", video_output_format="zip", output_type="raw_mask"
    )

    # eye contact correction is next most expensive
    ecc_out = ecc_function.push(video)
    # extract audio from video
    audio_path = "audio.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            video.path,
            "-q:a",
            "0",
            "-map",
            "a",
            audio_path,
        ]
    )

    audio_enhance_out = audio_enhance_fn.push(sieve.File(audio_path))

    from utils import get_fps

    original_video_fps = get_fps(video.path)

    # get ecc result
    ecc_output = ecc_out.result()

    # dump ecc frames
    ecc_frames_dir = "/tmp/ecc_frames"
    if os.path.exists(ecc_frames_dir):
        shutil.rmtree(ecc_frames_dir)
    os.makedirs(ecc_frames_dir)
    ecc_futures, ecc_frame_paths = write_video_futures(
        ecc_output.path, ecc_frames_dir, zeros=zeros
    )

    # unzip bgr masks
    bgr_masks_dir = "/tmp/bgr_masks"
    if os.path.exists(bgr_masks_dir):
        shutil.rmtree(bgr_masks_dir)
    os.makedirs(bgr_masks_dir)
    result = next(bgr_out.result())
    with zipfile.ZipFile(result.path, "r") as zip_ref:
        zip_ref.extractall(bgr_masks_dir)
    bgr_frame_paths = sorted([
        os.path.join(bgr_masks_dir, f)
        for f in os.listdir(bgr_masks_dir)
        if f.endswith(".png")
    ])

    # create output frames
    output_dir = "/tmp/output_frames"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    writer = concurrent.futures.ThreadPoolExecutor()
    write_futures = []
    for i, (ecc_frame_path, bgr_frame_path, ecc_future) in enumerate(
        zip(ecc_frame_paths, bgr_frame_paths, ecc_futures)
    ):
        ecc_future.result()  # wait for frame to be written
        ecc_frame = cv2.imread(ecc_frame_path)
        mask = cv2.imread(bgr_frame_path, cv2.IMREAD_GRAYSCALE) > 128

        masked_frame = np.where(mask[..., np.newaxis], ecc_frame, 0)
        output_path = os.path.join(output_dir, f"{str(i).zfill(zeros)}.png")
        write_futures.append(writer.submit(cv2.imwrite, output_path, masked_frame))

    concurrent.futures.wait(write_futures)

    # combine output frames into video
    temp_output_video_path = "/tmp/speaker_enhance_output.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(original_video_fps),
            "-i",
            os.path.join(output_dir, "%d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            temp_output_video_path,
        ]
    )

    # overlay audio on output video
    output_video_path = "/tmp/speaker_enhance_output_with_audio.mp4"
    audio_enhance_out_path = audio_enhance_out.result().path
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            temp_output_video_path,
            "-i",
            audio_enhance_out_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-strict",
            "experimental",
            output_video_path,
        ]
    )

    return sieve.File(output_video_path)


if __name__ == "__main__":
    enhance_speaker(sieve.File("/home/azureuser/sample_inputs/karp.mp4"))
