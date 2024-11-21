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
def enhance_speaker(
    video: sieve.File,
    background_img: sieve.File = None,
    blur_background: bool = False,
    background_color_rgb: str = "",
) -> sieve.File:
    import cv2
    import numpy as np
    from utils import masked_blur

    zeros = 8
    audio_enhance_fn = sieve.function.get("sieve/audio-enhance")
    bgr_function = sieve.function.get("sieve/background-removal")
    ecc_function = sieve.function.get("sieve/eye-contact-correction")

    if background_img is not None:
        try:
            background_img.path
        except Exception:
            background_img = None

    if background_color_rgb:
        try:
            background_color_rgb = [int(x) for x in background_color_rgb.split(",")]
        except Exception as e:
            raise ValueError(
                "background_color_rgb must be a comma-separated string of three integers between 0 and 255 if provided"
            ) from e
    else:
        background_color_rgb = []

    if background_img is None and len(background_color_rgb) != 3 and not blur_background:
        raise ValueError(
            "Must provide background_img, background_color_rgb, or set blur_background to True"
        )

    if len(background_color_rgb) == 3 and background_img is not None:
        print("Overriding background_color_rgb with background_img")

    bgr_out = bgr_function.push(
        video, backend="parallax", video_output_format="zip", output_type="raw_mask"
    )

    # eye contact correction is next most expensive
    ecc_out = ecc_function.push(video)

    # audio enhance takes the longest usually so push it first
    audio_path = "/tmp/audio.wav"
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

    if background_img is not None:
        # resize to fit video dimensions
        # get video height, width
        cap = cv2.VideoCapture(video.path)
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        img = cv2.imread(background_img.path)
        if img is not None:
            background_img = cv2.resize(img, (video_width, video_height))
        else:
            raise ValueError("Failed to load background_img")
        if blur_background:
            background_img = masked_blur(background_img, np.zeros((video_height, video_width), dtype=np.bool_))

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
    bgr_frame_paths = sorted(
        [
            os.path.join(bgr_masks_dir, f)
            for f in os.listdir(bgr_masks_dir)
            if f.endswith(".png")
        ]
    )

    # create output frames
    output_dir = "/tmp/output_frames"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    writer = concurrent.futures.ThreadPoolExecutor()
    write_futures = []
    print("Writing output frames...")
    for i, (ecc_frame_path, bgr_frame_path, ecc_future) in enumerate(
        zip(ecc_frame_paths, bgr_frame_paths, ecc_futures)
    ):
        if i % 100 == 0:
            print(f"Writing frame {i} of {len(ecc_frame_paths)}...")
        ecc_future.result()  # wait for frame to be written
        ecc_frame = cv2.imread(ecc_frame_path)
        mask = cv2.imread(bgr_frame_path, cv2.IMREAD_GRAYSCALE) > 128
        if background_img is not None:
            masked_frame = np.where(mask[..., np.newaxis], ecc_frame, background_img)
        elif len(background_color_rgb) == 3:
            masked_frame = np.where(
                mask[..., np.newaxis], ecc_frame, np.array(background_color_rgb[::-1])
            )
        else:
            blurred_background = masked_blur(ecc_frame, mask)
            masked_frame = np.where(mask[..., np.newaxis], ecc_frame, blurred_background)

        output_path = os.path.join(output_dir, f"{str(i).zfill(zeros)}.png")
        write_futures.append(writer.submit(cv2.imwrite, output_path, masked_frame))

    concurrent.futures.wait(write_futures)
    print("Finished writing output frames, creating output video...")

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
            os.path.join(output_dir, f"%{zeros}d.png"),
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
    enhance_speaker(
        sieve.File("/home/azureuser/sample_inputs/karp.mp4"),
        background_img=sieve.File(
            url="https://www.bu.edu/files/2022/07/feat-STScI-01G7ETNMR8CBHQQ64R4CVA1E6T.jpg"
        ),
        blur_background=True,
    )
