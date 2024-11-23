import concurrent.futures
import os
import shutil
import zipfile
import sieve
import subprocess



@sieve.function(
    name="speaker-enhance",
    metadata=sieve.Metadata(
        title="Speaker Enhance",
        description="Enhance a talking head video",
        tags=["talking head", "video", "speaker", "enhance", "background", "eye contact"],
        image=sieve.File(path=os.path.join(os.path.dirname(__file__), "icon.jpeg")),
        code_url="https://github.com/sieve-community/speaker-enhance",
        readme=open(os.path.join(os.path.dirname(__file__), "README.md")).read(),
    ),
    python_packages=["opencv-python-headless", "numpy"],
    system_packages=["ffmpeg"],
)
def enhance_speaker(
    video: sieve.File,
    background_img: sieve.File = None,
    blur_background: bool = False,
    blur_strength: int = 19,
    background_color_rgb: str = "",
) -> sieve.File:
    import cv2
    import numpy as np
    from utils import masked_blur, write_video_futures, write_output

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

    if blur_strength <= 0:
        blur_background = False

    if len(background_color_rgb) == 3 and background_img is not None:
        print("Overriding background_color_rgb with background_img")

    # start background removal job
    bgr_out = bgr_function.push(
        video, backend="parallax", video_output_format="zip", output_type="raw_mask"
    )

    # start eye contact correction job
    ecc_out = ecc_function.push(video)

    # extract audio from video
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

    # start audio enhance job
    audio_enhance_out = audio_enhance_fn.push(sieve.File(audio_path), backend="auphonic")

    if background_img is not None:
        # resize to fit video dimensions
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
            background_img = masked_blur(
                background_img,
                np.zeros((video_height, video_width), dtype=np.bool_),
                kernel_size=(blur_strength, blur_strength),
            )

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

    # write output frames
    writer = concurrent.futures.ThreadPoolExecutor()
    write_futures = []
    print("Writing output frames...")
    for i, (ecc_frame_path, bgr_frame_path, ecc_future) in enumerate(
        zip(ecc_frame_paths, bgr_frame_paths, ecc_futures)
    ):
        output_path = os.path.join(output_dir, f"{str(i).zfill(zeros)}.png")
        write_futures.append(
            writer.submit(
                write_output,
                ecc_future,
                ecc_frame_path,
                bgr_frame_path,
                output_path,
                background_img,
                background_color_rgb if len(background_color_rgb) == 3 else None,
                blur_strength,
            )
        )

    completed = 0
    for _ in concurrent.futures.as_completed(write_futures):
        completed += 1
        if completed % 100 == 0:
            print(f"Finished writing frame {completed} of {len(write_futures)}...")

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

    # get audio enhance result
    audio_enhance_out_path = audio_enhance_out.result().path
    # overlay enhanced audio on output video
    output_video_path = "/tmp/speaker_enhance_output_with_audio.mp4"
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
        blur_background=True,
    )
