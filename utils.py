
import concurrent.futures
import cv2
import os
import subprocess

import numpy as np

def write_video_futures(
    video_path: str,
    tempdir: str,
    start_frame: int = -1,
    end_frame: int = -1,
    frame_interval: int = 1,
    zeros: int = 8,
):
    cap = cv2.VideoCapture(video_path)
    if start_frame == -1:
        start_frame = 0
    if end_frame == -1:
        end_frame = None
    assert start_frame >= 0
    assert start_frame < end_frame if end_frame is not None else True
    frame_paths = []
    futures = []
    frame_num = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num >= start_frame:
                if frame_num % frame_interval == 0:
                    if end_frame is None or frame_num < end_frame:
                        frame_path = os.path.join(
                            tempdir, f"{str(frame_num).zfill(zeros)}.png"
                        )
                        frame_paths.append(frame_path)
                        futures.append(executor.submit(cv2.imwrite, frame_path, frame))
            frame_num += 1
            if end_frame is not None and frame_num > end_frame:
                break
    return futures, frame_paths


def get_fps(video_file_path: str):
    file_fps = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_file_path,
        ],
        capture_output=True,
        text=True,
    )
    if "/" in file_fps.stdout:
        file_fps = float(file_fps.stdout.split("/")[0]) / float(
            file_fps.stdout.split("/")[1]
        )
    else:
        file_fps = float(file_fps.stdout)
    return file_fps


def masked_blur(img, mask, kernel_size=(19, 19)):
    # weighted blur kernel
    inv_mask = ~mask
    conv_sum = cv2.boxFilter(inv_mask.astype(np.float32), -1, kernel_size, normalize=False).clip(min=0)
    inv_masked_frame = img * inv_mask[..., np.newaxis]
    # apply 2d kernel to 3d img
    blurred_frame = cv2.boxFilter(inv_masked_frame, cv2.CV_32F, kernel_size, normalize=False) / (
        conv_sum[..., np.newaxis] + 1e-7
    ).clip(min=0, max=255).astype(np.uint8)
    return np.where(mask[..., np.newaxis], img, blurred_frame)
