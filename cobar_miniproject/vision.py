import cv2
from flygym import Fly
import numpy as np
from pathlib import Path
import imageio
from copy import deepcopy


def get_fly_vision(fly: Fly):
    assert (
        fly._curr_visual_input is not None
    ), "fly vision isn't enabled. Make sure `enable_vision` is set to True."
    return 255 - (
        255
        * np.hstack(
            [
                fly.retina.hex_pxls_to_human_readable(
                    fly._curr_visual_input[eye], True
                ).max(axis=2)[::2, ::2]
                for eye in range(2)
            ]
        )
    ).astype(np.uint8)


def get_fly_vision_raw(fly: Fly):
    assert (
        fly._curr_raw_visual_input is not None
    ), "fly vision isn't enabled. Make sure `render_raw_vision` is set to True."

    return np.hstack(tuple(fly._curr_raw_visual_input))


def render_image_with_vision(
    image: np.ndarray,
    vision: np.ndarray,
    odor_intensity: np.ndarray,
):
    if vision.ndim < 3:
        vision = vision[..., np.newaxis]
    if vision.shape[2] == 1:
        vision = vision.repeat(3, axis=2)

    if image.shape[1] > vision.shape[1]:
        vision = np.pad(
            vision,
            ((0,), ((image.shape[1] - vision.shape[1]) // 2,), (0,)),
            constant_values=255,
        )
    elif vision.shape[1] > image.shape[1]:
        image = np.pad(
            image,
            ((0,), ((vision.shape[1] - image.shape[1]) // 2,), (0,)),
            constant_values=255,
        )
    
    im_olf = np.full((16, image.shape[1], 3), 255, dtype=np.uint8)
    x0 = image.shape[1] // 2
    odor_intensity = odor_intensity[0] # only display the 1st odor dimension
    # average over the two types of sensors (antennae and maxillary palps)
    # and calculate the right minus left difference
    diff = odor_intensity.reshape((2, 2)).mean(0) @ (-1, 1)
    log = np.log10(np.abs(diff))
    vmin = -7
    vmax = -1
    norm = np.clip((log - vmin) / (vmax - vmin), 0, 1)
    dx = np.round(norm * x0 * np.sign(diff)).astype(int)
    cv2.rectangle(
        im_olf,
        (x0, 0),
        (x0 + dx, im_olf.shape[0]),
        (255, 127, 14),
        -1,
        cv2.LINE_AA,
    ) 
    cv2.line(
        im_olf,
        (x0, 0),
        (x0, im_olf.shape[0]),
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((vision, im_olf, image))


class RawVideoHandler:
    def __init__(self, file_name_prefix):
        self.save_path = Path("outputs")/f"{file_name_prefix}_raw_vision.mp4"
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def __enter__(self): 
        self.writer = imageio.get_writer(self.save_path, fps=24) # unsure about fps

    def __exit__(self, exc_type, exc_value, traceback):
        self.writer.close()
    
    def handle_raw_video(self, info, obs, delete_video_from_dicts=True): 
        if delete_video_from_dicts: 
            frame = deepcopy(info["raw_vision"])
            del info["raw_vision"]
            del obs["raw_vision"]
        else:
            frame = info["raw_vision"]
    

        frame = frame.astype(np.uint8)
        frame = np.concatenate((frame[0, :, :, :], frame[1, :, :, :]), axis=1)

        # This is to make imageio stop complaining:
        # input image is not divisible by macro_block_size=16, resizing from (900, 514) to (912, 528) to ensure video compatibility with most codecs 
        # and players. To prevent resizing, make your input image divisible by the macro_block_size or set the macro_block_size to 1 
        # (risking incompatibility).
        if frame.shape[0] % 16 != 0 or frame.shape[1] % 16 != 0: 
            new_height = (frame.shape[0] + 15) // 16 * 16
            new_width = (frame.shape[1] + 15) // 16 * 16
            frame = np.pad(
                frame,
                ((0, new_height - frame.shape[0]), (0, new_width - frame.shape[1]), (0, 0)),
                mode='constant',
                constant_values=0
            )

        self.writer.append_data(frame)

    
