## Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##
"""The utilities for image comparison"""
from pathlib import Path
from pickletools import int4
import sys
import os
import platform

import omni.kit.test


OUTPUTS_DIR = Path(omni.kit.test.get_test_output_path())

class CompareError(Exception):
    pass


def compare_img(image_name: str, golden_img_dir: Path, filter_rect: int4 = None):
    """
    add a rect filter in omni.kit.test::compare

    """

    image1 = OUTPUTS_DIR.joinpath(image_name)
    image2 = golden_img_dir.joinpath(image_name)
    image_diffmap = OUTPUTS_DIR.joinpath(f"{Path(image_name).stem}.diffmap.png")
    alt_image2 = golden_img_dir.joinpath(f"{platform.system().lower()}/{image_name}")
    if os.path.exists(alt_image2):
        image2 = alt_image2

    if not image1.exists():
        raise CompareError(f"File image1 {image1} does not exist")
    if not image2.exists():
        raise CompareError(f"File image2 {image2} does not exist")

    if "PIL" not in sys.modules.keys():
        # Checking if we have Pillow imported
        try:
            from PIL import Image
        except ImportError:
            # Install Pillow if it's not installed
            import omni.kit.pipapi

            omni.kit.pipapi.install("Pillow", module="PIL")

    from PIL import Image
    from PIL import ImageChops

    original = Image.open(str(image1))
    contrast = Image.open(str(image2))

    if original.size != contrast.size:
        raise CompareError(
            f"[omni.ui.test] Can't compare different resolutions\n\n"
            f"{image1} {original.size[0]}x{original.size[1]}\n"
            f"{image2} {contrast.size[0]}x{contrast.size[1]}\n\n"
            f"It's possible that your monitor DPI is not 100%.\n\n"
        )

    # set black color in the rect 
    if filter_rect:
        for x in range(filter_rect[0], filter_rect[2]):
            for y in range(filter_rect[1], filter_rect[3]):
                original.putpixel((x, y), 0)
                contrast.putpixel((x, y), 0)

    difference = ImageChops.difference(original, contrast).convert("RGB")
    max_difference = sum([sum(i) for i in difference.getextrema()])
    if max_difference > 0:
        # Images are different
        # Multiply image by 255
        difference = difference.point(lambda i: min(i * 255, 255))
        difference.save(str(image_diffmap))
    return max_difference