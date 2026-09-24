"""
Utility functions.
"""
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 Gijs van Tulder / TU Delft

import imageio
import os
import os.path

def save_grayscale_image(filename, image):
    """Save a grayscale image to PNG or JPEG."""
    image = image.astype(float) - image.min()
    image *= 255 / image.max()
    imageio.imsave(filename, image.astype('uint8'))


def ensure_parent_directory(filename):
    """Recursively create the parent directories for this filename."""
    parent = os.path.dirname(filename)
    if parent:
        os.makedirs(parent, exist_ok=True)
