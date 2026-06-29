"""
Utility functions.
"""
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 Gijs van Tulder / TU Delft

import imageio

def save_grayscale_image(filename, image):
    """Save a grayscale image to PNG or JPEG."""
    image = image.astype(float) - image.min()
    image *= 255 / image.max()
    imageio.imsave(filename, image.astype('uint8'))

