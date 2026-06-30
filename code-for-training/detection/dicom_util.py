"""
Utility code to load DICOM files and apply simple intensity transformations.
"""
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 Gijs van Tulder / TU Delft

import numpy as np
import pydicom

def load_dicom_image(dicom_path):
    """Load a DICOM image from a file.

    Loads the DICOM image and applies the PhotometricInterpretation header
    by inverting the image (if necessary).

    Parameters
    ----------
    dicom_path : str
        the path to a DICOM image

    Returns
    -------
    ImageWithSpacing
        the loaded image

    """
    img = pydicom.dcmread(dicom_path)

    pixel_spacing = img.get('PixelSpacing') or img.get('ImagerPixelSpacing')
    # assert pixel_spacing is not None, 'no pixel spacing found'
    if pixel_spacing is not None:
        assert pixel_spacing[0] == pixel_spacing[1], \
               'anisotropic pixel spacing is untested'

    pixels = img.pixel_array

    # are the intensities stored as MONOCHROME2 (white=max, black=min) or
    # as MONOCHROME1 (white=min, black=max)?
    photometric_interpretation = img.get('PhotometricInterpretation')
    if photometric_interpretation == 'MONOCHROME1':
        # inverting intensities
        pixels = np.max(pixels) - pixels
    else:
        assert photometric_interpretation == 'MONOCHROME2', \
               f'{photometric_interpretation} not supported'

    # other checks
    assert img.get('VOILUTFunction', 'LINEAR') == 'LINEAR', \
        'only supporting VOILUTFunction LINEAR'

    return img, pixels, pixel_spacing
