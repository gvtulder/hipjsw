import numpy as np
import pydicom

def load_dicom_image(dicom_path):
    # load the DICOM image and apply the PhotometricInterpretation header
    # (if necessary)

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
