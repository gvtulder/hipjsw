import argparse
import numpy as np
import imageio
import skimage

from . import dicom_util
from . import util

from . import detect

# prepare HDF5 files for a joint space segmentation task
# based on a CSV list of input files


class ImageWithSpacing:
    def __init__(self, pixels, pixel_spacing, pixel_spacing_source=None):
        self.pixels = pixels
        # the pixel spacing of the image
        self.pixel_spacing = pixel_spacing
        # a string describing the source of the pixel spacing (file, given, estimated, resampled)
        self.pixel_spacing_source = pixel_spacing_source

    def resample(self, target_pixel_spacing):
        # resample to the required resolution
        assert self.pixel_spacing[0] == self.pixel_spacing[1]
        scale_factor = self.pixel_spacing[0] / target_pixel_spacing
        img_pixels = skimage.transform.rescale(self.pixels, scale_factor)
        pixel_spacing = [target_pixel_spacing, target_pixel_spacing]
        return ImageWithSpacing(img_pixels, pixel_spacing, 'resampled'), scale_factor

    def __getitem__(self, *idx):
        # crop image
        return ImageWithSpacing(self.pixels.__getitem__(*idx), self.pixel_spacing, self.pixel_spacing_source)

    def astype(self, *args, **kwargs):
        return ImageWithSpacing(self.pixels.astype(*args, **kwargs), self.pixel_spacing, self.pixel_spacing_source)

    def __sub__(self, *args, **kwargs):
        return ImageWithSpacing(self.pixels.__sub__(*args, **kwargs), self.pixel_spacing, self.pixel_spacing_source)

    def __truediv__(self, *args, **kwargs):
        return ImageWithSpacing(self.pixels.__truediv__(*args, **kwargs), self.pixel_spacing, self.pixel_spacing_source)

    @property
    def shape(self):
        return self.pixels.shape

    def __repr__(self):
        return f'Image<{self.pixels.shape}, {self.pixel_spacing} pixels/mm>'


def load_dicom_image(input_path, pixel_spacing=None, pixel_spacing_source=None):
    _, img_pixels, file_pixel_spacing = dicom_util.load_dicom_image(input_path)
    if file_pixel_spacing is not None and pixel_spacing is not None:
        matched = \
            np.round(file_pixel_spacing[0], 3) == np.round(pixel_spacing, 3) and \
            np.round(file_pixel_spacing[1], 3) == np.round(pixel_spacing, 3)
        if not matched:
            print(f'ignoring forced pixel spacing: input {input_path} already contains pixel spacing, but {file_pixel_spacing} != {pixel_spacing}')

    if file_pixel_spacing is not None:
        pixel_spacing = list(file_pixel_spacing)
        pixel_spacing_source = 'file'
    elif pixel_spacing is not None:
        pixel_spacing = [pixel_spacing, pixel_spacing]

    return ImageWithSpacing(img_pixels, pixel_spacing, pixel_spacing_source)


def load_jpeg_image(input_path, pixel_spacing=None, pixel_spacing_source=None):
    img_pixels = imageio.v2.imread(input_path).astype(float)
    if img_pixels.ndim == 3:
        img_pixels = np.mean(img_pixels, axis=2)
    return ImageWithSpacing(img_pixels,
                            [pixel_spacing, pixel_spacing] if pixel_spacing else None,
                            pixel_spacing_source)


def load_image(input_path, pixel_spacing=None, pixel_spacing_source='given'):
    if input_path.lower().endswith('.dcm'):
        return load_dicom_image(input_path, pixel_spacing, pixel_spacing_source)
    elif input_path.lower().endswith('.jpg') or input_path.lower().endswith('.png'):
        return load_jpeg_image(input_path, pixel_spacing, pixel_spacing_source)
    return ValueError(f'unknown file type {input_path}')



class Cropper:
    def __init__(self, target_pixel_spacing, crop_size_in_pixels):
        self.target_pixel_spacing = target_pixel_spacing
        self.crop_size_in_pixels = crop_size_in_pixels

    def process(self, image_input, hip_detection, side):
        ########################
        ## RESAMPLING
        ########################
        # resample to the required resolution
        if self.target_pixel_spacing is not None:
            image, scale = image_input.resample(self.target_pixel_spacing)
            hip_detection = hip_detection.scale(scale)
        else:
            image = image_input

        ########################
        ## CROPPING / FLIPPING
        ########################
        # crop the hip centered on the femoral head
        offset_y = int(np.clip(hip_detection.center_y - self.crop_size_in_pixels // 2,
                               0, image.shape[0] - self.crop_size_in_pixels))
        offset_x = int(np.clip(hip_detection.center_x - self.crop_size_in_pixels // 2,
                               0, image.shape[1] - self.crop_size_in_pixels))
        image_cropped = image[
            offset_y:offset_y + self.crop_size_in_pixels,
            offset_x:offset_x + self.crop_size_in_pixels
        ]

        # check cropped size
        assert image_cropped.shape[0] == self.crop_size_in_pixels, \
            f'incorrect image size after cropping {image_cropped.shape}'
        assert image_cropped.shape[1] == self.crop_size_in_pixels, \
            f'incorrect image size after cropping {image_cropped.shape}'

        # flip left to right
        if side == 'left':
            image_cropped = image_cropped[:, ::-1]

        ########################
        ## NORMALIZATION
        ########################
        # normalize intensities
        image_cropped = image_cropped.astype(float)
        percentile = np.percentile(image_cropped.pixels.flatten(), [5, 95])
        intensity_offset = percentile[0]
        intensity_slope = percentile[1] - percentile[0]
        image_cropped = (image_cropped - intensity_offset) / intensity_slope

        return image_cropped, {
            'intensity_offset': float(intensity_offset),
            'intensity_slope': float(intensity_slope),
            'crop_offset_y': offset_y,
            'crop_offset_x': offset_x,
            'crop_offset_y_mm': float(offset_y * image_cropped.pixel_spacing[1]),
            'crop_offset_x_mm': float(offset_x * image_cropped.pixel_spacing[0]),
            'center_y': hip_detection.center_y,
            'center_x': hip_detection.center_x,
            'scale': scale,
            **hip_detection.stats,
        }


parser = argparse.ArgumentParser(add_help=False)
group = parser.add_argument_group('Image input')
group.add_argument('--input-image', metavar='DCM',
                   help='input image in DICOM or JPEG format')
group.add_argument('--input-pixel-spacing', metavar='SPACING', type=float,
                   help='pixel spacing of input (mm/pixel), if not given in DICOM headers')
group.add_argument('--pixel-spacing', metavar='SPACING', type=float,
                   default=0.2,
                   help='resample image to target spacing (mm/pixel)')
group.add_argument('--crop-size', metavar='PIXELS', type=int,
                   default=512,
                   help='crop the hips to the required size')
group.add_argument('--save-image', metavar='PNG',
                   help='save cropped image')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[parser, detect.parser])
    args = parser.parse_args()

    image_input = load_dicom_image(args.input_image)
    print(image_input)

    hip_detections = detect.detect_from_args(args, image_input)

    cropper = Cropper(args.pixel_spacing, args.crop_size)
    for side, hip_detection in hip_detections.items():
        image_cropped, stats = cropper.process(image_input, hip_detection, side)
        print(image_cropped.shape)
        print(side, hip_detection, stats)

        if args.save_image:
            util.save_grayscale_image(args.save_image.format(side=side), image_cropped.pixels)
