import imageio

def save_grayscale_image(filename, image):
    image = image.astype(float) - image.min()
    image *= 255 / image.max()
    imageio.imsave(filename, image.astype('uint8'))

