from PIL import Image, ImageEnhance

def random_flip(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def random_rotation(image, max_angle=20):
    angle = np.random.uniform(-max_angle, max_angle)
    return image.rotate(angle)

def random_brightness(image, factor_range=(0.5, 1.5)):
    factor = np.random.uniform(*factor_range)
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def augment_image(image):
    augmentations = [random_flip, random_rotation, random_brightness]
    for aug in augmentations:
        if np.random.rand() > 0.5:
            image = aug(image)
    return image