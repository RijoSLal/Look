import os
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

# IMAGE_PATH = "static/unclear_image.jpeg"
# SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

# def preprocess_image(image_path):

#     """ 
#         Loads image from path and preprocesses to make it model ready
#         Args:
#             image_path: Path to the image file
#     """

#     hr_image = tf.image.decode_image(tf.io.read_file(image_path))
#     # If PNG, remove the alpha channel. The model only supports
#     # images with 3 color channels.
#     if hr_image.shape[-1] == 4:
#         hr_image = hr_image[...,:-1]

#     hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
#     hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
#     hr_image = tf.cast(hr_image, tf.float32)

#     return tf.expand_dims(hr_image, 0)



def preprocess_image(image_path):

    
    hr_image = tf.io.read_file(image_path)
    hr_image = tf.image.decode_image(hr_image, channels=3, expand_animations=False)

    # Ensure image dimensions are multiples of 4
    height, width = tf.shape(hr_image)[0], tf.shape(hr_image)[1]
    new_height, new_width = height - (height % 4), width - (width % 4)

    # Cropping instead of padding for efficiency
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, new_height, new_width)
    hr_image = tf.cast(hr_image, tf.float32)

    return tf.expand_dims(hr_image, 0)


def save_image(image, filename):

    """
        Saves unscaled Tensor Images.
        Args:
            image: 3D image tensor. [height, width, channels]
            filename: Name of the file to save.
    """

    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save(filename)

    print("Saved as %s" % filename)


def upscaling_function(IMAGE_PATH,SAVED_MODEL_PATH):
    """
      This function does all the processing on image.
      Args: 
          IMAGE_PATH : path where image should save on 
          SAVED_MODEL_PATH : model path
    
    """

    hr_image = preprocess_image(IMAGE_PATH)

    # save_image(tf.squeeze(hr_image), filename="Original_Image.jpg")

   
    
    model = hub.load(SAVED_MODEL_PATH)
    fake_image = model(hr_image)

    fake_image = tf.squeeze(fake_image)

    save_image(tf.squeeze(fake_image), filename="static/image/super_resolution.jpeg")

