import numpy as np
from PIL import Image
from object_detection.utils import label_map_util
import matplotlib.pyplot as plt
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import time

scriptStartTime = time.time()
imageSourceDir = '../server/uploads'
imageConvertedDir = './convertedImages'
imageOutputDir = '../server/images'
logText = ''
tf.get_logger().setLevel('ERROR')
scoreList = []

def cal_average(num):
    sum_num = 0
    for t in num:
        sum_num = sum_num + t           
    avg = sum_num / len(num)
    return avg

def filterJPGs(fileList):
  filteredList = []
  for file in fileList:
    if (file[-4:].lower() == '.jpg'):
      filteredList.append(file)
  return filteredList
    

originalImagesArray = os.listdir(imageSourceDir)
filteredOriginalImagesArray = filterJPGs(originalImagesArray)

convertImagesStartTime = time.time()
for imageFile in filteredOriginalImagesArray:
  with Image.open(imageSourceDir + '/' + imageFile) as im:
    width, height = im.size
    im.thumbnail((800,800), Image.NEAREST)
    im.save(imageConvertedDir + '/' + imageFile)
os.system('rm ' + imageSourceDir + '/*')
logText = logText + 'Took {} seconds to convert images\n'.format(time.time() - convertImagesStartTime)

convertedImageArray = os.listdir(imageConvertedDir)
filteredConvertedImageArray = filterJPGs(convertedImageArray)
# filteredConvertedImageArray = imageArray[0:10] # if you just want to analyse 10 images from the directory for test purposes



# Download and extract the model 
# SSD MobileNet v2 320x320
# MODEL_DATE = '20200711'
# MODEL_NAME = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'

# CenterNet Resnet50 V2 512x512
# MODEL_DATE = '20200711'
# MODEL_NAME = 'centernet_resnet50_v2_512x512_coco17_tpu-8'

# SSD MobileNet V2 FPNLite 320x320
# MODEL_DATE = '20200711'
# MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'

def download_model(model_name, model_date):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_date + '/' + model_file,
                                        untar=True)
    return str(model_dir)

MODEL_DATE = '20200711'
MODEL_NAME = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'
PATH_TO_MODEL_DIR = download_model(MODEL_NAME, MODEL_DATE)

# Download labels file
def download_labels(filename):
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
    label_dir = tf.keras.utils.get_file(fname=filename,
                                        origin=base_url + filename,
                                        untar=False)
    label_dir = pathlib.Path(label_dir)
    return str(label_dir)

LABEL_FILENAME = 'mscoco_label_map.pbtxt'
PATH_TO_LABELS = download_labels(LABEL_FILENAME)
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

logText = logText + 'Used model: ' + MODEL_NAME + '\n'

modelLoadStartTime = time.time()
# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

logText = logText + 'Took {} seconds to load model\n'.format(time.time() - modelLoadStartTime)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

for imageFile in filteredConvertedImageArray:
    print('Running inference for {}... '.format(imageFile), end='')
    image_np = load_image_into_numpy_array(imageConvertedDir + '/' + imageFile)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)
    
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    classesList = detections['detection_classes'].tolist()
    scoresList = detections['detection_scores'].tolist()
    birdsDetected = False
    
    try:
      count = classesList.count(16) # looking for birds(No 16) in detection classes
    except Exception:
      count = 0

    if (count > 0): 
      indexes = []
      lastIndex = 0
      for y in range(count): # get Indexes of all bird detections
        newIndex = (classesList.index(16, lastIndex))
        indexes.append(newIndex)
        lastIndex = newIndex + 1
      
      for index in indexes:
        birdScore = scoresList[index]
        if birdScore > 0.4:
          scoreList.append(birdScore)
          birdsDetected = True

    if birdsDetected:
      print('Bird(s) detected')
      # savePath = './' + imageOutputDir + '/' + imageFile
      # pil_image=Image.fromarray(image_np_with_detections)
      # pil_image.save(savePath, 'JPEG')
      os.system('mv ' + imageConvertedDir + '/' + imageFile + ' ' + imageOutputDir + '/' + imageFile)
    else:
      print('No bird detected')
      os.system('rm ' + imageConvertedDir + '/' + imageFile)

scriptExecutionTime = (time.time() - scriptStartTime)
logText = logText + 'scanned ' + str(len(filteredConvertedImageArray)) + ' files\n' + \
'Found ' + str(len(scoreList)) + ' scores above 0.4, average detection score is ' \
+ str(cal_average(scoreList)) + '\n' + 'Execution time in seconds: ' + str(scriptExecutionTime) + '\n\n'
logFile = open('tensorflow.log', 'a')
logFile.write(logText)
logFile.close()