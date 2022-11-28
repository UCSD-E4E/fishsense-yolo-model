import math
import os
# comment out below line to enable tensorflow outputs
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from detect_function import detect

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_list('images', './data/images/kite.jpg', 'path to input image')
flags.DEFINE_string('output', './detections/', 'path to output folder')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('count', False, 'count objects within images')
flags.DEFINE_boolean('dont_show', False, 'dont show image output')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('crop', False, 'crop detections from images')
flags.DEFINE_boolean('ocr', False, 'perform generic OCR on detection regions')
flags.DEFINE_boolean('plate', False, 'perform license plate recognition')
flags.DEFINE_string('dir', 'dummy', 'input directory')
def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    images = FLAGS.images

    # load model
    if FLAGS.framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
    else:
            saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    # for count, image_path in enumerate(images, 1):
    #     original_image = cv2.imread(image_path)
    #     original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    #
    #     image_data = cv2.resize(original_image, (input_size, input_size))
    #     image_data = image_data / 255.
    #
    #     # get image name by using split method
    #     image_name = image_path.split('/')[-1]
    #     image_name = image_name.split('.')[0]
    #     images_data = []
    #     for i in range(1):
    #         images_data.append(image_data)
    #     images_data = np.asarray(images_data).astype(np.float32)

    # # loop through images in list and run Yolov4 model on each
    if( FLAGS.dir == 'dummy'):
        for count, image_path in enumerate(images, 1):
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            image_data = cv2.resize(original_image, (input_size, input_size))
            image_data = image_data / 255.

            # get image name by using split method
            image_name = image_path.split('/')[-1]
            image_name = image_name.split('.')[0]
            images_data = []
            for i in range(1):
                images_data.append(image_data)
            images_data = np.asarray(images_data).astype(np.float32)
            if FLAGS.framework == 'tflite':
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                interpreter.set_tensor(input_details[0]['index'], images_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
            else:
                infer = saved_model_loaded.signatures['serving_default']
                batch_data = tf.constant(images_data)
                pred_bbox = infer(batch_data)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

            # run non max suppression on detections
            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=FLAGS.iou,
                score_threshold=FLAGS.score
            )

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
            original_h, original_w, _ = original_image.shape
            bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

            ##############################################################################################

            ##############################################################################################

            # hold all detection data in one variable
            pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

            # by default allow all classes in .names file
            allowed_classes = list(class_names.values())

            # custom allowed classes (uncomment line below to allow detections for only people)
            # allowed_classes = ['person']

            counted_things = (count_objects(pred_bbox, by_class=True, allowed_classes=allowed_classes))
            classesfound = []
            for i in range(len(class_names)):
                classesfound.append(class_names[i])
            fishes = []
            if ('fish' in counted_things):
                fishes_found = counted_things['fish']

                for i in range(fishes_found):
                    fishes.append([bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]])

            # if crop flag is enabled, crop each detection and save it as new image
            if FLAGS.crop:
                crop_path = os.path.join(os.getcwd(), 'detections', 'crop', image_name)
                try:
                    os.mkdir(crop_path)
                except FileExistsError:
                    pass
                crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox, crop_path, allowed_classes)

            # if count flag is enabled, perform counting of objects
            if FLAGS.count:
                # count objects found
                counted_classes = count_objects(pred_bbox, by_class=True, allowed_classes=allowed_classes)
                # loop through dict and print
                image = utils.draw_bbox(original_image, pred_bbox, FLAGS.info, counted_classes,
                                        allowed_classes=allowed_classes, show_label=False, read_plate=FLAGS.plate)
            else:
                image = utils.draw_bbox(original_image, pred_bbox, FLAGS.info, allowed_classes=allowed_classes,
                                        show_label=True, read_plate=FLAGS.plate)

            image = Image.fromarray(image.astype(np.uint8))

            image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

            # xmin, ymin, xmax, ymax

            # putting visual aids for center and bbox cords
            # for i in range(len(fishes)):
            #     cv2.circle(image,(int(round((fishes[i][0] +fishes[i][2])/2 )), int(round(fishes[i][3]+fishes[i][1])/2)), 5, [0,255,0],-1)
            #     cv2.circle(image,(int(round(fishes[i][0])), int(round(fishes[i][1]))), 5, [255,255,0],-1)
            #     cv2.circle(image,(int(round(fishes[i][2])), int(round(fishes[i][3]))), 5, [255,255,0],-1)

            mask = np.zeros(image.shape[:2], dtype="uint8")
            file = open(FLAGS.output + image_name + ".txt", "+w")
            # file2 = open(FLAGS.output + image_name+"centers.txt","+w")
            for i in range(len(fishes)):
                cv2.rectangle(mask, (fishes[i][0], fishes[i][1]), (fishes[i][2], fishes[i][3]), (255, 255, 255), -1)
                masked = cv2.bitwise_and(image, image, mask=mask)
                # cv2.imwrite(FLAGS.output + 'filtered' + image_name + '.png', masked)
                # class id
                # file.write("0 ")
                file.write(str(fishes[i][0]))
                file.write(" ")
                file.write(str(fishes[i][1]))
                file.write(" ")
                file.write(str(fishes[i][2]))
                file.write(" ")
                file.write(str(fishes[i][3]))
                file.write("\n")
                #
                # file2.write(str((fishes[i][0] +fishes[i][2])/2))
                # file2.write(" ")
                # file2.write(str((fishes[i][1] +fishes[i][3])/2))
                # file2.write("\n")

            file.close()
            # file2.close()

            if not FLAGS.dont_show:
                # resized = cv2.resize(image, (416,416), interpolation=cv2.INTER_AREA)
                # cv2.imshow("resize",resized)
                cv2.imshow("detection", image)
                cv2.waitKey(10)
                # image.show()

            cv2.imwrite(FLAGS.output + image_name + 'detection.png', image)
    else:
        for file in os.listdir(FLAGS.dir):
            original_image = cv2.imread(FLAGS.dir + file)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            image_data = cv2.resize(original_image, (input_size, input_size))
            image_data = image_data / 255.

            # get image name by using split method
            image_name = file.split('/')[-1]
            image_name = file.split('.')[0]

            images_data = []
            for i in range(1):
                images_data.append(image_data)
            images_data = np.asarray(images_data).astype(np.float32)
            if FLAGS.framework == 'tflite':
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                interpreter.set_tensor(input_details[0]['index'], images_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
            else:
                infer = saved_model_loaded.signatures['serving_default']
                batch_data = tf.constant(images_data)
                pred_bbox = infer(batch_data)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

            # run non max suppression on detections
            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=FLAGS.iou,
                score_threshold=FLAGS.score
            )

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
            original_h, original_w, _ = original_image.shape
            bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

            ##############################################################################################

            ##############################################################################################

            # hold all detection data in one variable
            pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

            # by default allow all classes in .names file
            allowed_classes = list(class_names.values())

            # custom allowed classes (uncomment line below to allow detections for only people)
            # allowed_classes = ['person']

            counted_things = (count_objects(pred_bbox, by_class=True, allowed_classes=allowed_classes))
            classesfound = []
            for i in range(len(class_names)):
                classesfound.append(class_names[i])
            fishes = []
            if ('fish' in counted_things):
                fishes_found = counted_things['fish']

                for i in range(fishes_found):
                    fishes.append([bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]])

            # if crop flag is enabled, crop each detection and save it as new image
            if FLAGS.crop:
                crop_path = os.path.join(os.getcwd(), 'detections', 'crop', image_name)
                try:
                    os.mkdir(crop_path)
                except FileExistsError:
                    pass
                crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox, crop_path, allowed_classes)

            # if count flag is enabled, perform counting of objects
            if FLAGS.count:
                # count objects found
                counted_classes = count_objects(pred_bbox, by_class=True, allowed_classes=allowed_classes)
                # loop through dict and print
                image = utils.draw_bbox(original_image, pred_bbox, FLAGS.info, counted_classes,
                                        allowed_classes=allowed_classes, show_label=False, read_plate=FLAGS.plate)
            else:
                image = utils.draw_bbox(original_image, pred_bbox, FLAGS.info, allowed_classes=allowed_classes,
                                        show_label=True, read_plate=FLAGS.plate)

            image = Image.fromarray(image.astype(np.uint8))

            image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

            # xmin, ymin, xmax, ymax

            # putting visual aids for center and bbox cords
            # for i in range(len(fishes)):
            #     cv2.circle(image,(int(round((fishes[i][0] +fishes[i][2])/2 )), int(round(fishes[i][3]+fishes[i][1])/2)), 5, [0,255,0],-1)
            #     cv2.circle(image,(int(round(fishes[i][0])), int(round(fishes[i][1]))), 5, [255,255,0],-1)
            #     cv2.circle(image,(int(round(fishes[i][2])), int(round(fishes[i][3]))), 5, [255,255,0],-1)

            mask = np.zeros(image.shape[:2], dtype="uint8")
            file = open(FLAGS.output + image_name + ".txt", "+w")
            # file2 = open(FLAGS.output + image_name+"centers.txt","+w")
            for i in range(len(fishes)):
                cv2.rectangle(mask, (fishes[i][0], fishes[i][1]), (fishes[i][2], fishes[i][3]), (255, 255, 255), -1)
                masked = cv2.bitwise_and(image, image, mask=mask)
                # cv2.imwrite(FLAGS.output + 'filtered' + image_name + '.png', masked)
                # class id
                # file.write("0 ")
                file.write(str(fishes[i][0]))
                file.write(" ")
                file.write(str(fishes[i][1]))
                file.write(" ")
                file.write(str(fishes[i][2]))
                file.write(" ")
                file.write(str(fishes[i][3]))
                file.write("\n")
                #
                # file2.write(str((fishes[i][0] +fishes[i][2])/2))
                # file2.write(" ")
                # file2.write(str((fishes[i][1] +fishes[i][3])/2))
                # file2.write("\n")

            file.close()
            # file2.close()

            if not FLAGS.dont_show:
                # resized = cv2.resize(image, (416,416), interpolation=cv2.INTER_AREA)
                # cv2.imshow("resize",resized)
                cv2.imshow("detection", image)
                cv2.waitKey(10)
                # image.show()

            cv2.imwrite(FLAGS.output + image_name + 'detection.png', image)



if __name__ == '__main__':
    list = detect(0.45,0.5,"./data/images/pacifico/")
    print(list)
    try:
        app.run(main)
    except SystemExit:
        pass
