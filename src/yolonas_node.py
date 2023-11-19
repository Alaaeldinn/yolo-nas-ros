#!/usr/bin/env python3

import cv2
import random

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose, Detection2DArray
from std_srvs.srv import SetBool

from super_gradients.training import models
from super_gradients.common.object_names import Models

class YoloNasNode:
    def __init__(self):
        rospy.init_node("yolonas_node")

        # params
        model = rospy.get_param("~model", Models.YOLO_NAS_S)
        pretrained_weights = rospy.get_param("~pretrained_weights", "coco")
        num_classes = rospy.get_param("~num_classes", -1)
        checkpoint_path = rospy.get_param("~checkpoint_path", "")
        device = rospy.get_param("~device", "cuda:0")
        threshold = rospy.get_param("~threshold", "0.5")
        self.enable = rospy.get_param("~enable", True)

        self.threshold = float(threshold)

        if num_classes < 0:
            num_classes = None

        if len(checkpoint_path) == 0:
            checkpoint_path = None

        self.class_to_color = {}
        self.cv_bridge = CvBridge()
        self.yolo = models.get(
            model,
            pretrained_weights=pretrained_weights,
            num_classes=num_classes,
            checkpoint_path=checkpoint_path
        )
        self.yolo.to(device)

        # topics
        self.pub = rospy.Publisher("detections", Detection2DArray, queue_size=10)
        self.dbg_pub = rospy.Publisher("dbg_image", Image, queue_size=10)
        self.sub = rospy.Subscriber("/camera/image_raw", Image, self.image_cb)

        # services
        self.srv = rospy.Service("enable", SetBool, self.enable_cb)

    def enable_cb(self, req):
        self.enable = req.data
        return SetBool.Response(success=True)

    def image_cb(self, msg):
        if self.enable:
            # convert image + predict
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            results = list(self.yolo.predict(cv_image)._images_prediction_lst)[0]

            # create detections msg
            detections_msg = Detection2DArray()
            detections_msg.header = msg.header

            for pred_i in range(len(results.prediction)):
                class_id = int(results.prediction.labels[pred_i])
                label = str(results.class_names[class_id])
                score = float(results.prediction.confidence[pred_i])

                if score < self.threshold:
                    continue

                x1 = int(results.prediction.bboxes_xyxy[pred_i, 0])
                y1 = int(results.prediction.bboxes_xyxy[pred_i, 1])
                x2 = int(results.prediction.bboxes_xyxy[pred_i, 2])
                y2 = int(results.prediction.bboxes_xyxy[pred_i, 3])

                # get boxes values
                x_s = float(x2 - x1)
                y_s = float(y2 - y1)
                x_c = x1 + x_s / 2
                y_c = y1 + y_s / 2

                detection = Detection2D()

                detection.bbox.center.x = x_c
                detection.bbox.center.y = y_c
                detection.bbox.size_x = x_s
                detection.bbox.size_y = y_s

                # get hypothesis
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.id = class_id
                hypothesis.score = score
                detection.results.append(hypothesis)

                # draw boxes for debug
                if label not in self.class_to_color:
                    r = random.randint(0, 255)
                    g = random.randint(0, 255)
                    b = random.randint(0, 255)
                    self.class_to_color[label] = (r, g, b)
                color = self.class_to_color[label]

                min_pt = (round(detection.bbox.center.x - detection.bbox.size_x / 2.0),
                          round(detection.bbox.center.y - detection.bbox.size_y / 2.0))
                max_pt = (round(detection.bbox.center.x + detection.bbox.size_x / 2.0),
                          round(detection.bbox.center.y + detection.bbox.size_y / 2.0))
                cv2.rectangle(cv_image, min_pt, max_pt, color, 2)

                label = "{} ({:.3f})".format(label, score)
                pos = (min_pt[0] + 5, min_pt[1] + 25)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(cv_image, label, pos, font,
                            1, color, 1, cv2.LINE_AA)

                # append msg
                detections_msg.detections.append(detection)

            # publish detections and dbg image
            self.pub.publish(detections_msg)
            self.dbg_pub.publish(self.cv_bridge.cv2_to_imgmsg(cv_image, "bgr8"))

if __name__ == "__main__":
    node = YoloNasNode()
    rospy.spin()

