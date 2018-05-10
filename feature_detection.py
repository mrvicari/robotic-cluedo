#!/usr/bin/env python

import cv2
import numpy as np
import rospy

from collections import namedtuple, Counter
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Variables for FLANN based matcher (nearest neighbours algorithm)
# These are the suggested values to use on the opencv tutorials page
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6
flann_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6,
                   key_size = 12,
                   multi_probe_level = 1)

# A template has to have this many matches at least to be considered detected
MINIMUM_MATCHES = 10

# Tuple for storing an image and its calculated keypoints/descriptors after
# ORB.detectAndCompute()
Template = namedtuple('Template', 'image, name, keypoints, descriptors')

class FeatureDetection():

    def __init__(self, camera=False):
        # Initialise the tools we will use within the class
        self.bridge = CvBridge()
        self.orb = cv2.ORB(nfeatures = 1000)
        self.flann = cv2.FlannBasedMatcher(flann_params, {})

        # Load in the templates from file
        self.templates = []
        self.loadTemplates()

        # Decide if we want to use the camera feed directly or more efficient image
        if camera:
            self.subscriber = rospy.Subscriber('camera/rgb/image_raw', Image, self.callback)

    def loadTemplates(self):
        # Simply load all of the template files into the FLANN
        templates = {'Colonel Mustard' : 'mustard.png',
                     'Wrench' : 'wrench.png',
                     'Mrs Peacock' : 'peacock.png',
                     'Professor Plum' : 'plum.png',
                     'Revolver' : 'revolver.png',
                     'Rope' : 'rope.png',
                     'Miss Scarlett' : 'scarlett.png'}

        for name, filename in templates.iteritems():
            image = cv2.imread('./renamed_images/' + filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #image = cv2.GaussianBlur(image,(11,11),0)
            self.addTemplate(image.copy(), name)

    def detectKeypoints(self, image):
        # Calculate the keypoints, if no descriptors return empty array
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        if descriptors is None:
            descriptors = []
        return keypoints, descriptors

    def addTemplate(self, image, name):
        # Calculate keypoint and descriptors and pass them to flann algorithm
        keypoints, descriptors = self.detectKeypoints(image)
        descriptors = np.uint8(descriptors)
        self.flann.add([descriptors])

        # Convert the template to a target tuple and store for later
        template = Template(image=image, name=name, keypoints=keypoints, descriptors=descriptors)
        self.templates.append(template)

    def compareFeatures(self, image):
        # Treat the new image as another template, calculating keypoints
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detectKeypoints(image)

        # If the number of keypoints found is less than minimum matches, its
        # not possible for FLANN to find that many matches, so no point
        # continuing.
        if len(keypoints) < MINIMUM_MATCHES:
            return []

        # Use FLANN to find the matches between the input image and templates
        matches = self.flann.knnMatch(descriptors, k = 2)

        # Filter out any neighbours that are too far apart (Lowe's ratio test)
        matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.75]

        # Check we still have enough matches
        if len(matches) < MINIMUM_MATCHES:
            return []

        # We want to get the indexes of the templates so that they can be iterated
        # through in a sensible way
        template_matches = [[] for _ in xrange(len(self.templates))]
        for match in matches:
            # Add the matches that we found for the input image so they can be
            # compared
            template_matches[match.imgIdx].append(match)

        # Will store and return the names of the detections for best possible
        # guess and to avoid mistakes
        detections = []

        # Loop through the templates trying to match the matches
        for imgIdx, matches in enumerate(template_matches):
            if len(matches) < MINIMUM_MATCHES:
                # Can skip if matches is not high enough
                continue

            template = self.templates[imgIdx]

            # They use this in the opencv tutorials to find the homography
            # between the sets of keypoints. template_matches stores the
            # coordinates of the matches feature on the template. image_matches
            # stores the same match but the coordinates in the input image
            template_matches = [template.keypoints[m.trainIdx].pt for m in matches]
            image_matches = [keypoints[m.queryIdx].pt for m in matches]
            template_matches, image_matches = np.float32((template_matches, image_matches))

            # Now we have the matches, can calculate the homography
            homography, status = cv2.findHomography(template_matches, image_matches, cv2.RANSAC, 3.0)
            status = status.ravel() != 0

            # If the homography doesnt match, skip - otherwise we have a detection
            if status.sum() < MINIMUM_MATCHES:
                continue

            detections.append(template.name)
        return detections

    def identifyImage(self, image):
        # Method simply runs compareFeatures multiple times to ensure that a
        # false positive doesnt ruin our results - unlikely but better to be
        # safe than sorry

        detected = []
        for i in range(0, 2):
            detected = detected + self.compareFeatures(image)

        if detected:
            counter = Counter(detected)
            print("Found: " + counter.most_common(1)[0][0])
        else:
            print("Not able to detect an image.")

    def callback(self, data):
        # Ideally we wont use the camera feed directly, but rather the image
        # cropped out of the feed. But useful for testing.
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            print(e)

        self.identifyImage(cv_image)

        cv2.namedWindow('CameraFeed')
        cv2.imshow('CameraFeed', cv_image)
        cv2.waitKey(1)

# Create a node for the feature detection to run directly
def main():
    fD = FeatureDetection(camera=True)
    rospy.init_node('featureDetection', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("sleeping")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
