import roslib
import rospy
import tf
import time
import numpy as np
import os

from go_to import *

from collections import Counter
from feature_detection import FeatureDetection
from geometry_msgs.msg import Twist
from nav_msgs.msg import MapMetaData, OccupancyGrid
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os

class RoboticCluedo():
    def __init__(self):
        self.arPosList = []

        self.listener = tf.TransformListener()

        self.pub = rospy.Publisher('mobile_base/commands/velocity', Twist)
        self.move = Twist()
        self.rate = rospy.Rate(10)

        self.fD = FeatureDetection(camera=False)
        self.detected = []

        self.navigator = GoToPose()

        self.take_picture = False
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('camera/rgb/image_raw', Image, self.img_callback)

    '''
        Calculate map position and angle for robot to be directly facing the AR marker
    '''
    def closeUpPosterGoTo(self, ar_distance, tol):
        time.sleep(1)
        try:
            # Check if AR marker has been seen before
            mapArPos, rot = self.listener.lookupTransform('/map', '/ar_marker_0', rospy.Time(0))
            for posCoordinate in self.arPosList:
                if mapArPos[0]-tol <= posCoordinate[0] <= mapArPos[0]+tol and mapArPos[1]-tol <= posCoordinate[1] <= mapArPos[1]+tol:
                    print("Already seen!")
                    return

            print("Found AR Marker at " + str(mapArPos[:2]))

            # Calculate 4x4 rotation matrix
            rotationMatrix = self.listener.fromTranslationRotation(mapArPos, rot)

            # Get the z vector (x, y) from rotation matrix
            arZvector = np.array([rotationMatrix[0][2], rotationMatrix[1][2]])

            # Calculate position at certain distance in front of AR marker
            newPosition = mapArPos[:2] + arZvector * ar_distance

            # Invert vector and find angle to point at AR marker
            arZvectorInv = arZvector * (-1)
            theta = np.arctan2(arZvectorInv[1], arZvectorInv[0])

            # Go to goal position
            position = {'x': newPosition[0], 'y' : newPosition[1]}
            quaternion = {'r1' : 0.000, 'r2' : 0.000, 'r3' : np.sin(theta/2.0), 'r4' : np.cos(theta/2.0)}

            if self.navigator.goto(position, quaternion):
                print("Success")
                # Store AR marker position on list
                self.arPosList.append(mapArPos[:2])

                print('Centering...')
                while True:
                    newPos = self.listener.lookupTransform('/base_link', '/ar_marker_0', rospy.Time(0))[0]
                    if newPos[1] > 0.01:
                        self.move.angular.z = 0.1
                    elif newPos[1] < -0.01:
                        self.move.angular.z = -0.1
                    else:
                        self.move.angular.z = 0
                        break
                    self.pub.publish(self.move)
                    self.rate.sleep()

                self.take_picture = True
            else:
                print("Fail")
                self.goToCenter(center)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("No AR Marker")

    '''
        Find and draw countours around poster
    '''
    def drawBoundaries(self):
        openCv_image = cv2.imread('detections/image' + str(len(self.arPosList))+'.jpeg')
        #resize image
        img= openCv_image

        #convert image to gray from bgr
        img_to_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #smooting image with GaussianBlur so that after that edge detection can be performed
        img_to_gray = cv2.GaussianBlur(img_to_gray,(11,11),0)

        #apply edge detection, minVal is 100 and maxVal is 200, every edge with color intensity gradient higher than 200 are sure to be edges and if lower than 100sure to be non edges
        edge = cv2.Canny(img_to_gray,100,200)

        #cv2.CHAIN_APPROX_SIMPLE used to remove redundant point and compress the contour ~(need only th edge point), saves memeory
        contours, hierarchy = cv2.findContours(edge.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        #draw contours on the original not gray image, pass the contours list, (-1) - to draw all contours, colour of contour [0,255,0] - lime colour, and 2 to indicate thickness of contour
        cv2.drawContours(img,contours,-1,[0,255,0],2)

        # Save new image with drawn contours
        cv2.imwrite('detections/image'+ str(len(self.arPosList))+'.jpeg', img)

    '''
        Take picure, draw boundaries and identify character
    '''
    def img_callback(self, data):
        if self.take_picture:
            print("Taking Pictue")
            self.take_picture = False
            try:
                # Convert your ROS Image message to OpenCV2
                cv2_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
                cv2.imwrite('detections/image' + str(len(self.arPosList))+'.jpeg', cv2_img)

                # Run the feature detection and find the best match
                results =  self.fD.compareFeatures(cv2_img)
                if results:
                    counter = Counter(results)
                    print("Found: " + counter.most_common(1)[0][0])
                    self.detected.append(counter.most_common(1)[0][0])
                else:
                    self.detected.append("Unknown")
                    print("Not able to detect an image.")

                self.drawBoundaries()
            except CvBridgeError, e:
                print(e)
    '''
        Spin around looking for AR markers
    '''
    def spin(self):
        self.move.angular.z = 2.0
        for i in range(10):
            self.pub.publish(self.move)
            self.rate.sleep()
            self.closeUpPosterGoTo(0.35, 0.4)
            self.move.angular.z = 2.0
            time.sleep(0.5)
        self.move.angular.z = 0

    def goToCenter(self, coordinates):
        position = {'x': coordinates[0], 'y' : coordinates[1]}
        quaternion = {'r1' : 0.000, 'r2' : 0.000, 'r3' : np.sin(coordinates[2]/2.0), 'r4' : np.cos(coordinates[2]/2.0)}
        rospy.loginfo("Go to center: (%s, %s)", position['x'], position['y'])
        self.navigator.goto(position, quaternion)

if __name__ == '__main__':
    rospy.init_node('robo_clue', anonymous=True)
    rc = RoboticCluedo()

    center = [0.4, -1.2, 0]
    rc.goToCenter(center)
    rc.spin()

    if not os.path.exists('detections'):
        os.makedirs('detections')

    dist = 0.5
    count = 0

    prev_point = center
    min_distance = 0.4

    start_time = time.time()

    try:
        # Don't stop until 2 objects are found
        while len(rc.arPosList) < 2:
            # Only move to a new point if is at least min_distance away
            while True:
                x = np.random.uniform(center[0] - dist, center[0] + dist, 1)
                y = np.random.uniform(center[1] - dist, center[1] + dist, 1)
                theta = np.random.uniform(-3.14, 3.14, 1)

                if np.sqrt((x - prev_point[0])**2 + (y - prev_point[1])**2) > min_distance:
                    break

            prev_point = [x, y]

            position = {'x': x, 'y' : y}
            quaternion = {'r1' : 0.000, 'r2' : 0.000, 'r3' : np.sin(theta/2.0), 'r4' : np.cos(theta/2.0)}
            rospy.loginfo("Go to (%s, %s) pose", position['x'], position['y'])
            if rc.navigator.goto(position, quaternion):
                count += 1
                rc.spin()
            if count > 5:
                dist += 0.5
                min_distance += 0.2
                count = 0

            # Stop execution after 20 minutes
            if time.time() - start_time > 20*60:
                print("Timeout")
                break

        print(rc.arPosList)
        print(rc.detected)

        f = open('detections/coordinates.txt','w')
        f.write('Found ' + str(rc.detected[0]) + ' at position ' + str(rc.arPosList[0]) + '\n')
        os.rename('detections/image1.jpeg', 'detections/' + str(rc.detected[0]) + '.jpeg')

        f.write('Found ' + str(rc.detected[1]) + ' at position ' + str(rc.arPosList[1]) + '\n')
        os.rename('detections/image2.jpeg', 'detections/' + str(rc.detected[1]) + '.jpeg')

    except KeyboardInterrupt:
		print("Shutting down")
