# import the necessary packages
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# if running script alone, accept command line arguments and pass them to main
if __name__ == "__main__":
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
		help="path to input image")
	ap.add_argument("-d", "--detector", required=True,
		help="path to OpenCV's deep learning face detector")
	ap.add_argument("-m", "--embedding-model", required=True,
		help="path to OpenCV's deep learning face embedding model")
	ap.add_argument("-r", "--recognizer", required=True,
		help="path to model trained to recognize faces")
	ap.add_argument("-l", "--le", required=True,
		help="path to label encoder")
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())

def main(image, detector, embedding_model, recognizer, label_encoder, confidence):
	# load our pre-trained serialized face detector from disk
	print("[INFO] loading face detector...")
	# protoPath = os.path.sep.join(detector, "deploy.prototxt")
	# modelPath = os.path.sep.join(detector,
	# 	"res10_300x300_ssd_iter_140000.caffemodel")
	protoPath = detector + "/" + "deploy.prototxt"
	modelPath = detector + "/" + "res10_300x300_ssd_iter_140000.caffemodel"
	dnn_detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

	# load our pre-trained serialized face embedding model from disk
	print("[INFO] loading face recognizer...")
	embedder = cv2.dnn.readNetFromTorch(embedding_model)

	# load the actual face recognition model along with the label encoder
	recognizer_model = pickle.loads(open(recognizer, "rb").read())
	le = pickle.loads(open(label_encoder, "rb").read())

	# load the image, resize it to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image dimensions
	image_obj = image
	image_obj = imutils.resize(image_obj, width=600)
	(h, w) = image_obj.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image_obj, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	dnn_detector.setInput(imageBlob)
	detections = dnn_detector.forward()

	# variable to hold the highest recognition confidence probability
	highest_prob = 0

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		pred_confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if pred_confidence > float(confidence):
			# compute the (x, y)-coordinates of the bounding box for the
			# face ROI
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI
			face = image_obj[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
				(0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# perform classification to recognize the face
			preds = recognizer_model.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]

			if proba > highest_prob:
				highest_prob = proba
				highest_face = Recognition(name, proba, startX, endX, startY, endY, image_obj)

	# show the output image and draw a box around the ROI with the highest prob
	# drawBox(highest_face)
	# cv2.imwrite("render.png", image_obj)
	# # cv2.imwshow("Image", image_obj)
	# # cv2.waitKey(0)

	return highest_face

# move to gui ??
# def drawBox(face):
# 	text = "{}: {:.2f}%".format(face.name, face.probability)
# 	y = face.y_cord[0] - 10 if face.y_cord[0] - 10 > 10 else face.y_cord[0] + 10
# 	cv2.rectangle(face.image, (face.x_cord[0], face.y_cord[0]), \
# 	(face.x_cord[1], face.y_cord[1]), (0, 0, 255), 2)
# 	cv2.putText(face.image, text, (face.x_cord[0], y),
# 		cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255), 2)

class Recognition:
	def __init__(self, name, probability, startX, endX, startY, endY, image):
		self.name = name
		self.probability = probability * 100
		self.x_cord = [startX, endX]
		self.y_cord = [startY, endY]
		self.image = image

	def name(self):
		return self.name

	def probability(self):
		return self.probability

	def x_cord(self):
		return self.x_cord

	def y_cord(self):
		return self.y_cord

	def image(self):
		return self.image
