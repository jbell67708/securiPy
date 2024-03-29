# USAGE
# python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle \
#	--detector face_detection_model --embedding-model openface_nn4.small2.v1.t7

# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils # image processing functions
import pickle
import cv2 # OpenCV
import os

# if running script alone, accept command line arguments and pass them to main
if __name__ == "__main__":
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--dataset", required=True,
		help="path to input directory of faces + images")
	ap.add_argument("-e", "--embeddings", required=True,
		help="path to output serialized db of facial embeddings")
	ap.add_argument("-d", "--detector", required=True,
		help="path to OpenCV's deep learning face detector")
	ap.add_argument("-m", "--embedding-model", required=True,
		help="path to OpenCV's deep learning face embedding model")
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")

	# list of command-line specified options (accessed via args[])
	args = vars(ap.parse_args())
	main(args["dataset"], args["embeddings"], args["detector"], \
	args["embedding_model"], args["confidence"])

# main method
def main(dataset, embeddings, detector, embedding_model, confidence):
	print("[INFO] loading face detector...")
	# determine path to prototxt and model files
	# protoPath = os.path.sep.join(detector, "deploy.prototxt")
	# modelPath = os.path.sep.join(detector,
	# 	"res10_300x300_ssd_iter_140000.caffemodel")

	protoPath = detector + "/" + "deploy.prototxt"
	modelPath = detector + "/" + "res10_300x300_ssd_iter_140000.caffemodel"

	# initialize a deep learning face detector for localization of faces
	detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

	print("[INFO] loading face recognizer...")
	# initalize a Torch-based extraction recognizer
	embedder = cv2.dnn.readNetFromTorch(embedding_model)

	print("[INFO] quantifying faces...")
	# list of image paths specified via command-line (-i) or parameters
	imagePaths = list(paths.list_images(dataset))

	# initialize our lists of extracted facial embeddings and
	# corresponding people names
	knownEmbeddings = []
	knownNames = []
	rejections = []

	# initialize the total number of faces processed
	total = 0

	# loop over the image paths
	for (i, imagePath) in enumerate(imagePaths):
		print(f"[INFO] processing image {i+1}/{len(imagePaths)}")

		# due to naming structure, the name of the person can be determined
		name = imagePath.split(os.path.sep)[-2]

		# load the image, resize it to have a width of 600 pixels (while
		# maintaining the aspect ratio), and then grab the image
		# dimensions
		image = cv2.imread(imagePath)
		image = imutils.resize(image, width=600)
		(h, w) = image.shape[:2]

		# perform a Gaussian_Blur on the image for processing
		image = cv2.GaussianBlur(image, (5,5), 0)

		# construct a blob from the image
		# blobFromImage(image,scalefactor,size,mean RGB,swap RGB channels)
		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(image, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)


		# apply OpenCV's deep learning-based face detector to localize
		# faces in the input image
		detector.setInput(imageBlob)
		detections = detector.forward()

		# ensure at least one face was found
		if len(detections) > 0:
			# we're making the assumption that each image has only ONE
			# face, so find the bounding box with the largest probability
			i = np.argmax(detections[0, 0, :, 2])
			box_confidence = detections[0, 0, i, 2]

			# ensure that the detection with the largest probability also
			# means our minimum probability test (thus helping filter out
			# weak detections)
			if box_confidence > float(confidence):
				# compute the (x, y)-coordinates of the bounding box for
				# the face
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# extract the face ROI and grab the ROI dimensions
				face = image[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

				# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					continue

				# construct a blob for the face ROI, then pass the blob
				# through our face embedding model to obtain the 128-d
				# quantification of the face
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					(96, 96), (0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()

				# add the name of the person + corresponding face
				# embedding to their respective lists
				knownNames.append(name)
				knownEmbeddings.append(vec.flatten())
				total += 1

		exitMsg = f"""
Successfully detected and serialized {total}/{len(imagePaths)} encodings.
"""

	# dump the facial embeddings + names to disk
	print(exitMsg)
	data = {"embeddings": knownEmbeddings, "names": knownNames}
	f = open(embeddings, "wb")
	f.write(pickle.dumps(data))
	f.close()
	return exitMsg
