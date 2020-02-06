# USAGE
# python train_model.py --embeddings output/embeddings.pickle \
#	--recognizer output/recognizer.pickle --le output/le.pickle

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

# if running script alone, accept command line arguments and pass them to main
if __name__ == "__main__":
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-e", "--embeddings", required=True,
		help="path to serialized db of facial embeddings")
	ap.add_argument("-r", "--recognizer", required=True,
		help="path to output model trained to recognize faces")
	ap.add_argument("-l", "--le", required=True,
		help="path to output label encoder")
	args = vars(ap.parse_args())
	main(args["embeddings"], args["recognizer"], args["le"])

# main method
def main(embeddings, recognizer, label_encoder):
	# load the .pickle face embeddings from extract_embeddings.py
	print("[INFO] loading face embeddings...")
	data = pickle.loads(open(embeddings, "rb").read())

	# encode the labels
	print("[INFO] encoding labels...")
	le = LabelEncoder()
	labels = le.fit_transform(data["names"])

	# train the model used to accept the 128-d embeddings of the face and
	# then produce the actual face recognition
	print("[INFO] training model...")
	recognizer_model = SVC(C=1.0, kernel="linear", probability=True)
	recognizer_model.fit(data["embeddings"], labels) # train the SVM model

	# write the actual face recognition model to disk
	f = open(recognizer, "wb")
	f.write(pickle.dumps(recognizer_model))
	f.close()

	# write the label encoder to disk
	f = open(label_encoder, "wb")
	f.write(pickle.dumps(le))
	f.close()
