#!/bin/bash
# Script to load all 3 Python scripts for securiPy

cd /Users/jakebell/Documents/Independent\ Study/securiPy
echo "Do embeddings need reserialized (y/n)?"
read response

if ["$response" = "y"]; then
	echo "What is the confidence threshold? (0.0-1.0, default is 0.5)"
	read confidence
	python3 extract_embeddings.py --dataset dataset \
	--embeddings output/embeddings.pickle \
	--detector face_detection_model \
	--embedding-model openface_nn4.small2.v1.t7 \
	--confidence "$confidence"
fi

echo "Does the model need re-trained (y/n)?"
read response
if ["$response" = "y"]; then
	python3 train_model.py --embeddings output/embeddings.pickle \
	--recognizer output/recognizer.pickle \
	--le output/le.pickle
fi

echo "What is the file you would like to test against?"
read response
python3 recognize.py --detector face_detection_model \
	--embedding-model openface_nn4.small2.v1.t7 \
	--recognizer output/recognizer.pickle \
	--le output/le.pickle \
	--image images/$response".png"
