from extract_embeddings import main as extract
from train_model import main as train
from recognize import main as recognize

def main():
    response = input("Do embeddings need reserialized? (y/n) ")

    if response.lower() == "y":
        cValue = input("What is the confidence threshold? (0.0-1.0, default 0.5) ")
        extract("dataset", "output/embeddings.pickle", "face_detection_model", \
        "openface_nn4.small2.v1.t7", cValue)

    response = input("Does the model need to be re-trained? (y/n) ")

    if response.lower() == "y":
        train("output/embeddings.pickle", "output/recognizer.pickle", \
        "output/le.pickle")

    response = input("What is the file you would like to test against? ")
    cValue = input ("What is the confidence threshold for recognition? \
(0.0-1.0, default 0.5) ")
    recognize("images/" + response + ".png", "face_detection_model", \
    "openface_nn4.small2.v1.t7", "output/recognizer.pickle", \
    "output/le.pickle", cValue)

if __name__ == "__main__":
    main()
