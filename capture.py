import cv2
import time
import os
import subprocess

# Captures png images (once every 2 seconds) and saves them for data
# collection

# Mac Dependencies
# - brew install python
# - pip install numpy
# - brew tap homebrew/science
# - brew install opencv
def main(data_path, name=None, video=cv2.VideoCapture(0)):
    stream = video
    DATA_PATH = os.getcwd() + "/dataset/"

    if not name:
        name = input("Name of person: ")

    offset = 0
    NAMED_PATH = DATA_PATH + name + "/"

    try:
        os.mkdir("dataset/" + name)
        print("Creating new directory...")
    except FileExistsError:
        print("Switching to existing directory...")
        print("Would you like to overwrite existing data? (y/n)")
        response = input()

        if response == "y":
            for file in os.listdir(NAMED_PATH):
                try:
                    os.remove(NAMED_PATH + str(file))
                except FileNotFoundError:
                    print("Couldn't find " + file)

        elif response == "n":
            offset = len(os.listdir(NAMED_PATH))

    img_max = 15

    for i in range(offset, img_max + offset + 1):
        print(f"Capturing image no. " + str(i))
        ret, frame = stream.read()
        filename = name + str(i) + ".png"
        # not sure why this isn't working
        # cv2.imshow(filename, frame)

        try:
            img_out = cv2.imwrite((NAMED_PATH + filename), frame)
        except:
            print(f"An error occurred when creating " + (name + str(i) + ".png"))

        time.sleep(2)
        # cv2.destroyWindow(filename)

    stream.release()
    subprocess.call([("open"), (NAMED_PATH)])

if __name__ == "__main__":
    main()
