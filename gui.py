# packages/libraries
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from PIL import Image, ImageTk
from queue import Queue
import cv2
import os
import pickle

# scripts
from extract_embeddings import main as extract
from train_model import main as train
from recognize import main as recognize
from recognize import Recognition

class MainApp(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        # directory of gui.py
        self.ROOT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/"
        self.parent = parent
        # confidence threshold for recognition
        self.rec_conf = 0.5
        # confidence threshold for detection
        self.emb_conf = 0.5
        # name of photo
        self.photo_path = "jacob02"
        # links to necessary paths for dependent scripts
        self.main_paths = {
            "dataset" : (self.ROOT_DIR + "dataset"),
            "embeddings" : (self.ROOT_DIR + "output/embeddings.pickle"),
            "detector" : (self.ROOT_DIR + "face_detection_model"),
            "embedding_model" : (self.ROOT_DIR + "openface_nn4.small2.v1.t7"),
            "recognizer" : (self.ROOT_DIR + "output/recognizer.pickle"),
            "label_encoder" : (self.ROOT_DIR + "output/le.pickle"),
            "image" : (self.ROOT_DIR + "/Images/" + self.photo_path + ".png"),
            "output_image" : (self.ROOT_DIR + "/render.png")
        }

        self.video_stream = cv2.VideoCapture(0)
        self.recognize_flag = False
        self.authorize = False

    # variable access methods
    def set_rec(self, new_val):
        try:
            if new_val <= 1.0 and new_val >= 0.0:
                self.rec_conf = new_val
            else:
                raise Exception("Value must be between 0.0 and 1.0.")
        except Exception as e:
            tk.messagebox.showerror("Error", e)

    def set_emb(self, new_val):
        self.emb_conf = new_val

    def get_rec(self):
        print(self.rec_conf)
        return self.rec_conf

    def get_emb(self):
        return self.emb_conf

    def set_main_path(self, key, value):
        self.main_paths[key] = value

    def get_main_path(self, key):
        return self.main_paths[key]

    def update_photo(self):
        self.photo_object = Image.open(self.ROOT_DIR + "Images/" + \
        self.photo_path + ".png")

    def set_photo_path(self, new_val):
        self.photo_path = new_val
        self.set_main_path("image", (self.ROOT_DIR + "/Images/" + self.photo_path))
        print(self.get_main_path("image"))
        self.temp_image = cv2.imread(self.get_main_path("image"))
        self.canvas_update(self.temp_image)

    def get_photo_path(self):
        return self.photo_path

    def get_photo(self):
        return self.photo_object

    def set_flag(self):
        self.recognize_flag = not self.recognize_flag

    # constructs the data preferences window
    def new_data_window(self):
        self.newWindow = tk.Toplevel(self.parent)
        self.data_win = DataWindow(self.newWindow, self)

    # runs re-serialization process
    def serialize_embeddings(self):
        if tk.messagebox.askyesno("Embeddings", \
        "Overwrite current embeddings database?"):
            status = extract(self.get_main_path("dataset"), \
            self.get_main_path("embeddings"), self.get_main_path("detector"), \
            self.get_main_path("embedding_model"), self.get_emb())

            tk.messagebox.showinfo(title="Embeddings", message=status)

    # re-trains the recognition model
    def retrain_model(self):
        if tk.messagebox.askyesno("Train", \
        "Re-train model based on current embeddings?"):
            train(self.get_main_path("embeddings"), \
            self.get_main_path("recognizer"), \
            self.get_main_path("label_encoder"))

            tk.messagebox.showinfo(title="Train", \
            message="Model trained successfully.")

    def recognize_face(self, frame):
        highest_face = recognize(frame, self.get_main_path("detector"), \
        self.get_main_path("embedding_model"), self.get_main_path("recognizer"), \
        self.get_main_path("label_encoder"), self.get_rec())
        self.draw_box(highest_face)
        self.canvas_update(highest_face.image)

    def convert_img(self, cv_image):
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGBA)
        cv_image = cv2.resize(cv_image, (600,600))
        temp_image = Image.fromarray(cv_image)
        return_image = ImageTk.PhotoImage(image=temp_image)
        return return_image

    def canvas_update(self):
        ret, frame = self.video_stream.read()
        self.init_image = frame

        if self.recognize_flag:
            self.highest_face = recognize(frame, self.get_main_path("detector"), \
            self.get_main_path("embedding_model"), self.get_main_path("recognizer"), \
            self.get_main_path("label_encoder"), self.get_rec())
            self.draw_box(self.highest_face)
            self.init_image = self.highest_face.image

        self.init_image = self.convert_img(self.init_image)
        self.video_label.imgtk = self.init_image
        self.video_label.configure(image=self.init_image)
        self.video_label.after(1, self.canvas_update)

    def draw_box(self, face):
    	text = "{}: {:.2f}%".format(face.name, face.probability)
    	y = face.y_cord[0] - 10 if face.y_cord[0] - 10 > 10 else face.y_cord[0] + 10
    	cv2.rectangle(face.image, (face.x_cord[0], face.y_cord[0]), \
    	(face.x_cord[1], face.y_cord[1]), (0, 0, 255), 2)
    	cv2.putText(face.image, text, (face.x_cord[0], y),
    		cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255), 2)

    # draw the ui
    def draw(self):
        # left-side frame for images/video
        self.video_frame = tk.Frame(self.parent, width=600, height=600)
        self.video_frame.grid(row=0, column=0, rowspan=15)
        self.video_label = tk.Label(self.video_frame)
        self.video_label.grid(row=0, column=0)
        self.canvas_update()

        # label text for variable section
        var_sec = tk.Label(self.parent, \
        text="Operational variables", underline=1, width=52, \
        bg="blue", fg="white")
        var_sec.grid(row=0, column=1, columnspan=2, sticky="sw")

        # label text for rec_conf value
        rec_conf_lbl = tk.Label(self.parent, \
        text="Recognition confidence threshold (0.0-1.0): ", \
        width=30, anchor="w")
        rec_conf_lbl.grid(row=1, column=1, rowspan=2)

        # entry box for setting value of rec_conf
        rec_ent_value = tk.StringVar()
        rec_ent_value.set(self.rec_conf)
        rec_conf_ent = tk.Entry(self.parent, textvariable=rec_ent_value)
        rec_conf_ent.grid(row=1, column=2, rowspan=2)

        # update button for updating value of rec_conf
        update_rec = tk.Button(self.parent, text="Update", width=21, pady=10, \
        command=lambda: self.set_rec(float(rec_conf_ent.get())))
        update_rec.grid(row=3, column=2, pady=5)

        # label text for emb_conf
        emb_conf_lbl = tk.Label(self.parent, \
        text="Embedding confidence threshold (0.0-1.0): ", width=30, anchor="w")
        emb_conf_lbl.grid(row=4, column=1, rowspan=2)

        # entry box for setting value of emb_conf
        emb_ent_value = tk.StringVar()
        emb_ent_value.set(self.emb_conf)
        emb_conf_ent = tk.Entry(self.parent, textvariable=emb_ent_value)
        emb_conf_ent.grid(row=4, column=2, rowspan=2)

        # update button for updating value of emb_conf
        update_emb = tk.Button(self.parent, text="Update", width=21, pady=10,  \
        command=lambda: self.set_emb(float(emb_conf_ent.get())))
        update_emb.grid(row=6, column=2, pady=5)

        # DEPRECATED (FOR STILL FRAMES): label text for photo entry
        # photo_lbl = tk.Label(self.parent, \
        # text="Name of file to test against: ", width=30, anchor="w")
        # photo_lbl.grid(row=7, column=1)

        # entry text for photo to test against
        # photo_ent_value = tk.StringVar()
        # photo_ent_value.set(self.photo_path + ".png")
        # photo_ent = tk.Entry(self.parent, textvariable=photo_ent_value)
        # photo_ent.grid(row=7, column=2)

        # update button for updating value of photo_path
        # update_photo = tk.Button(self.parent, text="Update", width=21, pady=10,  \
        # command=lambda: self.set_photo_path(photo_ent_value.get()))
        # update_photo.grid(row=8, column=2, pady=5)

        # label text for options section
        opt_sec = tk.Label(self.parent, \
        text="Options", underline=1, width=52, bg="blue", fg="white")
        opt_sec.grid(row=9, column=1, columnspan=2, sticky="sw")

        # reserialization button
        reser_btn = tk.Button(self.parent, text="Re-serialize embeddings...", \
        width=21, pady=10, command=lambda: self.serialize_embeddings())
        reser_btn.grid(row=10, column=1, padx=5, pady=5, sticky="w")

        # re-training button
        train_btn = tk.Button(self.parent, text="Re-train model...", \
        width=21, pady=10, command=lambda: self.retrain_model())
        train_btn.grid(row=10, column=2, padx=5, pady=5, sticky="w")

        # recognize button
        recog_btn = tk.Button(self.parent, text="Recognize", activebackground="red", \
        width=42, pady=10, command=lambda: self.set_flag())
        recog_btn.grid(row=11, column=1, pady=5, columnspan=2)

        # button to launch menu for data preferences
        data_btn = tk.Button(self.parent, text="Preferences...", width=42, \
        pady=10, command=self.new_data_window)
        data_btn.grid(row=12, column=1, pady=5, columnspan=2)

    # used by DataWindow to access root main_paths
    def update_embed_paths(self, child):
        self.child = child
        self.main_paths["dataset"] = self.child.data_path_text.get()
        self.main_paths["embeddings"] = self.child.main_path_text.get()
        self.main_paths["detector"] = self.child.det_path_text.get()
        self.main_paths["embedding_model"] = self.child.mod_path_text.get()


class DataWindow(MainApp):
    def __init__(self, parent, outside, *args, **kwargs):
        MainApp.__init__(self, parent, *args, **kwargs)
        self.outside = outside

        # dataset path
        data_path_lbl = tk.Label(self.parent, \
        text="Path to training dataset: ", width=30, anchor="w")
        data_path_lbl.grid(row=0, column=0, pady=5)
        self.data_path_text = tk.StringVar()
        self.data_path_text.set(outside.get_main_path("dataset"))
        data_path_ent = tk.Entry(self.parent, \
        textvariable=self.data_path_text, width=55)
        data_path_ent.grid(row=0, column=1, pady=5)

        # database path
        main_path_lbl = tk.Label(self.parent, \
        text="Path to embedding database: ", width=30, anchor="w")
        main_path_lbl.grid(row=1, column=0, pady=5)
        self.main_path_text = tk.StringVar()
        self.main_path_text.set(outside.get_main_path("embeddings"))
        main_path_ent = tk.Entry(self.parent, \
        textvariable=self.main_path_text, width=55)
        main_path_ent.grid(row=1, column=1, pady=5)

        # detector path
        det_path_lbl = tk.Label(self.parent, \
        text="Path to cv2 detector: ", width=30, anchor="w")
        det_path_lbl.grid(row=2, column=0, pady=5)
        self.det_path_text = tk.StringVar()
        self.det_path_text.set(outside.get_main_path("detector"))
        det_path_ent = tk.Entry(self.parent, \
        textvariable=self.det_path_text, width=55)
        det_path_ent.grid(row=2, column=1, pady=5)

        # model path
        mod_path_lbl = tk.Label(self.parent, \
        text="Path to learning model: ", width=30, anchor="w")
        mod_path_lbl.grid(row=3, column=0, pady=5)
        self.mod_path_text = tk.StringVar()
        self.mod_path_text.set(outside.get_main_path("embedding_model"))
        mod_path_ent = tk.Entry(self.parent, \
        textvariable=self.mod_path_text, width=55)
        mod_path_ent.grid(row=3, column=1, pady=5)

        # save/quit button
        quit_btn = tk.Button(self.parent, text="Save", width=30, \
        command=self.save).grid(row=4, column=1, pady=5, sticky="w")

    def save(self):
        print("Saving...")
        self.outside.update_embed_paths(self)
        self.parent.destroy()

def main():
    root = tk.Tk()
    root.title("securiPy")
    MainApp(root).draw()
    root.mainloop()

if __name__ == "__main__":
    main()
