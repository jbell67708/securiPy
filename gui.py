import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
# from extract_embeddings import main as extract
# from train_model import main as train
# from recognize import main as recognize
import os

class MainApp(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.rec_conf = 0.5
        self.emb_conf = 0.5
        self.emb_paths = {
            "dataset" : "dataset",
            "embeddings" : "output/embeddings.pickle",
            "detector" : "face_detection_model",
            "embedding_model" : "openface_nn4.small2.v1.t7"
        }

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
        return self.rec_conf

    def get_emb(self):
        return self.emb_conf

    def set_emb_path(key, value):
        self.emb_paths[key] = value

    def get_emb_path(self):
        return self.emb_paths

    def new_data_window(self):
        self.newWindow = tk.Toplevel(self.parent)
        self.data_win = DataWindow(self.newWindow, self)

    def draw(self):
        # label text for rec_conf value
        conf_lbl = tk.Label(self.parent, \
        text="Recognition confidence threshold (0.0-1.0): ")
        conf_lbl.grid(row=0, column=0)

        # entry box for setting value of rec_conf
        ent_value = tk.StringVar()
        ent_value.set(self.rec_conf)
        conf_ent = tk.Entry(self.parent, textvariable=ent_value)
        conf_ent.grid(row=0, column=1)

        # update button for updating value of rec_conf
        update_rec = tk.Button(self.parent, text="Update", \
        command=lambda: self.set_rec(float(conf_ent.get())))
        update_rec.grid(row=1, column=0)

        # button to launch menu for data preferences
        data_btn = tk.Button(self.parent, text="Preferences...", \
        command=self.new_data_window)
        data_btn.grid(row=1, column=1)

    # TODO:
    def update_embed_paths(self, child):
        print("TODO")

class DataWindow(MainApp):
    def __init__(self, parent, outside, *args, **kwargs):
        MainApp.__init__(self, parent, *args, **kwargs)
        self.outside = outside

        # dataset path
        data_path_lbl = tk.Label(self.parent, \
        text="Path to training dataset: ", width=30, anchor="w")
        data_path_lbl.grid(row=0, column=0)
        self.data_path_text = tk.StringVar()
        self.data_path_text.set(os.getcwd() + "/" + self.emb_paths["dataset"])
        data_path_ent = tk.Entry(self.parent, \
        textvariable=self.data_path_text, width=55)
        data_path_ent.grid(row=0, column=1)

        # database path
        emb_path_lbl = tk.Label(self.parent, \
        text="Path to embedding database: ", width=30, anchor="w")
        emb_path_lbl.grid(row=1, column=0)
        self.emb_path_text = tk.StringVar()
        self.emb_path_text.set(os.getcwd() + "/" + self.emb_paths["embeddings"])
        emb_path_ent = tk.Entry(self.parent, \
        textvariable=self.emb_path_text, width=55)
        emb_path_ent.grid(row=1, column=1)

        # detector path
        det_path_lbl = tk.Label(self.parent, \
        text="Path to cv2 detector: ", width=30, anchor="w")
        det_path_lbl.grid(row=2, column=0)
        self.det_path_text = tk.StringVar()
        self.det_path_text.set(os.getcwd() + "/" + self.emb_paths["detector"])
        det_path_ent = tk.Entry(self.parent, \
        textvariable=self.det_path_text, width=55)
        det_path_ent.grid(row=2, column=1)

        # model path
        mod_path_lbl = tk.Label(self.parent, \
        text="Path to learning model: ", width=30, anchor="w")
        mod_path_lbl.grid(row=3, column=0)
        self.mod_path_text = tk.StringVar()
        self.mod_path_text.set(os.getcwd() + "/" + self.emb_paths["embedding_model"])
        mod_path_ent = tk.Entry(self.parent, \
        textvariable=self.mod_path_text, width=55)
        mod_path_ent.grid(row=3, column=1)

        quit_btn = tk.Button(self.parent, text="Save", \
        command=self.save, padx=75).grid(row=4, column=1)

    def save(self):
        self.outside.emb_paths["dataset"] = self.data_path_text.get()
        self.outside.emb_paths["embeddings"] = self.emb_path_text.get()
        self.outside.emb_paths["detector"] = self.det_path_text.get()
        self.outside.emb_paths["embedding_model"] = self.mod_path_text.get()
        self.destroy()

def main():
    root = tk.Tk()
    root.title("securiPy")
    MainApp(root).draw()
    root.mainloop()

if __name__ == "__main__":
    main()
