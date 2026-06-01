import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import numpy as np
import json
import os
from tensorflow.keras.models import load_model

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class FlowerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Flower Recognition App")
        self.geometry("550x700")
        self.resizable(False, False)

        try:
            with open('keras_indices_to_folder.json', 'r') as f:
                self.keras_to_folder = {int(k): v for k, v in json.load(f).items()}
            with open('dataset/cat_to_name.json', 'r') as f:
                self.folder_to_name = json.load(f)
        except FileNotFoundError:
            print("Error: JSON mappings not found.")
            self.keras_to_folder, self.folder_to_name = {}, {}

        print("Loading AI Model...")
        try:
            self.model = load_model('Model.h5')
            print("Model loaded successfully!")
        except Exception as e:
            self.model = None
            print("Error loading Model.h5. Make sure it is in your project folder.")


        self.title_label = ctk.CTkLabel(self, text="AI Flower Predictor", font=ctk.CTkFont(size=28, weight="bold"))
        self.title_label.pack(pady=(30, 10))

        self.image_label = ctk.CTkLabel(self, text="No Image Selected", width=300, height=300,
                                        fg_color=("gray75", "gray25"), corner_radius=10)
        self.image_label.pack(pady=20)

        self.upload_btn = ctk.CTkButton(self, text="Select Image", font=ctk.CTkFont(size=16), height=40,
                                        command=self.upload_image)
        self.upload_btn.pack(pady=10)

        self.result_box = ctk.CTkTextbox(self, width=400, height=150, font=ctk.CTkFont(size=16))
        self.result_box.pack(pady=20)
        self.result_box.insert("0.0", "Waiting for image...\n")
        self.result_box.configure(state="disabled")  # Make it read-only

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])

        if file_path:
            pil_img = Image.open(file_path)

            display_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(300, 300))
            self.image_label.configure(image=display_img, text="")

            if self.model:
                self.predict(pil_img)
            else:
                self.update_results("Error: AI Model is not loaded.")

    def predict(self, pil_img):
        self.update_results("Analyzing image...")

        img = pil_img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = self.model.predict(img_array)[0]

        top_5_indices = np.argsort(predictions)[-5:][::-1]

        result_text = "--- Top 5 Predictions ---\n\n"
        for i, idx in enumerate(top_5_indices):
            folder_name = self.keras_to_folder.get(idx)
            flower_name = self.folder_to_name.get(folder_name, "Unknown").title()
            confidence = predictions[idx] * 100

            result_text += f"{i + 1}. {flower_name}: {confidence:.1f}%\n"

        self.update_results(result_text)

    def update_results(self, text):
        self.result_box.configure(state="normal")
        self.result_box.delete("0.0", "end")
        self.result_box.insert("0.0", text)
        self.result_box.configure(state="disabled")


if __name__ == "__main__":
    app = FlowerApp()
    app.mainloop()