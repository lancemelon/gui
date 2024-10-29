from ultralytics import YOLO
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageEnhance
from customtkinter import CTkImage
import os

model = YOLO("../best.pt")

ctk.set_default_color_theme('green')
ctk.set_appearance_mode('dark')

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Card Detector")
        self.geometry('800x800')

        # Widgets
        self.text = ctk.CTkTextbox(self, width=200, height=20, wrap=None)
        self.text.pack(pady=20, padx=20)

        # Button initialization
        self.button = ctk.CTkButton(self, text="Add File", command=self.openFile)
        self.button.pack(pady=20, padx=20)

        # Define self.image_label to hold the label for displaying images
        self.img_frame = ctk.CTkFrame(self)
        self.img_frame.pack()

        self.image_label = ctk.CTkLabel(self.img_frame, text="")

        self.res_img = ctk.CTkLabel(self.img_frame, text="")

        self.detect_button = ctk.CTkButton(self.img_frame, text='Detect', command=self.processImg)

    def openFile(self):
        file_path = filedialog.askopenfilename(
            initialdir=os.path.expanduser("~/Downloads"),
            title="Select File",
            filetypes=[("JPEG files", "*.jpeg")]
        )
        
        if file_path:
            self.text.delete("1.0", ctk.END)
            self.text.insert(ctk.END, file_path)

            file_image = Image.open(file_path)
            self.original_image = file_image  # Store the original image for reference

            # Resize image while maintaining aspect ratio
            bounding_width, bounding_height = 400, 500
            original_width, original_height = file_image.size
            aspect_ratio = original_width / original_height

            if bounding_width / bounding_height > aspect_ratio:
                new_width = int(bounding_height * aspect_ratio)
                new_height = bounding_height
            else:
                new_width = bounding_width
                new_height = int(bounding_width / aspect_ratio)

            # Resize the images
            self.original_image_resized = file_image.resize((new_width, new_height))
            self.darkened_image = self.create_darkened_image(self.original_image_resized)

            # Create CTkImages with the same size for consistency
            self.ctk_original = CTkImage(light_image=self.original_image_resized, dark_image=self.original_image_resized, size=(new_width, new_height))
            self.ctk_darkened = CTkImage(light_image=self.darkened_image, dark_image=self.darkened_image, size=(new_width, new_height))

            # Set the initial image and bind hover events
            self.image_label.configure(image=self.ctk_original)
            self.image_label.pack(side="left", anchor="w", padx=20, pady=20)
            self.image_label.bind("<Enter>", lambda e: self.imgHover(darken=True))
            self.img_frame.bind("<Leave>", lambda e: self.imgHover(darken=False))

    def create_darkened_image(self, image):
        enhancer = ImageEnhance.Brightness(image)
        darkened_image = enhancer.enhance(0.7)  # Darken by 30%
        return darkened_image

    def imgHover(self, darken):
        # Switch between original and darkened images without re-processing
        if darken:
            self.image_label.configure(image=self.ctk_darkened)
            self.detect_button.place(x=150, y=100)
        else:
            self.image_label.configure(image=self.ctk_original)
            self.detect_button.place_forget()

    def processImg(self):
        raw = self.text.get("1.0", ctk.END)
        actual = raw.strip()

        # Run prediction with bounding box drawing enabled
        result = model.predict(source=actual, conf=0.5, save=True, project="./guiResults")

        # Extract the image with bounding boxes
        image_with_boxes = result[0].plot()  # `plot()` typically returns a PIL Image with bounding boxes

        # Convert to PIL Image if necessary
        if not isinstance(image_with_boxes, Image.Image):
            image_with_boxes = Image.fromarray(image_with_boxes)

        # Create CTkImage
        result_ctk = CTkImage(light_image=image_with_boxes, dark_image=image_with_boxes, size=(400, 500))
        self.res_img.configure(image=result_ctk)
        self.res_img.pack(side="right", anchor="e", padx=20, pady=20)

    
app = App()
app.mainloop()
