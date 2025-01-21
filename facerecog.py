import os
import cv2
import face_recognition
import numpy as np
from tkinter import messagebox, filedialog
import customtkinter as ctk
from PIL import Image, ImageTk

# Initialize the GUI
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class FacialRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Recognition App")
        
        # Paths
        self.dataset_path = "./casia_faces/"
        self.database = {}
        
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        
        self.known_face_encodings = []
        self.known_face_names = []

        # Load existing database
        self.load_database()
        
        # UI Components
        self.setup_ui()
    def setup_ui(self):
        
        #Sets up the user interface
        # Display Frames
        self.image_frame = ctk.CTkLabel(self.root, text="Image Display", width=400, height=300)
        self.image_frame.grid(row=0, column=0, padx=10, pady=10)

        self.result_label = ctk.CTkLabel(self.root, text="Recognition Result", font=("Arial", 16))
        self.result_label.grid(row=1, column=0, pady=10)

        # Buttons
        self.acquire_button = ctk.CTkButton(self.root, text="Acquire Data", command=self.acquire_data)
        self.acquire_button.grid(row=2, column=0, pady=5)

        self.add_button = ctk.CTkButton(self.root, text="Add to Database", command=self.add_to_database)
        self.add_button.grid(row=3, column=0, pady=5)

        self.delete_button = ctk.CTkButton(self.root, text="Delete from Database", command=self.delete_from_database)
        self.delete_button.grid(row=4, column=0, pady=5)

    def acquire_data(self):
        #Opens a file dialog to select an image and performs recognition
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if file_path:
            try:
                self.display_image(file_path)
                image = cv2.imread(file_path)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                self.current_image = gray_image

                # Detect faces
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                if len(faces) == 0:
                    messagebox.showinfo("Info", "No faces detected in the image.")
                    return

                for (x, y, w, h) in faces:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Save and display the modified image
                detected_image_path = "detected_image.jpg"
                cv2.imwrite(detected_image_path, image)
                self.display_image(detected_image_path)

                # Perform recognition
                self.recognize_face(file_path)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image: {e}")

    def display_image(self, image_path):
        # Displays the acquired image in the GUI
        try:
            image = Image.open(image_path)  # Open the image using PIL
            resized_image = image.resize((400, 300))  # Resize to fit the label
            photo = ImageTk.PhotoImage(resized_image)  # Convert to a Tkinter-compatible image
            self.image_frame.configure(image=photo, text="")  # Update the label
            self.image_frame.image = photo  # Keep a reference to avoid garbage collection
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {e}")

    def add_to_database(self):
        # Adds the captured face to the database
        if hasattr(self, 'current_image'):
            name = ctk.CTkInputDialog(text="Enter name for the person:", title="Add to Database").get_input()
            if name:
                person_dir = os.path.join(self.dataset_path, name)
                if not os.path.exists(person_dir):
                    os.makedirs(person_dir)
                file_path = os.path.join(person_dir, f"{len(os.listdir(person_dir)) + 1}.jpg")
                cv2.imwrite(file_path, self.current_image)
                self.load_database()
                messagebox.showinfo("Info", f"Added {name} to the database.")
        else:
            messagebox.showerror("Error", "No image to add.")

    def delete_from_database(self):
        # Deletes a person from the database
        name = ctk.CTkInputDialog(text="Enter name to delete from database:", title="Delete from Database").get_input()
        if name:
            person_dir = os.path.join(self.dataset_path, name)
            if os.path.exists(person_dir):
                for file in os.listdir(person_dir):
                    os.remove(os.path.join(person_dir, file))
                os.rmdir(person_dir)
                self.load_database()
                messagebox.showinfo("Info", f"Deleted {name} from the database.")
            else:
                messagebox.showerror("Error", f"No data found for {name}.")

    def load_database(self):
        # Loads the images from the dataset and computes eigenfaces
        for person in os.listdir(self.dataset_path):
            person_path = os.path.join(self.dataset_path, person)
            if os.path.isdir(person_path):
                for image_name in os.listdir(person_path):
                    image_path = os.path.join(person_path, image_name)
                    print(image_path)
                    img = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(img)
                    if face_encodings:
                        img_face_encoding = face_encodings[0]

                    self.known_face_encodings.append(img_face_encoding)
                    self.known_face_names.append(person)

    def recognize_face(self, face_img):
        # Recognizes the face from the cropped image
        if self.known_face_names and self.known_face_encodings:
            img = face_recognition.load_image_file(face_img)
            face_locations = face_recognition.face_locations(img)
            face_encodings = face_recognition.face_encodings(img, face_locations)

            face_name = "Unknown"
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

                face_name = name

            if face_name != "Unknown":
                self.result_label.configure(text=f"Recognized: {face_name}")
                messagebox.showinfo("Recognition Result", f"Person is {face_name} from the database.")
            else:
                self.result_label.configure(text="No match found.")
                messagebox.showinfo("Recognition Result", "No match found in the database.")
        else:
            messagebox.showerror("Error", "No database loaded.")

if __name__ == "__main__":
    app = ctk.CTk()
    FacialRecognitionApp(app)
    app.mainloop()