import tkinter as tk
from PIL import Image, ImageTk

class JumpingImage:
    def __init__(self, root, image_path):
        self.root = root
        self.root.title("Jumping Image")
        
        # Load the image
        self.image = Image.open(image_path)
        self.photo = ImageTk.PhotoImage(self.image)
        
        # Create a label to display the image
        self.label = tk.Label(root, image=self.photo)
        self.label.pack()
        
        # Initial position
        self.x = 0
        self.direction = 1  # 1 for right, -1 for left
        
        # Start the animation
        self.animate()

    def animate(self):
        # Move the image
        self.x += self.direction * 5  # Change the speed here
        if self.x > self.root.winfo_width() - self.photo.width() or self.x < 0:
            self.direction *= -1  # Change direction on hitting the edge
        
        # Update the position of the label
        self.label.place(x=self.x, y=100)  # Change y for vertical position
        
        # Call this method again after 50 ms
        self.root.after(50, self.animate)

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x400")  # Set the window size
    app = JumpingImage(root, "260247.jpg")  # Update with your image path
    root.mainloop()