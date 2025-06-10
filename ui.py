import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Create main window
root = tk.Tk()
root.title("VöcaBot")
root.geometry("960x540")
root.configure(bg="white")



def load_icon(path, size=(40, 40)):
    img = Image.open(path)
    img = img.resize(size, Image.Resampling.LANCZOS)  # Use modern resampling
    return ImageTk.PhotoImage(img)

# Header with app name
header = tk.Label(root, text="VöcaBot", font=(
    "Helvetica", 28, "bold"), bg="white")
header.pack(pady=(20, 10))

# Main frame
main_frame = tk.Frame(root, bg="white")
main_frame.pack(fill="both", expand=True, padx=20, pady=10)

# Side buttons frame
side_frame = tk.Frame(main_frame, bg="white")
side_frame.pack(side="left", padx=20, fill="y")

# Icons (replace with your local image files or actual resource paths)
home_icon = load_icon("icons/home.png")
camera_icon = load_icon("icons/camera.png")
settings_icon = load_icon("icons/gear.png")


tk.Button(side_frame, image=home_icon, bd=0, bg="white").pack(pady=20)
tk.Button(side_frame, image=camera_icon, bd=0, bg="white").pack(pady=20)
tk.Button(side_frame, image=settings_icon, bd=0, bg="white").pack(pady=20)

# Camera view frame
camera_frame = tk.Frame(main_frame, bg="white", bd=2,
                        relief="solid", width=700, height=300)
camera_frame.pack(side="left", expand=True, fill="both")

# Draw corner markers
corner_size = 30
corner_thickness = 5


def draw_corner(canvas, x, y, position):
    if position == "tl":
        canvas.create_line(x, y, x + corner_size, y, width=corner_thickness)
        canvas.create_line(x, y, x, y + corner_size, width=corner_thickness)
    elif position == "tr":
        canvas.create_line(x, y, x - corner_size, y, width=corner_thickness)
        canvas.create_line(x, y, x, y + corner_size, width=corner_thickness)
    elif position == "bl":
        canvas.create_line(x, y, x + corner_size, y, width=corner_thickness)
        canvas.create_line(x, y, x, y - corner_size, width=corner_thickness)
    elif position == "br":
        canvas.create_line(x, y, x - corner_size, y, width=corner_thickness)
        canvas.create_line(x, y, x, y - corner_size, width=corner_thickness)


camera_canvas = tk.Canvas(camera_frame, bg="white", highlightthickness=0)
camera_canvas.pack(fill="both", expand=True)

# Draw corners when resized


def on_resize(event):
    camera_canvas.delete("all")
    w, h = event.width, event.height
    draw_corner(camera_canvas, 10, 10, "tl")
    draw_corner(camera_canvas, w - 10, 10, "tr")
    draw_corner(camera_canvas, 10, h - 10, "bl")
    draw_corner(camera_canvas, w - 10, h - 10, "br")


camera_canvas.bind("<Configure>", on_resize)

# Bottom note and buttons
bottom_frame = tk.Frame(root, bg="white")
bottom_frame.pack(pady=10)

note = tk.Label(bottom_frame, text="Note: This is where the translated text from the customer appear. "
                                   "Enable and disable button are for turning on and off the camera",
                font=("Helvetica", 10), wraplength=800, justify="left", bg="white")
note.pack(pady=5)

button_frame = tk.Frame(bottom_frame, bg="white")
button_frame.pack(pady=5)

enable_btn = ttk.Button(button_frame, text="Enable")
enable_btn.pack(side="left", padx=10)

disable_btn = ttk.Button(button_frame, text="Disable")
disable_btn.pack(side="left", padx=10)

root.mainloop()
