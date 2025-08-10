import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

class HelmetDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Helmet Detection System")
        self.root.geometry("1100x700")
        self.root.configure(bg="#2c3e50")

        self.video_path = None
        self.cap = None
        self.running = False
        self.paused = False
        self.thread = None
        self.model = YOLO(os.path.join(os.getcwd(), "output/best.pt"))

        self.helmet_count = 0
        self.no_helmet_count = 0

        self.setup_gui()

    def setup_gui(self):
        # === Title ===
        title = tk.Label(self.root, text="🪖 Helmet Detection System", font=("Segoe UI", 22, "bold"),
                         bg="#2c3e50", fg="white")
        title.pack(pady=10)

        # === Main layout frames ===
        self.main_frame = tk.Frame(self.root, bg="#34495e")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.left_frame = tk.Frame(self.main_frame, bg="#2c3e50", bd=2, relief=tk.GROOVE)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        self.right_frame = tk.Frame(self.main_frame, bg="#2c3e50", bd=2, relief=tk.GROOVE)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        self.main_frame.columnconfigure(0, weight=3)
        self.main_frame.columnconfigure(1, weight=2)

        # === Left: Video display ===
        tk.Label(self.left_frame, text="Video Display", font=("Segoe UI", 14, "bold"),
                 bg="#2c3e50", fg="white").pack(pady=5)
        self.video_label = tk.Label(self.left_frame, bg="#1c1c2e")
        self.video_label.pack(expand=True, padx=10, pady=10)

        # === Right: Controls & Stats ===
        self.control_frame = tk.Frame(self.right_frame, bg="#2c3e50")
        self.control_frame.pack(pady=15)

        self.upload_btn = tk.Button(self.control_frame, text="📂 Upload Video", command=self.upload_video,
                                    font=("Segoe UI", 12), bg="#3498db", fg="white", width=20)
        self.upload_btn.pack(pady=5)

        # Add Live from Camera button
        self.camera_btn = tk.Button(self.control_frame, text="📷 Live from Camera", command=self.start_camera_detection,
                                    font=("Segoe UI", 12), bg="#f39c12", fg="white", width=20)
        self.camera_btn.pack(pady=5)

        self.start_btn = tk.Button(self.control_frame, text="▶ Play", command=self.start_detection,
                                   font=("Segoe UI", 12), bg="#27ae60", fg="white", width=20, state="disabled")
        self.start_btn.pack(pady=5)

        self.stop_btn = tk.Button(self.control_frame, text="⏹ Stop", command=self.stop_detection,
                                  font=("Segoe UI", 12), bg="#e74c3c", fg="white", width=20, state="disabled")
        self.stop_btn.pack(pady=5)

        tk.Label(self.control_frame, text="Progress:", bg="#2c3e50", fg="white").pack(pady=(20, 0))
        self.progress = Progressbar(self.control_frame, length=200, mode="determinate")
        self.progress.pack(pady=5)

        self.status_label = tk.Label(self.control_frame, text="Status: Waiting for video...",
                                     bg="#2c3e50", fg="#2ecc71", font=("Segoe UI", 10))
        self.status_label.pack(pady=10)

        # === Detection Stats ===
        tk.Label(self.right_frame, text="Detection Statistics", font=("Segoe UI", 14, "bold"),
                 bg="#2c3e50", fg="white").pack(pady=(20, 10))

        self.stats_helmet = tk.Label(self.right_frame, text="🟢 With Helmet: 0", font=("Segoe UI", 12),
                                     bg="#2c3e50", fg="green")
        self.stats_helmet.pack()

        self.stats_no_helmet = tk.Label(self.right_frame, text="❌ Without Helmet: 0", font=("Segoe UI", 12),
                                        bg="#2c3e50", fg="red")
        self.stats_no_helmet.pack()

        self.stats_total = tk.Label(self.right_frame, text="📊 Total Detections: 0", font=("Segoe UI", 12),
                                    bg="#2c3e50", fg="white")
        self.stats_total.pack()

    def upload_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")])
        if path:
            self.video_path = path
            self.status_label.config(text=f"Status: Video loaded - {os.path.basename(path)}")
            self.start_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            self.helmet_count = 0
            self.no_helmet_count = 0
            self.update_stats()

            # Show first frame immediately
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                frame_resized = cv2.resize(frame, (500, 400))
                img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                imgtk = ImageTk.PhotoImage(Image.fromarray(img))
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)

    def start_detection(self):
        if not self.video_path:
            return
        self.running = True
        self.progress["value"] = 0
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.camera_btn.config(state='disabled')
        self.thread = threading.Thread(target=self.process_video)
        self.thread.start()

    def start_camera_detection(self):
        self.video_path = None  # Clear video path
        self.running = True
        self.progress["value"] = 0
        self.start_btn.config(state='disabled')
        self.camera_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.helmet_count = 0
        self.no_helmet_count = 0
        self.update_stats()
        self.status_label.config(text="Status: Live camera detection running...")
        self.thread = threading.Thread(target=self.process_camera)
        self.thread.start()

    def stop_detection(self):
        self.running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.camera_btn.config(state='normal')

    def update_stats(self):
        self.stats_helmet.config(text=f"🟢 With Helmet: {self.helmet_count}")
        self.stats_no_helmet.config(text=f"❌ Without Helmet: {self.no_helmet_count}")
        total = self.helmet_count + self.no_helmet_count
        self.stats_total.config(text=f"📊 Total Detections: {total}")

    def process_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

        while self.cap.isOpened() and self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.model(frame)[0]
            for result in results.boxes.data.tolist():
                frame = self.draw_box(result, frame, names=results.names)

            # Resize and update frame
            frame_resized = cv2.resize(frame, (500, 400))
            img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(Image.fromarray(img))
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

            current_frame += 1
            progress_value = (current_frame / total_frames) * 100
            self.progress["value"] = progress_value
            self.root.update_idletasks()

        self.cap.release()
        self.status_label.config(text="Status: Detection finished.")
        self.stop_detection()

    def process_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.config(text="Status: Could not open camera.")
            self.stop_detection()
            return
        while self.cap.isOpened() and self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            results = self.model(frame)[0]
            for result in results.boxes.data.tolist():
                frame = self.draw_box(result, frame, names=results.names)
            frame_resized = cv2.resize(frame, (500, 400))
            img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(Image.fromarray(img))
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
            self.root.update_idletasks()
            # Add a small delay to avoid high CPU usage
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        self.status_label.config(text="Status: Live camera detection stopped.")
        self.stop_detection()

    def draw_box(self, params, frame, threshold=0.5, names=None):
        x1, y1, x2, y2, score, class_id = params
        if score > threshold:
            name = names[int(class_id)].lower()
            if "helmet" in name:
                color = (0, 255, 0)
                self.helmet_count += 1
            else:
                color = (0, 0, 255)
                self.no_helmet_count += 1

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 5)  # Thicker box
            label = names[int(class_id)].upper()
            cv2.putText(frame, label, (int(x1), int(y1 - 15)), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, color, 3)  # Larger font

        self.update_stats()
        return frame

        self.update_stats()

if __name__ == "__main__":
    root = tk.Tk()
    app = HelmetDetectionApp(root)
    root.mainloop()
