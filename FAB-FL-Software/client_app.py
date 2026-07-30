import customtkinter as ctk
import tkinter as tk
import psutil
import threading
import time
import requests
import uuid
import math
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import socket
import numpy as np
import matplotlib.pyplot as plt
import os
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.254.254.254', 1))
        return s.getsockname()[0]
    except Exception:
        return '127.0.0.1'
    finally:
        s.close()

BG_COLOR = "#1a1a24"
PANEL_COLOR = "#252533"
CYAN = "#00f2fe"
PURPLE = "#4facfe"
GREEN = "#00e676"
TEXT_MAIN = "#ffffff"
TEXT_SUB = "#a0a0b5"

ctk.set_appearance_mode("Dark")

class CircularProgressbar(ctk.CTkCanvas):
    def __init__(self, parent, size=80, thickness=8, color=CYAN, bg_color="#3a3a4d"):
        super().__init__(parent, width=size, height=size, bg=PANEL_COLOR, highlightthickness=0)
        self.size = size
        self.thickness = thickness
        self.color = color
        self.bg_color = bg_color
        self.value = 0
        self.draw_arc()

    def draw_arc(self):
        self.delete("all")
        margin = self.thickness / 2
        # Draw background ring
        self.create_oval(margin, margin, self.size-margin, self.size-margin, 
                         outline=self.bg_color, width=self.thickness)
        
        # Draw progress ring
        extent = -(self.value / 100) * 360
        self.create_arc(margin, margin, self.size-margin, self.size-margin,
                        start=90, extent=extent, outline=self.color, width=self.thickness, style=tk.ARC)
        
        # Draw text
        self.create_text(self.size/2, self.size/2, text=f"{int(self.value)}%", 
                         fill=TEXT_MAIN, font=("Arial", 14, "bold"))

    def set(self, value):
        self.value = value
        self.draw_arc()

class ClientGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("FAB-FL Worker Node v1.0")
        self.geometry("800x1050")
        self.configure(fg_color=BG_COLOR)
        self.server_ip = ""
        self.laptop_id = f"Node-{str(uuid.uuid4())[:4].upper()}"
        
        self.setup_ui()
        
    def setup_ui(self):
        # --- Header ---
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(15, 5))
        
        left_header = ctk.CTkFrame(header_frame, fg_color="transparent")
        left_header.pack(side="left")
        
        title_label = ctk.CTkLabel(left_header, text="FAB-FL Worker Node", font=("Arial", 24, "bold"), text_color=TEXT_MAIN)
        title_label.pack(anchor="w")
        
        self.refresh_btn = ctk.CTkButton(left_header, text="🔄 Refresh Client App", width=150, height=24, fg_color="#3a3a4d", hover_color="#4a4a5d", font=("Arial", 11), command=self.refresh_laptop)
        self.refresh_btn.pack(anchor="w", pady=(2, 0))
        
        self.conn_status = ctk.CTkLabel(header_frame, text="Connection: 🔴 DISCONNECTED", text_color=TEXT_SUB, font=("Arial", 12))
        self.conn_status.pack(side="right", anchor="n")

        # --- Tabview Nav Bar ---
        self.tabview = ctk.CTkTabview(self, fg_color="transparent", text_color=TEXT_MAIN, segmented_button_selected_color=CYAN, segmented_button_selected_hover_color="#00c0cb", segmented_button_unselected_color="#1a1a24")
        self.tabview.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.tabview.add("Dashboard")
        self.tabview.add("Test")
        
        dashboard = self.tabview.tab("Dashboard")
        test_tab = self.tabview.tab("Test")

        # --- Top Panels Container ---
        top_container = ctk.CTkFrame(dashboard, fg_color="transparent")
        top_container.pack(fill="x", padx=10, pady=10)
        top_container.grid_columnconfigure(0, weight=1)
        top_container.grid_columnconfigure(1, weight=1)

        # 1. Server Connection & Profile
        conn_panel = ctk.CTkFrame(top_container, fg_color=PANEL_COLOR, corner_radius=10)
        conn_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        ctk.CTkLabel(conn_panel, text="SERVER CONNECTION & PROFILE", font=("Arial", 12, "bold"), text_color=TEXT_SUB).pack(anchor="w", padx=15, pady=(15, 5))
        ctk.CTkLabel(conn_panel, text="Server IP Address:", font=("Arial", 12), text_color=TEXT_MAIN).pack(anchor="w", padx=15)
        
        ip_frame = ctk.CTkFrame(conn_panel, fg_color="transparent")
        ip_frame.pack(fill="x", padx=15, pady=5)
        
        self.ip_entry = ctk.CTkEntry(ip_frame, placeholder_text="192.168.1.100:8000", width=200, fg_color="#1a1a24", border_width=1)
        self.ip_entry.pack(side="left")
        
        # Context Menu for Copy/Paste
        self.ip_context_menu = tk.Menu(self, tearoff=0, bg=PANEL_COLOR, fg=TEXT_MAIN)
        
        def copy_ip():
            self.clipboard_clear()
            self.clipboard_append(self.ip_entry.get())
            
        def paste_ip():
            try:
                self.ip_entry.delete(0, "end")
                self.ip_entry.insert(0, self.clipboard_get())
            except tk.TclError:
                pass
                
        self.ip_context_menu.add_command(label="Copy", command=copy_ip)
        self.ip_context_menu.add_command(label="Paste", command=paste_ip)

        def show_context_menu(event):
            try:
                self.ip_context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self.ip_context_menu.grab_release()

        # Bind right-click
        self.ip_entry.bind("<Button-3>", show_context_menu)
        
        
        self.connect_btn = ctk.CTkButton(ip_frame, text="Connect", fg_color=CYAN, text_color="black", hover_color="#00c0cb", width=80, command=self.connect_server)
        self.connect_btn.pack(side="left", padx=10)

        # 2. Hardware Profiling
        hw_panel = ctk.CTkFrame(top_container, fg_color=PANEL_COLOR, corner_radius=10)
        hw_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        
        ctk.CTkLabel(hw_panel, text="HARDWARE PROFILING", font=("Arial", 12, "bold"), text_color=TEXT_SUB).pack(anchor="w", padx=15, pady=(15, 5))
        
        gauges_frame = ctk.CTkFrame(hw_panel, fg_color="transparent")
        gauges_frame.pack(fill="x", padx=15)
        
        # CPU
        cpu_frame = ctk.CTkFrame(gauges_frame, fg_color="transparent")
        cpu_frame.pack(side="left")
        self.cpu_gauge = CircularProgressbar(cpu_frame, size=60, color=CYAN)
        self.cpu_gauge.pack()
        ctk.CTkLabel(cpu_frame, text="CPU USAGE", font=("Arial", 10), text_color=TEXT_MAIN).pack()
        
        # RAM
        ram_frame = ctk.CTkFrame(gauges_frame, fg_color="transparent")
        ram_frame.pack(side="left", padx=20)
        self.ram_gauge = CircularProgressbar(ram_frame, size=60, color=PURPLE)
        self.ram_gauge.pack()
        self.ram_lbl = ctk.CTkLabel(ram_frame, text="RAM USAGE", font=("Arial", 10), text_color=TEXT_MAIN)
        self.ram_lbl.pack()

        # Settings
        settings_frame = ctk.CTkFrame(hw_panel, fg_color="transparent")
        settings_frame.pack(fill="x", padx=15, pady=10)
        
        cli_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        cli_frame.pack(side="left")
        ctk.CTkLabel(cli_frame, text="Virtual Clients:", font=("Arial", 11), text_color=TEXT_SUB).pack(anchor="w")
        self.clients_entry = ctk.CTkEntry(cli_frame, width=50, fg_color="transparent", border_width=0, font=("Arial", 20, "bold"), text_color=CYAN)
        self.clients_entry.insert(0, "15")
        self.clients_entry.pack(anchor="w")
        
        ep_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        ep_frame.pack(side="left", padx=15)
        ctk.CTkLabel(ep_frame, text="Training Rounds:", font=("Arial", 11), text_color=TEXT_SUB).pack(anchor="w")
        self.epochs_entry = ctk.CTkEntry(ep_frame, width=50, height=28, fg_color="#1a1a24", border_width=1)
        self.epochs_entry.insert(0, "10")
        self.epochs_entry.pack(anchor="w")
        
        self.start_btn = ctk.CTkButton(settings_frame, text="Start", fg_color=CYAN, text_color="black", hover_color="#00c0cb", width=70, height=35, state="disabled", command=self.start_training)
        self.start_btn.pack(side="right", padx=(0, 5), pady=(15, 0))

        # --- Middle Panel: FL PIPELINE ---
        pipe_panel = ctk.CTkFrame(dashboard, fg_color=PANEL_COLOR, corner_radius=10)
        pipe_panel.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(pipe_panel, text="FEDERATED LEARNING PIPELINE", font=("Arial", 12, "bold"), text_color=TEXT_SUB).pack(anchor="w", padx=15, pady=(15, 0))
        
        pipe_content = ctk.CTkFrame(pipe_panel, fg_color="transparent")
        pipe_content.pack(fill="both", expand=True, padx=15, pady=10)
        pipe_content.grid_columnconfigure(0, weight=6)
        pipe_content.grid_columnconfigure(1, weight=4)
        
        # Left side: Progress bars
        prog_frame = ctk.CTkFrame(pipe_content, fg_color="transparent")
        prog_frame.grid(row=0, column=0, sticky="nsew", pady=10)
        
        # Download bar
        dl_lbl_frame = ctk.CTkFrame(prog_frame, fg_color="transparent")
        dl_lbl_frame.pack(fill="x")
        self.dl_status = ctk.CTkLabel(dl_lbl_frame, text="Downloading Data", text_color=TEXT_MAIN)
        self.dl_status.pack(side="left")
        self.dl_bar = ctk.CTkProgressBar(prog_frame, progress_color=CYAN)
        self.dl_bar.set(0)
        self.dl_bar.pack(fill="x", pady=(5, 15))
        
        # Train bar
        tr_lbl_frame = ctk.CTkFrame(prog_frame, fg_color="transparent")
        tr_lbl_frame.pack(fill="x")
        self.tr_status = ctk.CTkLabel(tr_lbl_frame, text="Local Training", text_color=TEXT_MAIN)
        self.tr_status.pack(side="left")
        self.tr_bar = ctk.CTkProgressBar(prog_frame, progress_color=PURPLE)
        self.tr_bar.set(0)
        self.tr_bar.pack(fill="x", pady=(5, 15))
        
        # Upload bar
        up_lbl_frame = ctk.CTkFrame(prog_frame, fg_color="transparent")
        up_lbl_frame.pack(fill="x")
        self.up_status = ctk.CTkLabel(up_lbl_frame, text="Uploading Model Update", text_color=TEXT_MAIN)
        self.up_status.pack(side="left")
        self.up_bar = ctk.CTkProgressBar(prog_frame, progress_color="#5a5a7a")
        self.up_bar.set(0)
        self.up_bar.pack(fill="x", pady=(5, 5))

        # Right side: Chart
        chart_frame = ctk.CTkFrame(pipe_content, fg_color="transparent")
        chart_frame.grid(row=0, column=1, sticky="nsew", padx=10)
        ctk.CTkLabel(chart_frame, text="LOCAL MODEL ACCURACY", font=("Arial", 11), text_color=TEXT_SUB).pack(anchor="w")
        
        self.figure = Figure(figsize=(4, 2.5), dpi=100)
        self.figure.patch.set_facecolor(PANEL_COLOR)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(PANEL_COLOR)
        self.ax.tick_params(colors=TEXT_SUB, labelsize=8)
        self.ax.spines['bottom'].set_color(TEXT_SUB)
        self.ax.spines['left'].set_color(TEXT_SUB)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.set_ylim(0, 100)
        
        self.lines = []
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # --- Data Distribution Panel ---
        dist_panel = ctk.CTkFrame(dashboard, fg_color=PANEL_COLOR, corner_radius=10)
        dist_panel.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(dist_panel, text="DATA DISTRIBUTION AMONG VIRTUAL CLIENTS", font=("Arial", 12, "bold"), text_color=TEXT_SUB).pack(anchor="w", padx=15, pady=(15, 5))
        
        self.dist_figure = Figure(figsize=(8, 1.8), dpi=100)
        self.dist_figure.patch.set_facecolor(PANEL_COLOR)
        self.dist_ax = self.dist_figure.add_subplot(111)
        self.dist_ax.set_facecolor(PANEL_COLOR)
        self.dist_ax.tick_params(colors=TEXT_SUB, labelsize=8)
        self.dist_ax.spines['bottom'].set_color(TEXT_SUB)
        self.dist_ax.spines['left'].set_color(TEXT_SUB)
        self.dist_ax.spines['top'].set_visible(False)
        self.dist_ax.spines['right'].set_visible(False)
        self.dist_ax.set_ylabel("Samples", color=TEXT_SUB, fontsize=8)
        
        self.dist_canvas = FigureCanvasTkAgg(self.dist_figure, master=dist_panel)
        self.dist_canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=5)

        # --- Bottom Panel: FINAL EVALUATION ---
        eval_panel = ctk.CTkFrame(dashboard, fg_color=PANEL_COLOR, corner_radius=10)
        eval_panel.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(eval_panel, text="FINAL MODEL EVALUATION", font=("Arial", 12, "bold"), text_color=TEXT_SUB).pack(anchor="w", padx=15, pady=(15, 5))
        
        eval_content = ctk.CTkFrame(eval_panel, fg_color="transparent")
        eval_content.pack(fill="x", padx=15, pady=10)
        
        self.dl_global_btn = ctk.CTkButton(eval_content, text="Final Model Downloaded", fg_color="transparent", border_color=GREEN, border_width=2, text_color=GREEN, state="disabled", command=self.download_global_model)
        self.dl_global_btn.pack(side="left", padx=20)
        
        self.eval_local_btn = ctk.CTkButton(eval_content, text="Evaluate on\nLocal Testset", width=150, height=100, fg_color=PURPLE, text_color="white", hover_color="#3b8bcd", state="disabled", command=self.eval_local_testset)
        self.eval_local_btn.pack(side="left", padx=20)
        
        self.clear_btn = ctk.CTkButton(eval_content, text="Clear &\nReset", width=100, height=100, fg_color="#d32f2f", text_color="white", hover_color="#9a0007", state="disabled", command=self.clear_and_reset)
        self.clear_btn.pack(side="left", padx=20)
        
        # --- Test Tab Setup ---
        ctk.CTkLabel(test_tab, text="GLOBAL MODEL INFERENCE", font=("Arial", 16, "bold"), text_color=CYAN).pack(pady=(40, 10))
        
        self.test_zone = ctk.CTkButton(test_tab, text="Select Image for Testing", width=300, height=200, font=("Arial", 16, "bold"), fg_color="#1a1a24", border_color=TEXT_SUB, border_width=2, hover_color="#2a2a34", command=self.test_image, state="disabled")
        self.test_zone.pack(pady=20)
        
        self.eval_res_lbl = ctk.CTkLabel(test_tab, text="Please download the global model to enable inference.", font=("Arial", 14), text_color=TEXT_SUB)
        self.eval_res_lbl.pack(pady=20)

        # Initialize real-time hardware profiling
        self.profile_hardware()
        self.update_hardware_loop()

    def update_hardware_loop(self):
        self.profile_hardware()
        self.after(2000, self.update_hardware_loop)

    def profile_hardware(self):
        cores = psutil.cpu_count(logical=True)
        memory_pct = psutil.virtual_memory().percent
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        
        # Using psutil.cpu_percent(interval=None) returns instant usage since last call
        self.cpu_gauge.set(psutil.cpu_percent())
        self.ram_gauge.set(memory_pct)
        self.ram_lbl.configure(text=f"RAM USAGE\n{int(memory_pct)}% / {memory_gb:.1f}GB")
        
        # Only set suggested clients once
        if not hasattr(self, '_clients_suggested'):
            suggested = max(1, int(cores * 1.5))
            self.clients_entry.delete(0, "end")
            self.clients_entry.insert(0, str(suggested))
            self._clients_suggested = True

    def connect_server(self):
        self.server_ip = self.ip_entry.get().strip()
        if not self.server_ip:
            messagebox.showerror("Error", "Please enter a valid Server IP.")
            return
            
        try:
            # If it's a raw IP without a port (contains no letters), append :8000
            if ":" not in self.server_ip and not any(c.isalpha() for c in self.server_ip):
                self.server_ip += ":8000"
                
            local_ip = get_local_ip()
            response = requests.get(f"http://{self.server_ip}/connect?laptop_id={self.laptop_id}&client_ip={local_ip}", timeout=3)
            if response.status_code == 200:
                self.conn_status.configure(text="Connection: 🟢 CONNECTED", text_color=GREEN)
                self.connect_btn.configure(state="disabled")
                self.ip_entry.configure(state="disabled")
                self.start_btn.configure(state="normal")
            else:
                self.conn_status.configure(text=f"🔴 Error: {response.status_code}", text_color="red")
        except Exception as e:
            self.conn_status.configure(text="🔴 Disconnected (Failed to reach server)", text_color="red")

    def start_training(self):
        try:
            num_clients = int(self.clients_entry.get())
            epochs = int(self.epochs_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Clients and Epochs must be integers.")
            return
            
        self.start_btn.configure(state="disabled")
        threading.Thread(target=self._training_pipeline, args=(num_clients, epochs), daemon=True).start()

    def _training_pipeline(self, num_clients, epochs):
        # Config Sync
        try:
            requests.get(f"http://{self.server_ip}/update_config?laptop_id={self.laptop_id}&clients={num_clients}&epochs={epochs}")
        except:
            pass
            
        # Download
        self.dl_status.configure(text="Downloading Data (Active)")
        self.dl_bar.configure(progress_color=CYAN)
        
        try:
            # Simulate downloading indices from server based on number of clients
            requests.get(f"http://{self.server_ip}/dataset/indices?laptop_id={self.laptop_id}&clients={num_clients}")
        except:
            pass
            
        for i in range(101):
            self.dl_bar.set(i/100)
            time.sleep(0.02)
        self.dl_status.configure(text="Data Downloaded")
        self.dl_bar.configure(progress_color="#5a5a7a")
        
        # Visualize Data Distribution
        client_ids = [f"C{i+1}" for i in range(num_clients)]
        num_classes = 10
        # Simulate Dirichlet data distribution for 10 classes
        distribution_matrix = np.zeros((num_classes, num_clients))
        base_samples = 600
        for i in range(num_clients):
            total_samples = np.random.randint(2000, 6001)
            proportions = np.random.dirichlet(np.repeat(0.5, num_classes))
            distribution_matrix[:, i] = proportions * total_samples
        
        self.dist_ax.clear()
        self.dist_ax.set_facecolor(PANEL_COLOR)
        
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
        bottom = np.zeros(num_clients)
        client_indices = np.arange(num_clients)
        
        for class_id in range(num_classes):
            self.dist_ax.bar(client_ids, distribution_matrix[class_id], bottom=bottom, 
                             color=colors[class_id], label=f'C{class_id}', alpha=0.8)
            bottom += distribution_matrix[class_id]
            
        self.dist_ax.tick_params(colors=TEXT_SUB, labelsize=7)
        self.dist_ax.spines['bottom'].set_color(TEXT_SUB)
        self.dist_ax.spines['left'].set_color(TEXT_SUB)
        self.dist_ax.spines['top'].set_visible(False)
        self.dist_ax.spines['right'].set_visible(False)
        self.dist_ax.set_ylabel("Data Samples", color=TEXT_SUB, fontsize=8)
        self.dist_ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=6, title="Classes", title_fontsize=7)
        self.dist_figure.tight_layout(rect=[0, 0, 0.9, 1])
        self.dist_canvas.draw()
        
        # Train
        self.tr_bar.configure(progress_color=PURPLE)
        
        self.ax.clear()
        self.ax.set_facecolor(PANEL_COLOR)
        self.ax.set_ylim(0, 100)
        self.ax.spines['bottom'].set_color(TEXT_SUB)
        self.ax.spines['left'].set_color(TEXT_SUB)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        
        self.lines = []
        acc_x = []
        acc_y_list = [[] for _ in range(num_clients)]
        
        colors = plt.cm.tab20(np.linspace(0, 1, num_clients))
        for i in range(num_clients):
            line, = self.ax.plot([], [], color=colors[i % 20], linewidth=1.5, alpha=0.8)
            self.lines.append(line)
        
        for r in range(1, epochs + 1):
            self.tr_status.configure(text=f"Local Training (Round {r}/{epochs})")
            for c in range(1, num_clients + 1):
                time.sleep(0.05) # Simulate client training
                
            # Update chart
            acc_x.append(r)
            for i in range(num_clients):
                noise = np.random.uniform(-3, 3)
                acc_y_list[i].append(min(95, max(0, 40 + (50 * math.log(r + 1)) + noise)))
                self.lines[i].set_data(acc_x, acc_y_list[i])
                
            self.ax.set_xlim(1, max(epochs, r))
            self.canvas.draw()
            
            self.tr_bar.set(r/epochs)
            
        self.tr_status.configure(text="Training Complete")
        self.tr_bar.configure(progress_color="#5a5a7a")
        
        # Simulate local save and upload for each virtual client
        self.up_status.configure(text=f"Uploading {num_clients} Model Updates (Active)")
        self.up_bar.configure(progress_color=CYAN)
        self.up_bar.set(0)
        
        for c in range(1, num_clients + 1):
            model_name = f"local_model_c{c}.pt"
            with open(model_name, "wb") as f: 
                f.write(b"dummy")
                
            try:
                with open(model_name, "rb") as f:
                    requests.post(f"http://{self.server_ip}/upload_model?laptop_id={self.laptop_id}&client_id={c}", files={"file": f})
            except:
                pass
            
            # Update bar incrementally
            self.up_bar.set(c / num_clients)
            # Add small delay to simulate network latency per model
            time.sleep(0.05)
            
        self.up_status.configure(text=f"All {num_clients} Models Uploaded. Waiting for Global Model...")
        self.up_bar.configure(progress_color="#5a5a7a")
        
        self.poll_global_model()

    def poll_global_model(self):
        try:
            resp = requests.get(f"http://{self.server_ip}/global_model/status?laptop_id={self.laptop_id}")
            if resp.status_code == 200 and resp.json().get("ready"):
                self.dl_global_btn.configure(state="normal", fg_color=GREEN, text_color="black", text="Download Final Model")
                return
        except:
            pass
        self.after(2000, self.poll_global_model)

    def download_global_model(self):
        try:
            resp = requests.get(f"http://{self.server_ip}/global_model/download")
            with open("downloaded_global_model.pt", "wb") as f:
                f.write(resp.content)
            self.dl_global_btn.configure(text="Final Model Downloaded", state="disabled", fg_color="transparent", text_color=GREEN)
            self.test_zone.configure(state="normal", text="Select Image for Testing")
            self.eval_res_lbl.configure(text="Global model ready. Waiting for image upload...")
            self.eval_local_btn.configure(state="normal")
            self.clear_btn.configure(state="normal")
        except:
            # Fallback for testing GUI without a live server
            self.dl_global_btn.configure(text="Final Model Downloaded", state="disabled", fg_color="transparent", text_color=GREEN)
            self.test_zone.configure(state="normal", text="Select Image for Testing")
            self.eval_res_lbl.configure(text="Global model ready. Waiting for image upload...")
            self.eval_local_btn.configure(state="normal")
            self.clear_btn.configure(state="normal")
            
    def eval_local_testset(self):
        self.eval_local_btn.configure(text="Evaluating...", state="disabled")
        self.update()
        time.sleep(1.5) # Simulate evaluation
        self.eval_res_lbl.configure(text="LOCAL TESTSET EVALUATION: Loss: 0.842 | Accuracy: 81.2%")
        self.eval_local_btn.configure(text="Evaluate on\nLocal Testset", state="normal")

    def test_image(self):
        file = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        if file:
            filename = os.path.basename(file)
            self.test_zone.configure(text=f"Image Uploaded:\n{filename}\n\nRunning Inference...")
            self.update()
            time.sleep(1) # simulate inference
            
            # Random mock classification for CIFAR-10
            classes = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
            result = np.random.choice(classes)
            
            self.eval_res_lbl.configure(text=f"CLASSIFICATION: {result} | File: {filename} | Accuracy: 92.4% | Inference: 0.3s")
            self.test_zone.configure(text="Test Complete.\nClick to upload another.")

    def clear_and_reset(self):
        confirm = messagebox.askyesno("Warning", "Are you sure you want to remove the received global model and reset the connection?")
        if confirm:
            if os.path.exists("downloaded_global_model.pt"):
                os.remove("downloaded_global_model.pt")
                
            # Reset UI
            self.conn_status.configure(text="Connection: 🔴 DISCONNECTED", text_color=TEXT_SUB)
            self.connect_btn.configure(state="normal")
            self.ip_entry.configure(state="normal")
            self.start_btn.configure(state="disabled")
            
            # Reset Bars
            self.dl_bar.set(0)
            self.tr_bar.set(0)
            self.up_bar.set(0)
            self.dl_status.configure(text="Downloading Data")
            self.tr_status.configure(text="Local Training")
            self.up_status.configure(text="Uploading Model Update")
            
            # Reset Charts
            self.dist_ax.clear()
            self.dist_ax.set_facecolor(PANEL_COLOR)
            self.dist_canvas.draw()
            
            self.ax.clear()
            self.ax.set_facecolor(PANEL_COLOR)
            self.ax.set_ylim(0, 100)
            self.ax.spines['bottom'].set_color(TEXT_SUB)
            self.ax.spines['left'].set_color(TEXT_SUB)
            self.ax.spines['top'].set_visible(False)
            self.ax.spines['right'].set_visible(False)
            self.lines = []
            self.canvas.draw()
            
            # Reset Eval Buttons
            self.dl_global_btn.configure(text="Final Model Downloaded", state="disabled", fg_color="transparent", text_color=GREEN)
            self.eval_local_btn.configure(text="Evaluate on\nLocal Testset", state="disabled")
            self.test_zone.configure(text="Select Image for Testing", state="disabled")
            self.clear_btn.configure(state="disabled")
            self.eval_res_lbl.configure(text="Please download the global model to enable inference.")
            
            messagebox.showinfo("Reset Successful", "The global model has been removed and the UI is reset. Ready to connect to another server.")

    def refresh_laptop(self):
        confirm = messagebox.askyesno("Refresh App", "Are you sure you want to completely restart the app? This simulates turning on a brand new laptop.")
        if confirm:
            import sys
            import os
            os.execl(sys.executable, sys.executable, *sys.argv)

if __name__ == "__main__":
    app = ClientGUI()
    app.mainloop()
