import customtkinter as ctk
import threading
import sys
import os
import time
import tkinter.filedialog as filedialog
from PIL import Image

# Add parent directory to path to import backend logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Client
from predictor import predict_image

# --- Styling Constants ---
BG_COLOR = "#121212"
PANEL_COLOR = "#222222"
SIDEBAR_COLOR = "#0a0a0a"
YELLOW = "#ffff00"
GREEN = "#00e676"
TEXT_MAIN = "#ffffff"
TEXT_SUB = "#cccccc"

CLASS_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

ctk.set_appearance_mode("Dark")

class ClientGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("FAB-FL Client Node v2.0")
        self.geometry("1280x800")
        self.configure(fg_color=BG_COLOR)
        
        self.last_log_count = 0
        
        self.frames = {}
        self.sidebar_btns = {}
        self.current_frame = None

        self.setup_ui()
        self.select_frame("HOME")
        self.update_dashboard()

    def connect_client(self):
        url = self.url_entry.get().strip()
        n_val = self.n_entry.get().strip()
        dev_name = self.dev_entry.get().strip()
        
        if not url or not n_val.isdigit() or not dev_name:
            return
            
        n = int(n_val)
        
        success = Client.connect_device(url, n, dev_name)
        if success:
            self.btn_connect.configure(state="disabled")
            self.btn_start.configure(state="normal")
            
    def start_process(self):
        url = self.url_entry.get().strip()
        n_val = self.n_entry.get().strip()
        n = int(n_val)
        alpha = self.alpha_slider.get()
        
        self.btn_start.configure(state="disabled", text="Running...")
        self.btn_stop.configure(state="normal")
        self.url_entry.configure(state="disabled")
        self.n_entry.configure(state="disabled")
        self.dev_entry.configure(state="disabled")
        self.alpha_slider.configure(state="disabled")
        
        for widget in self.pool_scroll.winfo_children():
            widget.destroy()
            
        threading.Thread(target=Client.start_process, args=(url, n, alpha), daemon=True).start()

    def update_alpha_label(self, value):
        self.alpha_lbl.configure(text=f"{value:.2f}")

    def stop_process(self):
        Client.stop_process()
        self.btn_start.configure(state="normal", text="▶ START PROCESS")
        self.btn_stop.configure(state="disabled")
        self.btn_connect.configure(state="normal")
        
        self.url_entry.configure(state="normal")
        self.n_entry.configure(state="normal")
        self.dev_entry.configure(state="normal")
        self.alpha_slider.configure(state="normal")

    def browse_image(self):
        filepath = filedialog.askopenfilename(
            title="Select Image for Prediction",
            filetypes=(("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*"))
        )
        if filepath:
            self.selected_img_path = filepath
            self.lbl_selected_img.configure(text=f"Selected: {os.path.basename(filepath)}")
            
            img = Image.open(filepath)
            img.thumbnail((150, 150))
            self.img_preview = ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
            self.lbl_preview.configure(image=self.img_preview, text="")

    def predict(self):
        if not hasattr(self, 'selected_img_path') or not self.selected_img_path:
            self.lbl_pred_result.configure(text="Please select an image first.", text_color="red")
            return
            
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'local_global_model.pt'))
        
        self.lbl_pred_result.configure(text="Predicting...", text_color=YELLOW)
        self.update_idletasks()
        
        result = predict_image(model_path, self.selected_img_path)
        
        if "Error" in result:
            self.lbl_pred_result.configure(text=result, text_color="red")
        else:
            self.lbl_pred_result.configure(text=f"Prediction: {result.upper()}", text_color=GREEN)

    def select_frame(self, name):
        # Update button colors
        for btn_name, btn in self.sidebar_btns.items():
            if btn_name == name:
                btn.configure(fg_color=YELLOW, text_color="black")
            else:
                btn.configure(fg_color="transparent", text_color=TEXT_MAIN)
                
        # Show selected frame
        if self.current_frame is not None:
            self.current_frame.grid_forget()
            
        self.current_frame = self.frames[name]
        self.current_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

    def create_glass_panel(self, parent):
        return ctk.CTkFrame(parent, fg_color=PANEL_COLOR, border_width=1, border_color=YELLOW, corner_radius=15)

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # --- SIDEBAR (Left) ---
        sidebar = ctk.CTkFrame(self, fg_color=SIDEBAR_COLOR, width=220, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_rowconfigure(6, weight=1)
        
        # Dashboard icon / Logo
        logo_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        logo_frame.pack(pady=(30, 40))
        ctk.CTkLabel(logo_frame, text="⛙", font=("Arial", 40), text_color=YELLOW).pack()
        
        menu_items = [("HOME", "🏠"), ("MODEL PREDICTION", "📈"), ("SETTINGS", "⚙️")]
        for item_name, icon in menu_items:
            # For multiline text in buttons or long text, we ensure it fits or adjust width
            display_name = item_name.replace(" ", "\n") if item_name == "MODEL PREDICTION" else item_name
            btn = ctk.CTkButton(sidebar, text=f"{icon}   {display_name}", font=("Arial", 13, "bold"), 
                                fg_color="transparent", text_color=TEXT_MAIN, hover_color="#333333", anchor="w",
                                corner_radius=20, width=180, height=55 if "\n" in display_name else 45,
                                command=lambda name=item_name: self.select_frame(name))
            btn.pack(padx=20, pady=5)
            self.sidebar_btns[item_name] = btn

        # Exit button at bottom
        btn_exit = ctk.CTkButton(sidebar, text="🚪", font=("Arial", 20), fg_color="transparent", 
                                 hover_color="#333333", width=50, command=self.quit)
        btn_exit.pack(side="bottom", anchor="sw", padx=20, pady=30)

        # --- MAIN CONTENT CONTAINER (Right) ---
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=0, column=1, sticky="nsew")
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)
        
        self.setup_home_frame()
        self.setup_model_frame()
        self.setup_settings_frame()

    def add_input(self, parent, label, default_val):
        ctk.CTkLabel(parent, text=label, anchor="w", text_color=TEXT_MAIN, font=("Arial", 12)).pack(anchor="w", padx=15, pady=(5, 0))
        entry = ctk.CTkEntry(parent, fg_color="transparent", border_color="#555", text_color=TEXT_MAIN)
        entry.insert(0, str(default_val))
        entry.pack(fill="x", padx=15, pady=(0, 5))
        return entry

    def setup_home_frame(self):
        frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.frames["HOME"] = frame
        
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(header, text="Client Node Dashboard", font=("Arial", 32, "bold"), text_color=TEXT_MAIN).pack(side="left")
        
        dash_body = ctk.CTkFrame(frame, fg_color="transparent")
        dash_body.pack(fill="both", expand=True)
        
        dash_body.grid_columnconfigure(0, weight=1)
        dash_body.grid_columnconfigure(1, weight=2)
        dash_body.grid_rowconfigure(0, weight=1)
        
        # --- LEFT COLUMN ---
        left_col = ctk.CTkFrame(dash_body, fg_color="transparent")
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # 1. Connection Setup
        conn_frame = self.create_glass_panel(left_col)
        conn_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(conn_frame, text="CONNECTION SETUP", font=("Arial", 12, "bold"), text_color=TEXT_MAIN).pack(anchor="nw", padx=15, pady=(15, 5))
        
        self.dev_entry = self.add_input(conn_frame, "Device Name:", "Laptop-2")
        self.url_entry = self.add_input(conn_frame, "Server URL:", "http://127.0.0.1:8000")
        self.n_entry = self.add_input(conn_frame, "Clients (N):", "5")
        
        alpha_frame = ctk.CTkFrame(conn_frame, fg_color="transparent")
        alpha_frame.pack(fill="x", padx=15, pady=(5, 0))
        ctk.CTkLabel(alpha_frame, text="Alpha (Non-IID):", anchor="w", text_color=TEXT_MAIN, font=("Arial", 12)).pack(side="left")
        self.alpha_lbl = ctk.CTkLabel(alpha_frame, text="0.50", text_color=YELLOW, font=("Arial", 12))
        self.alpha_lbl.pack(side="right")
        
        self.alpha_slider = ctk.CTkSlider(conn_frame, from_=0.1, to=5.0, number_of_steps=49, button_color=YELLOW, progress_color=YELLOW, command=self.update_alpha_label)
        self.alpha_slider.set(0.5)
        self.alpha_slider.pack(fill="x", padx=15, pady=(0, 5))
        
        self.btn_connect = ctk.CTkButton(conn_frame, text="🔌 CONNECT", fg_color=YELLOW, text_color="black", hover_color="#cccc00", corner_radius=20, font=("Arial", 14, "bold"), command=self.connect_client)
        self.btn_connect.pack(pady=(15, 5), padx=15, fill="x")
        
        self.lbl_handshake = ctk.CTkLabel(conn_frame, text="Handshake Status: Not Connected", font=("Arial", 12), text_color=YELLOW)
        self.lbl_handshake.pack(anchor="w", padx=15, pady=(0, 15))
        
        # 2. Training Status
        status_frame = self.create_glass_panel(left_col)
        status_frame.pack(fill="both", expand=True, pady=(10, 0))
        
        ctk.CTkLabel(status_frame, text="TRAINING STATUS", font=("Arial", 12, "bold"), text_color=TEXT_MAIN).pack(anchor="nw", padx=15, pady=(15, 5))
        ctk.CTkLabel(status_frame, text="ACTIVE TRAINING & MEMORY STATUS", font=("Arial", 11), text_color=TEXT_SUB).pack(anchor="nw", padx=15)
        
        self.lbl_active = ctk.CTkLabel(status_frame, text="Currently Training:\n[ None ]", font=("Arial", 24, "bold"), text_color=TEXT_MAIN)
        self.lbl_active.pack(pady=40)
        
        self.lbl_memory = ctk.CTkLabel(status_frame, text="Memory Management: idle.\nWaiting for tasks.", font=("Arial", 12), text_color=YELLOW, justify="left")
        self.lbl_memory.pack(anchor="sw", side="bottom", padx=15, pady=20)
        
        # --- RIGHT COLUMN ---
        right_col = ctk.CTkFrame(dash_body, fg_color="transparent")
        right_col.grid(row=0, column=1, sticky="nsew")
        
        # 3. Process Controls
        proc_frame = self.create_glass_panel(right_col)
        proc_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(proc_frame, text="PROCESS CONTROLS", font=("Arial", 12, "bold"), text_color=TEXT_MAIN).pack(anchor="nw", padx=15, pady=(15, 10))
        
        btn_row = ctk.CTkFrame(proc_frame, fg_color="transparent")
        btn_row.pack(anchor="w", padx=15, pady=5)
        
        self.btn_start = ctk.CTkButton(btn_row, text="▶ START PROCESS", width=140, fg_color=YELLOW, text_color="black", hover_color="#cccc00", corner_radius=20, font=("Arial", 12, "bold"), command=self.start_process, state="disabled")
        self.btn_start.pack(side="left", padx=(0, 10))
        
        self.btn_stop = ctk.CTkButton(btn_row, text="⏹ STOP", width=80, fg_color="transparent", border_width=1, border_color=YELLOW, text_color=YELLOW, hover_color="#333300", corner_radius=20, font=("Arial", 12, "bold"), command=self.stop_process, state="disabled")
        self.btn_stop.pack(side="left", padx=10)
        
        self.lbl_status = ctk.CTkLabel(proc_frame, text="Starts data download, Non-IID partitioning, registers clients, polls server.\nOverall Status: Idle", font=("Arial", 12), text_color=TEXT_SUB, justify="left")
        self.lbl_status.pack(anchor="w", padx=15, pady=(10, 5))
        
        self.progress_bar = ctk.CTkProgressBar(proc_frame, progress_color=YELLOW, fg_color="#333333", height=15, corner_radius=10)
        self.progress_bar.pack(fill="x", padx=15, pady=(0, 15))
        self.progress_bar.set(0)
        
        # 4. Local Virtual Clients
        pool_frame = self.create_glass_panel(right_col)
        pool_frame.pack(fill="both", expand=True, pady=10)
        
        ctk.CTkLabel(pool_frame, text="LOCAL VIRTUAL CLIENTS (CIFAR-10 NON-IID SKEW)", font=("Arial", 12, "bold"), text_color=TEXT_MAIN).pack(anchor="nw", padx=15, pady=(15, 5))
        
        hdr = ctk.CTkFrame(pool_frame, fg_color="transparent")
        hdr.pack(fill="x", padx=15, pady=(5, 0))
        ctk.CTkLabel(hdr, text="Client ID", font=("Arial", 12, "bold"), text_color=TEXT_MAIN, width=60, anchor="w").pack(side="left")
        ctk.CTkLabel(hdr, text="Data Skew Visualizer", font=("Arial", 12, "bold"), text_color=TEXT_MAIN).pack(side="left", padx=20)
        ctk.CTkLabel(hdr, text="Memory Usage", font=("Arial", 12, "bold"), text_color=TEXT_MAIN).pack(side="right")
        
        self.pool_scroll = ctk.CTkScrollableFrame(pool_frame, fg_color="transparent")
        self.pool_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(pool_frame, text="Colored blocks show data skew (width visually represents class presence).", font=("Arial", 11), text_color=TEXT_SUB).pack(anchor="w", padx=15, pady=(0, 10))
        
        # 5. Client Communication Log
        log_frame = self.create_glass_panel(right_col)
        log_frame.pack(fill="both", expand=True, pady=(10, 0))
        
        ctk.CTkLabel(log_frame, text="CLIENT COMMUNICATION LOG (Terminal)", font=("Arial", 12, "bold"), text_color=TEXT_MAIN).pack(anchor="nw", padx=15, pady=(15, 5))
        
        self.log_textbox = ctk.CTkTextbox(log_frame, fg_color="transparent", text_color=TEXT_MAIN, font=("Courier", 12), state="disabled")
        self.log_textbox.pack(fill="both", expand=True, padx=10, pady=(0, 15))

    def setup_model_frame(self):
        frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.frames["MODEL PREDICTION"] = frame
        ctk.CTkLabel(frame, text="Client-Side Model Inference", font=("Arial", 22, "bold"), text_color=YELLOW).pack(pady=(0, 10), anchor="w")
        ctk.CTkLabel(frame, text="Test the received global model locally.", font=("Arial", 14), text_color=TEXT_SUB).pack(pady=(0, 20), anchor="w")
        
        pred_frame = self.create_glass_panel(frame)
        pred_frame.pack(fill="both", expand=True, pady=10)
        
        btn_browse = ctk.CTkButton(pred_frame, text="📁 Browse Image", command=self.browse_image, fg_color="transparent", border_width=1, border_color=YELLOW, text_color=YELLOW, hover_color="#333300")
        btn_browse.pack(pady=20)
        
        self.lbl_selected_img = ctk.CTkLabel(pred_frame, text="No image selected.", text_color=TEXT_SUB)
        self.lbl_selected_img.pack(pady=5)
        
        self.lbl_preview = ctk.CTkLabel(pred_frame, text="[ Image Preview ]", width=150, height=150, fg_color=BG_COLOR)
        self.lbl_preview.pack(pady=20)
        
        btn_predict = ctk.CTkButton(pred_frame, text="🎯 Predict Class", command=self.predict, fg_color=YELLOW, text_color="black", hover_color="#cccc00")
        btn_predict.pack(pady=20)
        
        self.lbl_pred_result = ctk.CTkLabel(pred_frame, text="", font=("Arial", 24, "bold"))
        self.lbl_pred_result.pack(pady=10)

    def setup_settings_frame(self):
        frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.frames["SETTINGS"] = frame
        ctk.CTkLabel(frame, text="Settings", font=("Arial", 22, "bold"), text_color=YELLOW).pack(pady=(0, 10), anchor="w")
        ctk.CTkLabel(frame, text="Client configuration options.", font=("Arial", 14), text_color=TEXT_SUB).pack(pady=(0, 20), anchor="w")

    def draw_graph(self, parent_frame, class_counts, total):
        graph_frame = ctk.CTkFrame(parent_frame, fg_color="transparent", height=20)
        graph_frame.pack(side="left", fill="x", expand=True, padx=20)
        
        if total == 0:
            return
            
        for c in range(10):
            count = class_counts.get(c, 0)
            if count > 0:
                block = ctk.CTkFrame(graph_frame, fg_color=CLASS_COLORS[c], height=20, corner_radius=2)
                block.pack(side="left", fill="x", expand=True, padx=1)

    def update_dashboard(self):
        state = Client.client_state
        
        status_text = state['current_status']
        if "Connected" in status_text:
            self.lbl_handshake.configure(text=f"Handshake Status: {status_text}", text_color=GREEN)
            self.lbl_status.configure(text=f"Starts data download, Non-IID partitioning, registers clients, polls server.\nOverall Status: {status_text}")
        else:
            self.lbl_handshake.configure(text=f"Handshake Status: {status_text}", text_color=YELLOW)
            self.lbl_status.configure(text=f"Starts data download, Non-IID partitioning, registers clients, polls server.\nOverall Status: {status_text}")
            
        if self.progress_bar.winfo_ismapped() and "partition_progress" in state:
            self.progress_bar.set(state["partition_progress"])
        
        if len(self.pool_scroll.winfo_children()) == 0 and len(state["local_clients"]) > 0:
            for client in state["local_clients"]:
                gid = client["global_id"]
                counts = sum(client["class_counts"].values())
                
                row = ctk.CTkFrame(self.pool_scroll, fg_color="transparent")
                row.pack(fill="x", pady=2)
                
                ctk.CTkLabel(row, text=gid, font=("Arial", 13), text_color=TEXT_MAIN, width=60, anchor="w").pack(side="left")
                
                # Mock memory usage for visual completeness as shown in photo
                import random
                mock_mem = f"[e.g., {random.uniform(2.0, 2.5):.1f} GB]"
                ctk.CTkLabel(row, text=mock_mem, font=("Arial", 12), text_color=TEXT_SUB).pack(side="right")
                
                self.draw_graph(row, client["class_counts"], counts)
                
        active_gid = state["active_training_gid"]
        for widget in self.pool_scroll.winfo_children():
            try:
                lbl_gid = widget.winfo_children()[0].cget("text")
                if lbl_gid == active_gid:
                    widget.configure(fg_color="#333333")
                else:
                    widget.configure(fg_color="transparent")
            except:
                pass
                
        if active_gid:
            self.lbl_active.configure(text=f"Currently Training:\n[{active_gid}]")
        else:
            self.lbl_active.configure(text="Currently Training:\n[ None ]")
            
        if state["memory_cleared_msg"]:
            mem_text = "Memory Management: gc.collect()\nsuccess, torch.cuda.empty_cache()\nsuccess, strictly managed for\nsequential local training."
            self.lbl_memory.configure(text=mem_text)
        
        if len(state["logs"]) > self.last_log_count:
            new_logs = state["logs"][self.last_log_count:]
            self.last_log_count = len(state["logs"])
            
            self.log_textbox.configure(state="normal")
            for log in new_logs:
                self.log_textbox.insert("end", log + "\n")
            self.log_textbox.see("end")
            self.log_textbox.configure(state="disabled")
            
        self.after(500, self.update_dashboard)

if __name__ == "__main__":
    gui = ClientGUI()
    gui.mainloop()
