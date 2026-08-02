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
BG_COLOR = "#1a1a24"
PANEL_COLOR = "#252533"
SIDEBAR_COLOR = "#1e1e2d"
CYAN = "#00f2fe"
GREEN = "#00e676"
PURPLE = "#4facfe"
TEXT_MAIN = "#ffffff"
TEXT_SUB = "#a0a0b5"

CLASS_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

ctk.set_appearance_mode("Dark")

class ClientGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("FAB-FL Client Node v2.0")
        self.geometry("1100x750")
        self.configure(fg_color=BG_COLOR)
        
        self.last_log_count = 0
        
        self.frames = {}
        self.sidebar_btns = {}
        self.current_frame = None

        self.setup_ui()
        self.select_frame("Home")
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
        
        self.btn_start.configure(state="disabled", text="Running...")
        self.btn_stop.configure(state="normal")
        self.url_entry.configure(state="disabled")
        self.n_entry.configure(state="disabled")
        self.dev_entry.configure(state="disabled")
        self.lbl_pool.configure(text=f"Local Client Pool (N={n})")
        
        for widget in self.pool_scroll.winfo_children():
            widget.destroy()
            
        self.progress_bar.set(0)
        self.progress_bar.pack(side="left", padx=10, expand=True, fill="x")
        
        threading.Thread(target=Client.start_process, args=(url, n), daemon=True).start()

    def stop_process(self):
        Client.stop_process()
        self.btn_start.configure(state="normal", text="▶ Start Process")
        self.btn_stop.configure(state="disabled")
        self.btn_connect.configure(state="normal")
        
        self.url_entry.configure(state="normal")
        self.n_entry.configure(state="normal")
        self.dev_entry.configure(state="normal")
        
        self.progress_bar.pack_forget()

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
        
        self.lbl_pred_result.configure(text="Predicting...", text_color=CYAN)
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
                btn.configure(fg_color=PANEL_COLOR)
            else:
                btn.configure(fg_color="transparent")
                
        # Show selected frame
        if self.current_frame is not None:
            self.current_frame.grid_forget()
            
        self.current_frame = self.frames[name]
        self.current_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # --- SIDEBAR (Left) ---
        sidebar = ctk.CTkFrame(self, fg_color=SIDEBAR_COLOR, width=200, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_rowconfigure(3, weight=1)
        
        ctk.CTkLabel(sidebar, text="FAB-FL Edge", font=("Arial", 22, "bold"), text_color=PURPLE).pack(pady=(20, 30), padx=20)
        
        menu_items = ["Home", "Model Prediction"]
        for item in menu_items:
            btn = ctk.CTkButton(sidebar, text=item, font=("Arial", 16), fg_color="transparent", text_color=TEXT_MAIN, hover_color=PANEL_COLOR, anchor="w",
                                command=lambda name=item: self.select_frame(name))
            btn.pack(fill="x", padx=10, pady=5)
            self.sidebar_btns[item] = btn

        # --- MAIN CONTENT CONTAINER (Right) ---
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=0, column=1, sticky="nsew")
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)
        
        self.setup_home_frame()
        self.setup_model_frame()

    def setup_home_frame(self):
        frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.frames["Home"] = frame
        
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x", pady=(0, 20))
        self.lbl_status = ctk.CTkLabel(header, text="Status: Waiting to Connect...", font=("Arial", 16, "bold"), text_color=TEXT_SUB)
        self.lbl_status.pack(side="right")
        
        conn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        conn_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(conn_frame, text="Device:", font=("Arial", 14)).pack(side="left", padx=5)
        self.dev_entry = ctk.CTkEntry(conn_frame, width=100, font=("Arial", 14))
        self.dev_entry.insert(0, "Laptop-2")
        self.dev_entry.pack(side="left", padx=5)
        
        ctk.CTkLabel(conn_frame, text="Server URL:", font=("Arial", 14)).pack(side="left", padx=5)
        self.url_entry = ctk.CTkEntry(conn_frame, width=180, font=("Arial", 14))
        self.url_entry.insert(0, "http://127.0.0.1:8000")
        self.url_entry.pack(side="left", padx=10)
        
        ctk.CTkLabel(conn_frame, text="Clients (N):", font=("Arial", 14)).pack(side="left", padx=5)
        self.n_entry = ctk.CTkEntry(conn_frame, width=60, font=("Arial", 14))
        self.n_entry.insert(0, "20")
        self.n_entry.pack(side="left", padx=10)
        
        self.btn_connect = ctk.CTkButton(conn_frame, text="🔌 Connect", fg_color=PURPLE, hover_color="#3a90da", command=self.connect_client)
        self.btn_connect.pack(side="left", padx=5)
        
        self.btn_start = ctk.CTkButton(conn_frame, text="▶ Start Process", fg_color=GREEN, hover_color="#00c853", command=self.start_process, state="disabled")
        self.btn_start.pack(side="left", padx=5)

        self.btn_stop = ctk.CTkButton(conn_frame, text="⏹ Stop", fg_color="#d62728", hover_color="#b51a1a", command=self.stop_process, state="disabled")
        self.btn_stop.pack(side="left", padx=5)
        
        self.progress_bar = ctk.CTkProgressBar(conn_frame, fg_color="#333", progress_color=CYAN)
        self.progress_bar.set(0)

        dash_body = ctk.CTkFrame(frame, fg_color="transparent")
        dash_body.pack(fill="both", expand=True, pady=10)
        dash_body.grid_columnconfigure((0, 1), weight=1)
        dash_body.grid_rowconfigure(0, weight=1)
        
        pool_frame = ctk.CTkFrame(dash_body, fg_color=PANEL_COLOR, corner_radius=10)
        pool_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.lbl_pool = ctk.CTkLabel(pool_frame, text="Local Client Pool (Waiting...)", font=("Arial", 14, "bold"), text_color=TEXT_SUB)
        self.lbl_pool.pack(pady=10)
        self.pool_scroll = ctk.CTkScrollableFrame(pool_frame, fg_color="transparent")
        self.pool_scroll.pack(fill="both", expand=True, padx=10, pady=10)

        log_frame = ctk.CTkFrame(dash_body, fg_color=PANEL_COLOR, corner_radius=10)
        log_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        ctk.CTkLabel(log_frame, text="Communication Terminal", font=("Arial", 14, "bold"), text_color=TEXT_SUB).pack(pady=10)
        self.log_textbox = ctk.CTkTextbox(log_frame, fg_color="#1a1a24", text_color=GREEN, font=("Courier", 12), state="disabled")
        self.log_textbox.pack(fill="both", expand=True, padx=10, pady=10)

        status_frame = ctk.CTkFrame(frame, fg_color=PANEL_COLOR, corner_radius=10)
        status_frame.pack(fill="x", pady=10)
        self.lbl_active = ctk.CTkLabel(status_frame, text="Active Client: None", font=("Arial", 16, "bold"), text_color=CYAN)
        self.lbl_active.pack(side="left", padx=20, pady=15)
        self.lbl_memory = ctk.CTkLabel(status_frame, text="", font=("Arial", 14, "bold"), text_color=GREEN)
        self.lbl_memory.pack(side="right", padx=20, pady=15)

    def setup_model_frame(self):
        frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.frames["Model Prediction"] = frame
        ctk.CTkLabel(frame, text="Client-Side Model Inference", font=("Arial", 22, "bold"), text_color=CYAN).pack(pady=(0, 10), anchor="w")
        ctk.CTkLabel(frame, text="Test the received global model locally.", font=("Arial", 14), text_color=TEXT_SUB).pack(pady=(0, 20), anchor="w")
        
        pred_frame = ctk.CTkFrame(frame, fg_color=PANEL_COLOR, corner_radius=10)
        pred_frame.pack(fill="both", expand=True, pady=10)
        
        btn_browse = ctk.CTkButton(pred_frame, text="📁 Browse Image", command=self.browse_image)
        btn_browse.pack(pady=20)
        
        self.lbl_selected_img = ctk.CTkLabel(pred_frame, text="No image selected.", text_color=TEXT_SUB)
        self.lbl_selected_img.pack(pady=5)
        
        self.lbl_preview = ctk.CTkLabel(pred_frame, text="[ Image Preview ]", width=150, height=150, fg_color="#1a1a24")
        self.lbl_preview.pack(pady=20)
        
        btn_predict = ctk.CTkButton(pred_frame, text="🎯 Predict Class", command=self.predict, fg_color=GREEN, hover_color="#00c853")
        btn_predict.pack(pady=20)
        
        self.lbl_pred_result = ctk.CTkLabel(pred_frame, text="", font=("Arial", 24, "bold"))
        self.lbl_pred_result.pack(pady=10)

    def draw_graph(self, parent_frame, class_counts, total):
        graph_frame = ctk.CTkFrame(parent_frame, fg_color="transparent", height=15)
        graph_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        if total == 0:
            return
            
        for c in range(10):
            count = class_counts.get(c, 0)
            if count > 0:
                block = ctk.CTkFrame(graph_frame, fg_color=CLASS_COLORS[c], height=15, corner_radius=2)
                block.pack(side="left", fill="x", expand=True)

    def update_dashboard(self):
        state = Client.client_state
        
        status_text = state['current_status']
        color = GREEN if status_text == "Connected!" else TEXT_SUB
        self.lbl_status.configure(text=f"Status: {status_text}", text_color=color)
        
        if self.progress_bar.winfo_ismapped():
            self.progress_bar.set(state["partition_progress"])
        
        if len(self.pool_scroll.winfo_children()) == 0 and len(state["local_clients"]) > 0:
            for client in state["local_clients"]:
                gid = client["global_id"]
                counts = sum(client["class_counts"].values())
                
                row = ctk.CTkFrame(self.pool_scroll, fg_color="#1a1a24", corner_radius=5)
                row.pack(fill="x", pady=5)
                
                info_frame = ctk.CTkFrame(row, fg_color="transparent")
                info_frame.pack(fill="x")
                ctk.CTkLabel(info_frame, text=gid, font=("Arial", 12, "bold")).pack(side="left", padx=10, pady=2)
                ctk.CTkLabel(info_frame, text=f"Data Samples: {counts}", font=("Arial", 12), text_color=TEXT_SUB).pack(side="right", padx=10, pady=2)
                
                self.draw_graph(row, client["class_counts"], counts)
                
        active_gid = state["active_training_gid"]
        for widget in self.pool_scroll.winfo_children():
            try:
                info_frame = widget.winfo_children()[0]
                lbl_gid = info_frame.winfo_children()[0].cget("text")
                if lbl_gid == active_gid:
                    widget.configure(border_width=2, border_color=CYAN)
                else:
                    widget.configure(border_width=0)
            except:
                pass
                
        if active_gid:
            self.lbl_active.configure(text=f"Active Client: {active_gid} (Training...)")
        else:
            self.lbl_active.configure(text="Active Client: None")
            
        self.lbl_memory.configure(text=state["memory_cleared_msg"])
        
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
