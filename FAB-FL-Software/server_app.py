import customtkinter as ctk
import threading
import uvicorn
import time
import sys
import os
import socket
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter.filedialog as filedialog
import shutil
from PIL import Image

# Add parent directory to path to import backend logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Server import app, state
from predictor import predict_image

# --- Styling Constants ---
BG_COLOR = "#121212"
PANEL_COLOR = "#222222"
SIDEBAR_COLOR = "#0a0a0a"
YELLOW = "#ffff00"
GREEN = "#00e676"
TEXT_MAIN = "#ffffff"
TEXT_SUB = "#cccccc"

ctk.set_appearance_mode("Dark")

class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        
    def write(self, string):
        self.text_widget.configure(state="normal")
        self.text_widget.insert("end", string)
        self.text_widget.see("end")
        self.text_widget.configure(state="disabled")
        
    def flush(self):
        pass
        
    def isatty(self):
        return False

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.254.254.254', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

class ServerGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("FAB-FL Server Dashboard v2.0")
        self.geometry("1280x800")
        self.configure(fg_color=BG_COLOR)
        
        self.uvicorn_server = None
        self.server_thread = None

        self.frames = {}
        self.sidebar_btns = {}
        self.current_frame = None

        self.setup_ui()
        self.select_frame("HOME")
        self.update_dashboard()

    def start_fastapi(self):
        if self.uvicorn_server is not None:
            return
            
        def run_server():
            try:
                config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
                self.uvicorn_server = uvicorn.Server(config)
                self.uvicorn_server.run()
            except Exception as e:
                import traceback
                with open("server_error.log", "w") as f:
                    f.write("Server failed to start:\n" + traceback.format_exc())
                self.lbl_server_status.configure(text="Status: ERROR", text_color="red")
            
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.lbl_server_status.configure(text=f"Server running: uvicorn server_app.py\nHost IP: {get_local_ip()}:8000", text_color=TEXT_SUB)
        
    def stop_fastapi(self):
        if self.uvicorn_server is not None:
            self.uvicorn_server.should_exit = True
            self.uvicorn_server = None
            self.lbl_server_status.configure(text="Server stopped", text_color="red")
            
    def update_params(self):
        try:
            if self.entry_rounds.get():
                state.max_rounds = int(self.entry_rounds.get())
            if self.entry_m.get():
                state.m = int(self.entry_m.get())
            if self.entry_k.get():
                state.k = float(self.entry_k.get())
            
            # Send to API if running
            def send_params():
                try:
                    import requests
                    requests.post("http://127.0.0.1:8000/set_params", json={
                        "max_rounds": state.max_rounds,
                        "m": state.m,
                        "k": state.k
                    }, timeout=2)
                except Exception:
                    pass
            threading.Thread(target=send_params, daemon=True).start()
        except Exception as e:
            pass
        self.update_dashboard()

    def restart_fastapi(self):
        self.stop_fastapi()
        self.after(1000, self.start_fastapi)

    def trigger_training(self):
        self.update_params()
        def start_train():
            try:
                import requests
                requests.post("http://127.0.0.1:8000/start_round", timeout=2)
            except Exception as e:
                print(f"Error starting training: {e}")
        threading.Thread(target=start_train, daemon=True).start()

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
            
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'global_model.pt'))
        
        self.lbl_pred_result.configure(text="Predicting...", text_color=YELLOW)
        self.update_idletasks()
        
        result = predict_image(model_path, self.selected_img_path)
        
        if "Error" in result:
            self.lbl_pred_result.configure(text=result, text_color="red")
        else:
            self.lbl_pred_result.configure(text=f"Prediction: {result.upper()}", text_color=GREEN)

    def download_model(self):
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'global_model.pt'))
        if not os.path.exists(model_path):
            print("Error: global_model.pt does not exist. Train the model first.")
            return
            
        save_path = filedialog.asksaveasfilename(
            defaultextension=".pt",
            initialfile="global_model_export.pt",
            title="Save Global Model As",
            filetypes=(("PyTorch Model", "*.pt"), ("All files", "*.*"))
        )
        if save_path:
            shutil.copy2(model_path, save_path)
            print(f"Model successfully saved to {save_path}")

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
        ctk.CTkLabel(logo_frame, text="⚙️", font=("Arial", 32), text_color=YELLOW).pack()
        
        menu_items = [("HOME", "🏠"), ("CLIENTS", "👥"), ("MODEL", "🧠"), ("TERMINAL", "💻"), ("HELP", "❓")]
        for item_name, icon in menu_items:
            btn = ctk.CTkButton(sidebar, text=f"{icon}   {item_name}", font=("Arial", 14, "bold"), 
                                fg_color="transparent", text_color=TEXT_MAIN, hover_color="#333333", anchor="w",
                                corner_radius=20, width=180, height=45,
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
        self.setup_clients_frame()
        self.setup_model_frame()
        self.setup_terminal_frame()
        self.setup_help_frame()

    def create_glass_panel(self, parent):
        return ctk.CTkFrame(parent, fg_color=PANEL_COLOR, border_width=1, border_color=YELLOW, corner_radius=15)

    def setup_home_frame(self):
        frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.frames["HOME"] = frame
        
        ctk.CTkLabel(frame, text="Dashboard", font=("Arial", 32, "bold"), text_color=TEXT_MAIN).pack(anchor="w", pady=(0, 20))
        
        dash_body = ctk.CTkFrame(frame, fg_color="transparent")
        dash_body.pack(fill="both", expand=True)
        
        # Grid config: 3 columns
        dash_body.grid_columnconfigure(0, weight=1)
        dash_body.grid_columnconfigure(1, weight=1)
        dash_body.grid_columnconfigure(2, weight=1)
        
        dash_body.grid_rowconfigure(0, weight=1)
        dash_body.grid_rowconfigure(1, weight=3)
        dash_body.grid_rowconfigure(2, weight=3)
        
        # --- TOP ROW ---
        # 1. Server Controls
        server_controls_frame = self.create_glass_panel(dash_body)
        server_controls_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(server_controls_frame, text="SERVER CONTROLS", font=("Arial", 12, "bold"), text_color=TEXT_MAIN).pack(anchor="nw", padx=15, pady=(15, 10))
        
        btn_row = ctk.CTkFrame(server_controls_frame, fg_color="transparent")
        btn_row.pack(anchor="w", padx=15, pady=5)
        
        ctk.CTkButton(btn_row, text="▶ START", width=80, fg_color=YELLOW, text_color="black", hover_color="#cccc00", corner_radius=20, font=("Arial", 12, "bold"), command=self.start_fastapi).pack(side="left", padx=(0, 10))
        ctk.CTkButton(btn_row, text="⏹ STOP", width=80, fg_color="transparent", border_width=1, border_color=YELLOW, text_color=YELLOW, hover_color="#333300", corner_radius=20, font=("Arial", 12, "bold"), command=self.stop_fastapi).pack(side="left", padx=10)
        ctk.CTkButton(btn_row, text="🔄 RESTART", width=90, fg_color="transparent", border_width=1, border_color=YELLOW, text_color=YELLOW, hover_color="#333300", corner_radius=20, font=("Arial", 12, "bold"), command=self.restart_fastapi).pack(side="left", padx=10)
        
        self.lbl_server_status = ctk.CTkLabel(server_controls_frame, text="Server running: None", font=("Arial", 12), text_color=TEXT_SUB, justify="left")
        self.lbl_server_status.pack(anchor="w", padx=15, pady=10)
        
        # 2. Connected Edge Nodes
        edge_frame = self.create_glass_panel(dash_body)
        edge_frame.grid(row=0, column=1, columnspan=2, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(edge_frame, text="CONNECTED EDGE NODES", font=("Arial", 12, "bold"), text_color=TEXT_MAIN).pack(anchor="nw", padx=15, pady=(15, 5))
        
        self.edge_textbox = ctk.CTkTextbox(edge_frame, fg_color="transparent", text_color=TEXT_MAIN, font=("Courier", 13), state="disabled")
        self.edge_textbox.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # --- MIDDLE ROW ---
        # 3. Training Controls
        training_frame = self.create_glass_panel(dash_body)
        training_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(training_frame, text="TRAINING CONTROLS", font=("Arial", 12, "bold"), text_color=TEXT_MAIN).pack(anchor="nw", padx=15, pady=(15, 10))
        
        def add_input(parent, label, default_val):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=15, pady=5)
            ctk.CTkLabel(row, text=label, width=60, anchor="w", text_color=TEXT_MAIN, font=("Arial", 14)).pack(side="left")
            entry = ctk.CTkEntry(row, fg_color="transparent", border_color="#555", text_color=TEXT_MAIN)
            entry.insert(0, str(default_val))
            entry.pack(side="right", fill="x", expand=True, padx=(10, 0))
            return entry
            
        self.entry_rounds = add_input(training_frame, "Rounds:", state.max_rounds)
        self.entry_m = add_input(training_frame, "m:", state.m)
        self.entry_k = add_input(training_frame, "k:", state.k)
        
        # Reduce pady here so buttons fit inside the frame without getting cropped
        ctk.CTkButton(training_frame, text="🚀 START TRAINING", fg_color=YELLOW, text_color="black", hover_color="#cccc00", corner_radius=20, font=("Arial", 14, "bold"), command=self.trigger_training).pack(pady=(15, 5), padx=15, fill="x")
        
        # Add the requested new button for FREQSEL & BSFL
        ctk.CTkButton(training_frame, text="⚡ RUN FREQSEL & BSFL AGGREGATION", fg_color="transparent", border_width=1, border_color=YELLOW, text_color=YELLOW, hover_color="#333300", corner_radius=20, font=("Arial", 12, "bold"), command=self.trigger_training).pack(pady=(5, 10), padx=15, fill="x")
        
        # 4. UCB Joint Scoring Metrics
        ucb_frame = self.create_glass_panel(dash_body)
        ucb_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(ucb_frame, text="UCB JOINT SCORING METRICS", font=("Arial", 12, "bold"), text_color=TEXT_MAIN).pack(anchor="nw", padx=15, pady=(15, 5))
        
        self.ucb_textbox = ctk.CTkTextbox(ucb_frame, fg_color="transparent", text_color=TEXT_MAIN, font=("Courier", 12), state="disabled")
        self.ucb_textbox.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # 5. Active Selection Monitor
        sel_frame = self.create_glass_panel(dash_body)
        sel_frame.grid(row=1, column=2, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(sel_frame, text="ACTIVE SELECTION MONITOR", font=("Arial", 12, "bold"), text_color=TEXT_MAIN).pack(anchor="nw", padx=15, pady=(15, 5))
        
        # Create a header row for Active Selection Monitor like in the image
        hdr = ctk.CTkFrame(sel_frame, fg_color="transparent")
        hdr.pack(fill="x", padx=15)
        ctk.CTkLabel(hdr, text="Client ID", font=("Arial", 12, "bold"), text_color=TEXT_MAIN).pack(side="left")
        ctk.CTkLabel(hdr, text="Status", font=("Arial", 12, "bold"), text_color=TEXT_MAIN).pack(side="right")
        
        self.sel_labels_frame = ctk.CTkScrollableFrame(sel_frame, fg_color="transparent")
        self.sel_labels_frame.pack(fill="both", expand=True, padx=5, pady=(0, 10))
        
        # --- BOTTOM ROW ---
        # 6. Aggregation Progress Bar
        agg_frame = self.create_glass_panel(dash_body)
        agg_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(agg_frame, text="AGGREGATION PROGRESS BAR", font=("Arial", 12, "bold"), text_color=TEXT_MAIN).pack(anchor="nw", padx=15, pady=(15, 20))
        
        prog_info = ctk.CTkFrame(agg_frame, fg_color="transparent")
        prog_info.pack(fill="x", padx=15)
        self.lbl_progress_txt = ctk.CTkLabel(prog_info, text="Round 0: 0% Complete", font=("Arial", 14), text_color=TEXT_MAIN)
        self.lbl_progress_txt.pack(side="left")
        self.lbl_progress_clients = ctk.CTkLabel(prog_info, text="0/0 Clients", font=("Arial", 14), text_color=TEXT_SUB)
        self.lbl_progress_clients.pack(side="right")
        
        self.progress_bar = ctk.CTkProgressBar(agg_frame, progress_color=YELLOW, fg_color="#333333", height=20, corner_radius=10)
        self.progress_bar.pack(fill="x", padx=15, pady=(10, 20))
        self.progress_bar.set(0)
        
        # 7. Simulated Accuracy vs Comm. Rounds
        chart_frame_container = self.create_glass_panel(dash_body)
        chart_frame_container.grid(row=2, column=1, columnspan=2, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(chart_frame_container, text="SIMULATED ACCURACY VS COMM. ROUNDS", font=("Arial", 12, "bold"), text_color=TEXT_MAIN).pack(anchor="nw", padx=15, pady=(15, 0))
        ctk.CTkLabel(chart_frame_container, text="Live Matplotlib Chart", font=("Arial", 12), text_color=TEXT_SUB).pack(anchor="nw", padx=15)
        
        self.chart_frame = ctk.CTkFrame(chart_frame_container, fg_color="transparent")
        self.chart_frame.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        
        self.fig, self.ax = plt.subplots(figsize=(8, 2.5), dpi=100)
        self.fig.patch.set_facecolor(PANEL_COLOR)
        self.ax.set_facecolor(PANEL_COLOR)
        self.ax.tick_params(colors=TEXT_SUB, labelsize=8)
        self.ax.xaxis.label.set_color(TEXT_SUB)
        self.ax.yaxis.label.set_color(TEXT_SUB)
        for spine in self.ax.spines.values():
            spine.set_color(TEXT_SUB)
            
        self.ax.set_ylabel("Accuracy", fontsize=8)
        self.ax.set_xlabel("Rounds", fontsize=8)
        self.ax.set_xlim(0, 50)
        self.ax.set_ylim(0, 1.0)
        self.line, = self.ax.plot([], [], color=YELLOW, marker='o', markersize=4)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def setup_clients_frame(self):
        frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.frames["CLIENTS"] = frame
        ctk.CTkLabel(frame, text="Virtual Client Registry & Statistics", font=("Arial", 22, "bold"), text_color=YELLOW).pack(pady=(0, 20), anchor="w")
        self.clients_textbox = ctk.CTkTextbox(frame, fg_color=PANEL_COLOR, text_color=TEXT_MAIN, font=("Courier", 13), state="disabled", border_width=1, border_color=YELLOW)
        self.clients_textbox.pack(fill="both", expand=True, pady=10)

    def setup_model_frame(self):
        frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.frames["MODEL"] = frame
        ctk.CTkLabel(frame, text="Model Management & Inference", font=("Arial", 22, "bold"), text_color=YELLOW).pack(pady=(0, 20), anchor="w")
        
        dl_frame = self.create_glass_panel(frame)
        dl_frame.pack(fill="x", pady=10)
        ctk.CTkLabel(dl_frame, text="Export the current Global Model weights (.pt format)", font=("Arial", 14)).pack(side="left", padx=20, pady=20)
        ctk.CTkButton(dl_frame, text="💾 Save Global Model As...", command=self.download_model, fg_color=YELLOW, text_color="black", hover_color="#cccc00").pack(side="right", padx=20, pady=20)
        
        pred_frame = self.create_glass_panel(frame)
        pred_frame.pack(fill="both", expand=True, pady=10)
        
        ctk.CTkLabel(pred_frame, text="CIFAR-10 Image Prediction", font=("Arial", 18, "bold"), text_color=TEXT_MAIN).pack(pady=20)
        
        btn_browse = ctk.CTkButton(pred_frame, text="📁 Browse Image", command=self.browse_image, fg_color="transparent", border_width=1, border_color=YELLOW, text_color=YELLOW, hover_color="#333300")
        btn_browse.pack(pady=10)
        
        self.lbl_selected_img = ctk.CTkLabel(pred_frame, text="No image selected.", text_color=TEXT_SUB)
        self.lbl_selected_img.pack(pady=5)
        
        self.lbl_preview = ctk.CTkLabel(pred_frame, text="[ Image Preview ]", width=150, height=150, fg_color=BG_COLOR)
        self.lbl_preview.pack(pady=20)
        
        btn_predict = ctk.CTkButton(pred_frame, text="🎯 Predict Class", command=self.predict, fg_color=YELLOW, text_color="black", hover_color="#cccc00")
        btn_predict.pack(pady=10)
        
        self.lbl_pred_result = ctk.CTkLabel(pred_frame, text="", font=("Arial", 24, "bold"))
        self.lbl_pred_result.pack(pady=10)

    def setup_terminal_frame(self):
        frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.frames["TERMINAL"] = frame
        ctk.CTkLabel(frame, text="System Terminal", font=("Arial", 22, "bold"), text_color=YELLOW).pack(pady=(0, 10), anchor="w")
        self.terminal_textbox = ctk.CTkTextbox(frame, fg_color=PANEL_COLOR, text_color=GREEN, font=("Courier", 13), border_width=1, border_color=YELLOW)
        self.terminal_textbox.pack(fill="both", expand=True, pady=10)
        sys.stdout = StdoutRedirector(self.terminal_textbox)
        print("--- FAB-FL Terminal Initialized ---")

    def setup_help_frame(self):
        frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.frames["HELP"] = frame
        ctk.CTkLabel(frame, text="Help & Documentation", font=("Arial", 22, "bold"), text_color=YELLOW).pack(pady=(0, 20), anchor="w")
        
        help_text = """
        FAB-FL: Federated Learning Orchestrator
        =======================================
        
        * Home: Overview of the FL process, UCB metrics, and global accuracy.
        * Clients: Detailed view of all registered clients and their data skew.
        * Model: Export the trained global model and test it using a local image.
        * Terminal: View server logs and print statements in real-time.
        
        Parameters:
        - Rounds (K): Total number of communication rounds.
        - m: Number of clients selected per round.
        - k: Selection hyperparameter for the BSFL scheduler.
        
        Workflow:
        1. Start the server (Start).
        2. Launch Edge Nodes to connect and register virtual clients.
        3. Set parameters and click 'Start Training'.
        """
        help_frame = self.create_glass_panel(frame)
        help_frame.pack(fill="both", expand=True, pady=10)
        help_lbl = ctk.CTkLabel(help_frame, text=help_text, justify="left", font=("Courier", 14), text_color=TEXT_MAIN)
        help_lbl.pack(padx=20, pady=20, anchor="nw")

    def update_dashboard(self):
        # Update Edge Nodes Textbox
        edge_text = f"{'Node ID':<15} {'Physical HW':<15} {'Clients':<8} {'Status':<8}\n"
        edge_text += "-"*50 + "\n"
        for dev in state.connected_devices:
            dev_name = dev['device_name']
            clients = dev['num_clients']
            edge_text += f"{dev_name:<15} {'NVIDIA Jetson':<15} {clients:<8} {'Online':<8}\n" # Hardware mocked as in photo
            
        self.edge_textbox.configure(state="normal")
        self.edge_textbox.delete("1.0", "end")
        self.edge_textbox.insert("end", edge_text)
        self.edge_textbox.configure(state="disabled")
        
        # Update Clients Textbox (Clients Tab)
        clients_text = f"{'Global ID':<15} | {'Data Dist(L2)':<15} | {'Speed(ms)':<10} | {'Selections':<10}\n"
        clients_text += "-"*60 + "\n"
        for temp_id, global_id in state.temp_to_global.items():
            info = state.clients.get(global_id, {})
            dist = info.get("distance", 0.0)
            spd = info.get("hardware_speed_ms", 0.0)
            sels = info.get("N_k", 0)
            clients_text += f"{global_id:<15} | {dist:<15.3f} | {spd:<10.1f} | {sels:<10}\n"
            
        self.clients_textbox.configure(state="normal")
        self.clients_textbox.delete("1.0", "end")
        self.clients_textbox.insert("end", clients_text)
        self.clients_textbox.configure(state="disabled")
        
        # Update UCB Textbox
        ucb_text = f"{'Client ID':<12} {'Data Skew':<10} {'Latency':<9} {'UCB Score':<10}\n"
        ucb_text += "-"*45 + "\n"
        for gid, info in state.clients.items():
            util = 1.0 - info.get('distance', 1.0)
            spd = info.get('hardware_speed_ms', 1.0)
            score = info.get('ucb_score', 0.0)
            score_str = f"{score:.3f}" if score != float('inf') else "INF"
            ucb_text += f"{gid:<12} {util:<10.3f} {spd:<7.1f}ms {score_str:<10}\n"
            
        self.ucb_textbox.configure(state="normal")
        self.ucb_textbox.delete("1.0", "end")
        self.ucb_textbox.insert("end", ucb_text)
        self.ucb_textbox.configure(state="disabled")
        
        # Update Active Selection Monitor
        for widget in self.sel_labels_frame.winfo_children():
            widget.destroy()
            
        if not state.selected_clients:
            ctk.CTkLabel(self.sel_labels_frame, text="Waiting for round to start...", text_color=TEXT_SUB).pack(pady=20)
        else:
            for gid in state.selected_clients:
                is_uploaded = gid in state.uploaded_models
                status_text = "🟢 UPLOADED" if is_uploaded else "⏳ TRAINING"
                lbl_color = GREEN if is_uploaded else YELLOW
                
                row = ctk.CTkFrame(self.sel_labels_frame, fg_color="transparent")
                row.pack(fill="x", pady=2, padx=10)
                ctk.CTkLabel(row, text=gid, font=("Arial", 12), text_color=TEXT_MAIN).pack(side="left")
                ctk.CTkLabel(row, text=status_text, font=("Arial", 12, "bold"), text_color=lbl_color).pack(side="right")
                
        # Update Progress Bar
        total_rounds = max(1, state.max_rounds)
        progress = state.current_round / total_rounds
        self.progress_bar.set(progress)
        self.lbl_progress_txt.configure(text=f"Round {state.current_round}: {int(progress * 100)}% Complete")
        
        selected_count = len(state.selected_clients)
        uploaded_count = len(state.uploaded_models)
        self.lbl_progress_clients.configure(text=f"{uploaded_count}/{selected_count} Clients")
        
        # Update Chart
        if state.current_round > 0:
            rounds = list(range(1, state.current_round + 1))
            # Simulate accuracy values for plotting based on rounds
            accs = [min(0.90, 0.40 + 0.15 * np.log(r)) for r in rounds]
            self.line.set_data(rounds, accs)
            self.ax.set_xlim(0, max(12, total_rounds))
            self.canvas.draw()
            
        self.after(500, self.update_dashboard)

if __name__ == "__main__":
    gui = ServerGUI()
    gui.mainloop()
