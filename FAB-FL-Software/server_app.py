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
BG_COLOR = "#1a1a24"
PANEL_COLOR = "#252533"
SIDEBAR_COLOR = "#1e1e2d"
CYAN = "#00f2fe"
GREEN = "#00e676"
PURPLE = "#4facfe"
TEXT_MAIN = "#ffffff"
TEXT_SUB = "#a0a0b5"

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
        self.geometry("1200x850")
        self.configure(fg_color=BG_COLOR)
        
        self.uvicorn_server = None
        self.server_thread = None

        self.frames = {}
        self.sidebar_btns = {}
        self.current_frame = None

        self.setup_ui()
        self.select_frame("Home")
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
        self.lbl_server_status.configure(text="Status: ONLINE", text_color=GREEN)
        
    def stop_fastapi(self):
        if self.uvicorn_server is not None:
            self.uvicorn_server.should_exit = True
            self.uvicorn_server = None
            self.lbl_server_status.configure(text="Status: OFFLINE", text_color="red")
            
    def update_params(self):
        try:
            if self.entry_rounds.get():
                state.max_rounds = int(self.entry_rounds.get())
            if self.entry_m.get():
                state.m = int(self.entry_m.get())
            if self.entry_k.get():
                state.k = int(self.entry_k.get())
            
            # Send to API if running
            def send_params():
                try:
                    import requests
                    requests.post("http://127.0.0.1:8000/set_params", json={
                        "max_rounds": state.max_rounds,
                        "m": state.m,
                        "k": state.k
                    }, timeout=2)
                    print(f"Updated parameters: Rounds={state.max_rounds}, m={state.m}, k={state.k}")
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
        
        self.lbl_pred_result.configure(text="Predicting...", text_color=CYAN)
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
        sidebar.grid_rowconfigure(6, weight=1)
        
        ctk.CTkLabel(sidebar, text="FAB-FL Server", font=("Arial", 22, "bold"), text_color=CYAN).pack(pady=(20, 30), padx=20)
        
        menu_items = ["Home", "Clients", "Model", "Terminal", "Help"]
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
        self.setup_clients_frame()
        self.setup_model_frame()
        self.setup_terminal_frame()
        self.setup_help_frame()

    def setup_home_frame(self):
        frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.frames["Home"] = frame
        
        # Header controls inside Home
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x", pady=(0, 20))
        
        controls = ctk.CTkFrame(header, fg_color="transparent")
        controls.pack(side="left")
        
        ctk.CTkButton(controls, text="▶ Start", width=80, fg_color=GREEN, hover_color="#00c853", command=self.start_fastapi).pack(side="left", padx=5)
        ctk.CTkButton(controls, text="⏹ Stop", width=80, fg_color="#d50000", hover_color="#b71c1c", command=self.stop_fastapi).pack(side="left", padx=5)
        ctk.CTkButton(controls, text="🔄 Restart", width=80, fg_color="#3a3a4d", command=self.restart_fastapi).pack(side="left", padx=5)
        
        ctk.CTkButton(controls, text="🚀 Start Training", width=120, fg_color="#ff9100", hover_color="#ff6d00", command=self.trigger_training).pack(side="left", padx=15)
        
        self.lbl_server_status = ctk.CTkLabel(controls, text="Status: OFFLINE", font=("Arial", 14, "bold"), text_color="red")
        self.lbl_server_status.pack(side="left", padx=20)
        
        local_ip = get_local_ip()
        ip_label = ctk.CTkLabel(controls, text=f"Host IP: {local_ip}:8000", font=("Arial", 14, "bold"), text_color=CYAN)
        ip_label.pack(side="left", padx=10)
        
        self.lbl_round = ctk.CTkLabel(header, text="Round: 0 / 50", font=("Arial", 18, "bold"), text_color=GREEN)
        self.lbl_round.pack(side="right", padx=20)
        
        params_frame = ctk.CTkFrame(header, fg_color="transparent")
        params_frame.pack(side="right", padx=20)
        
        ctk.CTkLabel(params_frame, text="Rounds:", text_color=TEXT_SUB).pack(side="left", padx=2)
        self.entry_rounds = ctk.CTkEntry(params_frame, width=45)
        self.entry_rounds.pack(side="left", padx=2)
        self.entry_rounds.insert(0, str(state.max_rounds))
        
        ctk.CTkLabel(params_frame, text="m:", text_color=TEXT_SUB).pack(side="left", padx=2)
        self.entry_m = ctk.CTkEntry(params_frame, width=45)
        self.entry_m.pack(side="left", padx=2)
        self.entry_m.insert(0, str(state.m))
        
        ctk.CTkLabel(params_frame, text="k:", text_color=TEXT_SUB).pack(side="left", padx=2)
        self.entry_k = ctk.CTkEntry(params_frame, width=45)
        self.entry_k.pack(side="left", padx=2)
        self.entry_k.insert(0, str(state.k))
        
        ctk.CTkButton(params_frame, text="Apply", width=50, command=self.update_params, fg_color="#3a3a4d").pack(side="left", padx=10)

        # Dashboard body
        dash_body = ctk.CTkFrame(frame, fg_color="transparent")
        dash_body.pack(fill="both", expand=True)
        dash_body.grid_columnconfigure((0, 1), weight=1)
        dash_body.grid_rowconfigure((0, 1), weight=1)
        
        edge_frame = ctk.CTkFrame(dash_body, fg_color=PANEL_COLOR, corner_radius=10)
        edge_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        ctk.CTkLabel(edge_frame, text="Connected Edge Nodes", font=("Arial", 14, "bold"), text_color=TEXT_SUB).pack(pady=5)
        self.edge_textbox = ctk.CTkTextbox(edge_frame, fg_color="#1a1a24", text_color=TEXT_MAIN, state="disabled")
        self.edge_textbox.pack(fill="both", expand=True, padx=10, pady=5)
        
        ucb_frame = ctk.CTkFrame(dash_body, fg_color=PANEL_COLOR, corner_radius=10)
        ucb_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        ctk.CTkLabel(ucb_frame, text="UCB Joint Scoring Metrics", font=("Arial", 14, "bold"), text_color=TEXT_SUB).pack(pady=5)
        self.ucb_textbox = ctk.CTkTextbox(ucb_frame, fg_color="#1a1a24", text_color=TEXT_MAIN, font=("Courier", 12), state="disabled")
        self.ucb_textbox.pack(fill="both", expand=True, padx=10, pady=5)
        
        sel_frame = ctk.CTkFrame(dash_body, fg_color=PANEL_COLOR, corner_radius=10)
        sel_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        ctk.CTkLabel(sel_frame, text="Active Selection Monitor", font=("Arial", 14, "bold"), text_color=TEXT_SUB).pack(pady=5)
        self.sel_labels_frame = ctk.CTkScrollableFrame(sel_frame, fg_color="transparent")
        self.sel_labels_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        agg_frame = ctk.CTkFrame(dash_body, fg_color=PANEL_COLOR, corner_radius=10)
        agg_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        
        prog_header = ctk.CTkFrame(agg_frame, fg_color="transparent")
        prog_header.pack(fill="x", padx=20, pady=(10,0))
        ctk.CTkLabel(prog_header, text="Aggregation Progress", font=("Arial", 14, "bold"), text_color=TEXT_SUB).pack(side="left")
        self.lbl_progress_pct = ctk.CTkLabel(prog_header, text="0%", font=("Arial", 14, "bold"), text_color=CYAN)
        self.lbl_progress_pct.pack(side="right")
        
        self.progress_bar = ctk.CTkProgressBar(agg_frame, progress_color=CYAN, height=15)
        self.progress_bar.pack(fill="x", padx=20, pady=5)
        self.progress_bar.set(0)
        
        chart_header = ctk.CTkFrame(agg_frame, fg_color="transparent")
        chart_header.pack(fill="x", padx=20)
        self.lbl_accuracy = ctk.CTkLabel(chart_header, text="Current Accuracy: 0.0%", font=("Arial", 14, "bold"), text_color=GREEN)
        self.lbl_accuracy.pack(side="right")
        
        self.chart_frame = ctk.CTkFrame(agg_frame, fg_color="transparent")
        self.chart_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.fig, self.ax = plt.subplots(figsize=(5, 2.5), dpi=100)
        self.fig.patch.set_facecolor(PANEL_COLOR)
        self.ax.set_facecolor("#1a1a24")
        self.ax.tick_params(colors=TEXT_SUB)
        self.ax.xaxis.label.set_color(TEXT_SUB)
        self.ax.yaxis.label.set_color(TEXT_SUB)
        for spine in self.ax.spines.values():
            spine.set_color(TEXT_SUB)
            
        self.ax.set_title("Simulated Accuracy Growth", color=TEXT_MAIN, fontsize=10)
        self.ax.set_xlim(0, 50)
        self.ax.set_ylim(0, 100)
        self.line, = self.ax.plot([], [], color=GREEN, marker='o')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def setup_clients_frame(self):
        frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.frames["Clients"] = frame
        ctk.CTkLabel(frame, text="Virtual Client Registry & Statistics", font=("Arial", 22, "bold"), text_color=CYAN).pack(pady=(0, 20), anchor="w")
        self.clients_textbox = ctk.CTkTextbox(frame, fg_color=PANEL_COLOR, text_color=TEXT_MAIN, font=("Courier", 13), state="disabled")
        self.clients_textbox.pack(fill="both", expand=True, pady=10)

    def setup_model_frame(self):
        frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.frames["Model"] = frame
        ctk.CTkLabel(frame, text="Model Management & Inference", font=("Arial", 22, "bold"), text_color=CYAN).pack(pady=(0, 20), anchor="w")
        
        dl_frame = ctk.CTkFrame(frame, fg_color=PANEL_COLOR, corner_radius=10)
        dl_frame.pack(fill="x", pady=10)
        ctk.CTkLabel(dl_frame, text="Export the current Global Model weights (.pt format)", font=("Arial", 14)).pack(side="left", padx=20, pady=20)
        ctk.CTkButton(dl_frame, text="💾 Save Global Model As...", command=self.download_model, fg_color=PURPLE).pack(side="right", padx=20, pady=20)
        
        pred_frame = ctk.CTkFrame(frame, fg_color=PANEL_COLOR, corner_radius=10)
        pred_frame.pack(fill="both", expand=True, pady=10)
        
        ctk.CTkLabel(pred_frame, text="CIFAR-10 Image Prediction", font=("Arial", 18, "bold"), text_color=TEXT_SUB).pack(pady=20)
        
        btn_browse = ctk.CTkButton(pred_frame, text="📁 Browse Image", command=self.browse_image)
        btn_browse.pack(pady=10)
        
        self.lbl_selected_img = ctk.CTkLabel(pred_frame, text="No image selected.", text_color=TEXT_SUB)
        self.lbl_selected_img.pack(pady=5)
        
        self.lbl_preview = ctk.CTkLabel(pred_frame, text="[ Image Preview ]", width=150, height=150, fg_color="#1a1a24")
        self.lbl_preview.pack(pady=20)
        
        btn_predict = ctk.CTkButton(pred_frame, text="🎯 Predict Class", command=self.predict, fg_color=GREEN, hover_color="#00c853")
        btn_predict.pack(pady=10)
        
        self.lbl_pred_result = ctk.CTkLabel(pred_frame, text="", font=("Arial", 24, "bold"))
        self.lbl_pred_result.pack(pady=10)

    def setup_terminal_frame(self):
        frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.frames["Terminal"] = frame
        ctk.CTkLabel(frame, text="System Terminal", font=("Arial", 22, "bold"), text_color=CYAN).pack(pady=(0, 10), anchor="w")
        self.terminal_textbox = ctk.CTkTextbox(frame, fg_color=PANEL_COLOR, text_color=GREEN, font=("Courier", 13))
        self.terminal_textbox.pack(fill="both", expand=True, pady=10)
        sys.stdout = StdoutRedirector(self.terminal_textbox)
        print("--- FAB-FL Terminal Initialized ---")

    def setup_help_frame(self):
        frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.frames["Help"] = frame
        ctk.CTkLabel(frame, text="Help & Documentation", font=("Arial", 22, "bold"), text_color=CYAN).pack(pady=(0, 20), anchor="w")
        
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
        help_frame = ctk.CTkFrame(frame, fg_color=PANEL_COLOR, corner_radius=10)
        help_frame.pack(fill="both", expand=True, pady=10)
        help_lbl = ctk.CTkLabel(help_frame, text=help_text, justify="left", font=("Courier", 14))
        help_lbl.pack(padx=20, pady=20, anchor="nw")

    def update_dashboard(self):
        self.lbl_round.configure(text=f"Round: {state.current_round} / {state.max_rounds}")
        
        edge_text = f"{'Device Name':<20} | Hosted Clients\n"
        edge_text += "-"*40 + "\n"
        for dev in state.connected_devices:
            edge_text += f"{dev['device_name']:<20} | {dev['num_clients']}\n"
            
        self.edge_textbox.configure(state="normal")
        self.edge_textbox.delete("1.0", "end")
        self.edge_textbox.insert("end", edge_text)
        self.edge_textbox.configure(state="disabled")
        
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
        
        ucb_text = f"{'Client ID':<15} | {'Data Util':<10} | {'Speed(ms)':<10} | {'UCB Score':<10}\n"
        ucb_text += "-"*55 + "\n"
        for gid, info in state.clients.items():
            util = 1.0 - info.get('distance', 1.0)
            spd = info.get('hardware_speed_ms', 1.0)
            score = info.get('ucb_score', 0.0)
            score_str = f"{score:.3f}" if score != float('inf') else "INF"
            ucb_text += f"{gid:<15} | {util:<10.3f} | {spd:<10.1f} | {score_str:<10}\n"
            
        self.ucb_textbox.configure(state="normal")
        self.ucb_textbox.delete("1.0", "end")
        self.ucb_textbox.insert("end", ucb_text)
        self.ucb_textbox.configure(state="disabled")
        
        for widget in self.sel_labels_frame.winfo_children():
            widget.destroy()
            
        if not state.selected_clients:
            ctk.CTkLabel(self.sel_labels_frame, text="Waiting for round to start...", text_color=TEXT_SUB).pack(pady=20)
        else:
            for gid in state.selected_clients:
                status = "🟢 UPLOADED" if gid in state.uploaded_models else "⏳ TRAINING"
                lbl_color = GREEN if gid in state.uploaded_models else CYAN
                row = ctk.CTkFrame(self.sel_labels_frame, fg_color="#1a1a24", corner_radius=5)
                row.pack(fill="x", pady=2, padx=10)
                ctk.CTkLabel(row, text=gid, font=("Arial", 14, "bold"), text_color=TEXT_MAIN).pack(side="left", padx=10, pady=5)
                ctk.CTkLabel(row, text=status, font=("Arial", 12, "bold"), text_color=lbl_color).pack(side="right", padx=10, pady=5)
                
        progress = state.current_round / max(1, state.max_rounds)
        self.progress_bar.set(progress)
        self.lbl_progress_pct.configure(text=f"{int(progress * 100)}%")
        
        if state.current_round > 0:
            rounds = list(range(1, state.current_round + 1))
            accs = [min(90.0, 40 + 15 * np.log(r)) for r in rounds]
            self.line.set_data(rounds, accs)
            self.ax.set_xlim(0, max(10, state.max_rounds))
            self.canvas.draw()
            self.lbl_accuracy.configure(text=f"Current Accuracy: {accs[-1]:.1f}%")
            
        self.after(500, self.update_dashboard)

if __name__ == "__main__":
    gui = ServerGUI()
    gui.mainloop()
