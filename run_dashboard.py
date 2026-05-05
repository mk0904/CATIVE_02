import http.server
import socketserver
import webbrowser
import threading
import time
import os

PORT = 8080
DIRECTORY = "." # Serve from root so ../outputs/results works

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    # Silence logging
    def log_message(self, format, *args):
        pass

def start_server():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving CATIVE Dashboard at http://localhost:{PORT}/dashboard/")
        httpd.serve_forever()

if __name__ == "__main__":
    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Start server in a background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Wait a second for server to start, then open browser
    time.sleep(1)
    webbrowser.open(f'http://localhost:{PORT}/dashboard/')
    
    print("Dashboard opened in your browser! Press Ctrl+C to stop the server.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
