import argparse
import http.server
import os
import socketserver
import sys
import webbrowser
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

import cv2
from rmn import RMN


def run_image(image_path: str, show_window: bool = True):
    path = Path(image_path)
    if not path.is_absolute():
        path = PROJECT_DIR / image_path

    if not path.exists():
        print(f"[Error] Image file not found: {path}")
        return

    print(f"\n[+] Loading model and analyzing image: {path.name}...")
    rmn_model = RMN()
    img = cv2.imread(str(path))
    if img is None:
        print(f"[Error] Could not read image at {path}")
        return

    results = rmn_model.detect_emotion_for_single_frame(img)
    print(f"[+] Faces detected: {len(results)}")

    for i, r in enumerate(results, 1):
        emo = r["emo_label"]
        prob = r["emo_proba"] * 100
        box = f"({r['xmin']}, {r['ymin']}) -> ({r['xmax']}, {r['ymax']})"
        print(f"\n--- Face #{i} ---")
        print(f"Bounding box: {box}")
        print(f"Dominant emotion: {emo.upper()} ({prob:.1f}%)")
        print("Emotion distribution:")
        for item in r["proba_list"]:
            for name, score in item.items():
                bars = "#" * int(score * 20)
                print(f"  {name.rjust(9)}: {bars.ljust(20)} {score*100:5.1f}%")

    output_path = PROJECT_DIR / "output_prediction.png"
    annotated = rmn_model.draw(img.copy(), results)
    cv2.imwrite(str(output_path), annotated)
    print(f"\n[+] Saved annotated result to: {output_path}")

    if show_window:
        try:
            print("[+] Displaying result image (press any key to close window)...")
            cv2.imshow("Residual Masking Network - Prediction", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"[Info] Could not open display window: {e}")


def run_webcam():
    print("\n[+] Starting live webcam demo...")
    print("[+] Note: Press 'q' on the video window anytime to quit.")
    rmn_model = RMN()
    rmn_model.video_demo()


def run_web_app(port=8000):
    space_dir = PROJECT_DIR / "space"
    if not space_dir.exists():
        print(f"[Error] 'space' directory not found at {space_dir}")
        return

    os.chdir(str(space_dir))
    url = f"http://localhost:{port}/index.html"
    print(f"\n[+] Starting local web server at {url}")
    print("[+] Press Ctrl+C in terminal to stop server.\n")

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            # Keep output clean
            pass

    webbrowser.open(url)
    with socketserver.TCPServer(("", port), QuietHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[+] Web server stopped.")


def main():
    parser = argparse.ArgumentParser(description="Residual Masking Network Runner")
    parser.add_argument("--image", nargs="?", const="sample_face.jpg", help="Path to image for emotion detection (default: sample_face.jpg)")
    parser.add_argument("--webcam", action="store_true", help="Launch real-time webcam demo")
    parser.add_argument("--web", action="store_true", help="Launch in-browser WebAssembly demo (space/index.html)")
    parser.add_argument("--port", type=int, default=8000, help="Port for web server (default: 8000)")

    parser.add_argument("--no-window", action="store_true", help="Do not open OpenCV GUI preview window")

    args = parser.parse_args()

    if args.web:
        run_web_app(port=args.port)
    elif args.webcam:
        run_webcam()
    elif args.image is not None:
        run_image(args.image, show_window=not args.no_window)
    else:
        print("\n=======================================================")
        print("  Facial Expression Recognition (ResidualMaskingNetwork)")
        print("=======================================================")
        print("Choose an option:")
        print("  1. Run emotion detection on sample image (sample_face.jpg)")
        print("  2. Run real-time webcam demo")
        print("  3. Launch in-browser web app (space/index.html)")
        print("  4. Run on a custom image file")
        print("  5. Exit")
        choice = input("\nEnter choice [1-5]: ").strip()

        if choice == "1":
            run_image("sample_face.jpg")
        elif choice == "2":
            run_webcam()
        elif choice == "3":
            run_web_app(port=args.port)
        elif choice == "4":
            custom_path = input("Enter path to image: ").strip().strip('"')
            run_image(custom_path)
        else:
            print("Exiting.")


if __name__ == "__main__":
    main()
