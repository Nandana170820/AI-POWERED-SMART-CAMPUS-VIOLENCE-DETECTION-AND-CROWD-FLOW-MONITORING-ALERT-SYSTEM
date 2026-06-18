import os
import cv2
import argparse

def extract_frames(input_folder, output_folder, num_frames=8, size=(224,224), quality=80):
    os.makedirs(output_folder, exist_ok=True)

    # Loop through ALL video files in the folder
    for idx, filename in enumerate(os.listdir(input_folder)):
        if filename.endswith(".avi"):   # RWF dataset uses .avi
            video_path = os.path.join(input_folder, filename)
            cap = cv2.VideoCapture(video_path)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                print(f"⚠️ Skipping {filename}, no frames detected.")
                continue

            # ✅ Step size to sample evenly across the video
            step = max(1, total_frames // num_frames)

            saved_count = 0
            frame_count = 0
            clip_folder = os.path.join(output_folder, f"clip_{idx:04d}")
            os.makedirs(clip_folder, exist_ok=True)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Save frame if it's at the right interval
                if frame_count % step == 0 and saved_count < num_frames:
                    frame = cv2.resize(frame, size)
                    frame_path = os.path.join(clip_folder, f"frame_{saved_count:04d}.jpg")
                    cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                    saved_count += 1
                frame_count += 1

            cap.release()
            print(f"✅ Extracted {saved_count} frames from {filename} into {clip_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input folder with videos")
    parser.add_argument("--output", required=True, help="Output folder for frames")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of frames to save per video")
    parser.add_argument("--size", type=int, nargs=2, default=[224,224], help="Resize width height")
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality (0-100)")
    args = parser.parse_args()

    extract_frames(args.input, args.output, num_frames=args.num_frames, size=tuple(args.size), quality=args.quality)