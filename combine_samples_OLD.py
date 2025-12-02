import os
import glob
import argparse
import cv2
import sys

#!/usr/bin/env python3
"""
combine_samples.py

Read a folder of 3 types of images (via 3 glob patterns), arrange each triple side-by-side,
and write out a video.

Usage:
    python combine_samples.py --dir /path/to/images \
        --patterns "*_input.png" "*_pred.png" "*_gt.png" \
        --output combined.mp4 --fps 10 --height 512 --labels Input Pred GT
"""


def parse_args():
    p = argparse.ArgumentParser(description="Combine 3 image streams side-by-side into a video")
    p.add_argument("--dir", "-d", required=True, help="Directory containing images")
    p.add_argument("--patterns", "-p", nargs=3, required=False, default=['dynmask_sparse_*.png', 'rgb_*.png', 'validmask_sparse_*.png'],
                   help="Three glob patterns (quoted) to match each image type, e.g. 'dynmask_sparse_*.png' 'rgb_*.png' 'validmask_sparse_*.png'")
    p.add_argument("--output_dir", default=None, help="Output video directory")
    p.add_argument("--fps", type=int, default=5, help="Frames per second")
    p.add_argument("--height", type=int, default=None, help="Resize each image to this height (preserve aspect ratio). If omitted, uses min height of each triple.")
    p.add_argument("--labels", nargs=3, default=["Dymask","RGB","Valid"], help="Optional labels for the three panels")
    return p.parse_args()

def sorted_glob(directory, pattern):
    matches = glob.glob(os.path.join(directory, pattern))
    matches.sort()
    return matches

def read_img(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img

def resize_to_height(img, target_h):
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

def main():
    args = parse_args()

    for sample_dir in os.listdir(args.dir):
        dir = os.path.join(args.dir, sample_dir)
        if args.output_dir is None:
            args.output_dir = args.dir
        os.makedirs(args.output_dir, exist_ok=True)
        output = os.path.join(args.output_dir, f'{sample_dir}.mp4')

        lists = [ sorted_glob(dir, pat) for pat in args.patterns ]
        lengths = [len(l) for l in lists]
        if any(l == 0 for l in lengths):
            print("No matches for some patterns. Matches:", lengths, file=sys.stderr)
            sys.exit(1)

        n_frames = min(lengths)
        if any(l != n_frames for l in lengths):
            print(f"Different counts found {lengths}, using first {n_frames} frames (min).", file=sys.stderr)

        # Prepare first triple to determine width/height
        first_imgs = [ read_img(lists[i][0]) for i in range(3) ]
        if args.height is None:
            target_h = min(img.shape[0] for img in first_imgs)
        else:
            target_h = args.height

        # Resize first triple to compute combined width
        resized_first = [ resize_to_height(img, target_h) for img in first_imgs ]
        frame_width = sum(img.shape[1] for img in resized_first)
        frame_height = target_h

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print(f"Writing video to {output} at {frame_width}x{frame_height} @ {args.fps} fps")
        out = cv2.VideoWriter(output, fourcc, float(args.fps), (frame_width, frame_height))
        if not out.isOpened():
            print("Failed to open video writer. Check output path/codec.", file=sys.stderr)
            sys.exit(1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.6, target_h / 300.0)
        thickness = max(1, int(round(font_scale * 2)))

        for i in range(n_frames):
            try:
                imgs = [ read_img(lists[j][i]) for j in range(3) ]
            except RuntimeError as e:
                print("Skipping frame due to read error:", e, file=sys.stderr)
                continue

            resized = [ resize_to_height(img, target_h) for img in imgs ]
            combined = cv2.hconcat(resized)

            # Draw labels centered above each panel
            x_offset = 0
            for idx, img in enumerate(resized):
                w = img.shape[1]
                label = args.labels[idx]
                (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                text_x = x_offset + max(5, (w - text_w) // 2)
                text_y = max(20, text_h + 6)
                # background rectangle for readability
                rect_tl = (text_x - 6, text_y - text_h - 4)
                rect_br = (text_x + text_w + 6, text_y + 4)
                cv2.rectangle(combined, rect_tl, rect_br, (0,0,0), thickness=cv2.FILLED)
                cv2.putText(combined, label, (text_x, text_y), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)
                x_offset += w

            out.write(combined)

        out.release()
        print(f"Wrote {output} ({n_frames} frames @ {args.fps} fps)")

if __name__ == "__main__":
    main()