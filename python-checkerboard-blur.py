import numpy as np
import cv2
import os

def show_image(img, title="image"):
    try:
        cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL)
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    except Exception:
        pass
    try:
        import matplotlib.pyplot as plt
        if img.ndim == 2:
            plt.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
        return
    except Exception:
        pass
    out = "my_checkerboard.png"
    cv2.imwrite(out, img)
    print(f"表示できなかったので '{out}' に保存しました。")

def generate_checkerboard(tile_size, width, height, output_filename="checkerboard.png"):
    image = np.zeros((height, width), dtype=np.uint8)
    num_rows = height // tile_size
    num_cols = width // tile_size
    for r in range(num_rows):
        for c in range(num_cols):
            y_start = r * tile_size
            x_start = c * tile_size
            if (r + c) % 2 == 0:
                image[y_start : y_start + tile_size, x_start : x_start + tile_size] = 255
    success = cv2.imwrite(output_filename, image)
    if success:
        print(f"✅ 市松模様を '{output_filename}' に保存しました。")
    else:
        print("❌ 市松模様の保存に失敗しました。")
    return image

# --- 追加: 領域指定ぼかし関数 ---
def apply_gaussian_blur_region(image, ksize=(5,5), sigmaX=0, roi=None, mask=None, output_filename=None):
    """
    image: 2D numpy.ndarray (グレースケール)
    ksize: タプルまたはint (奇数推奨)
    roi: (x, y, w, h) の矩形。指定するとその領域のみをぼかす。
    mask: image と同サイズの2値配列（0/1 または False/True）。Trueの部分をぼかす。
          roi と mask が両方指定された場合は mask が優先されます。
    output_filename: 指定すると保存する
    """
    if isinstance(ksize, int):
        ksize = (ksize, ksize)

    h, w = image.shape[:2]
    result = image.copy()

    if mask is not None:
        # mask をブーリアンに変換
        mask_bool = np.asarray(mask).astype(bool)
        # 全体をぼかした画像を作り、マスクで切り替える
        blurred_all = cv2.GaussianBlur(image, ksize, sigmaX)
        result = np.where(mask_bool, blurred_all, image).astype(np.uint8)
    elif roi is not None:
        x, y, rw, rh = roi
        # 範囲クリップ
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(w, int(x + rw))
        y2 = min(h, int(y + rh))
        if x1 < x2 and y1 < y2:
            roi_img = image[y1:y2, x1:x2]
            blurred_roi = cv2.GaussianBlur(roi_img, ksize, sigmaX)
            result[y1:y2, x1:x2] = blurred_roi
        else:
            print("⚠️ ROI が画像範囲外です。何も変更しません。")
    else:
        # 全体ぼかし
        result = cv2.GaussianBlur(image, ksize, sigmaX)

    if output_filename:
        cv2.imwrite(output_filename, result)
        print(f"✅ ぼかし画像を '{output_filename}' に保存しました。")

    return result

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Checkerboard + rectangular blur (no feather)")
    parser.add_argument("--tile", type=int, default=2, help="tile size")
    parser.add_argument("--width", type=int, default=512, help="image width")
    parser.add_argument("--height", type=int, default=512, help="image height")
    parser.add_argument("--kernel", type=str, default="9,9", help="kernel as WxH, e.g. 5,5")
    parser.add_argument("--sigma", type=float, default=0.0, help="sigmaX for GaussianBlur (0 = auto)")
    parser.add_argument("--roi", type=str, default="", help="roi as x,y,w,h (optional). Omit to blur entire image")
    args = parser.parse_args()

    TILE_SIZE = args.tile
    TOTAL_WIDTH = args.width
    TOTAL_HEIGHT = args.height
    kx, ky = map(int, args.kernel.split(","))
    KERNEL = (kx, ky)

    base_filename = f"tilesize{TILE_SIZE}_w{TOTAL_WIDTH}xh{TOTAL_HEIGHT}.png"
    img = generate_checkerboard(TILE_SIZE, TOTAL_WIDTH, TOTAL_HEIGHT, output_filename=base_filename)

    if args.roi:
        rx, ry, rw, rh = map(int, args.roi.split(","))
        out_fname = base_filename.replace(".png", f"_roi_{rx}x{ry}_{rw}x{rh}_gauss{KERNEL[0]}x{KERNEL[1]}.png")
        blurred = apply_gaussian_blur_region(img, ksize=KERNEL, sigmaX=args.sigma, roi=(rx, ry, rw, rh), output_filename=out_fname)
    else:
        out_fname = base_filename.replace(".png", f"_gauss{KERNEL[0]}x{KERNEL[1]}.png")
        blurred = apply_gaussian_blur_region(img, ksize=KERNEL, sigmaX=args.sigma, output_filename=out_fname)

    try:
        os.startfile(out_fname)
    except Exception:
        show_image(blurred, title="blurred")