import numpy as np
import cv2
import os

# 直接ここにサイズを指定するとコマンドライン引数より優先されます。
# 例: TILE_SIZES = [1,2,3,13,30]
TILE_SIZES = []  # 空にしておくとコマンドライン (--tiles) を使用

# ファイル先頭に置けるオプション（スクリプト内指定、ファイル指定が無い場合に使われます）
# ROI_IN_FILE = [128,128,256,256]          # 単一 ROI
# ROI_IN_FILE = [[128,128,256,256],[0,0,100,100]]  # 複数 ROI
ROI_IN_FILE = None

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

def apply_gaussian_blur_region(image, ksize=(5,5), sigmaX=0, roi=None, output_filename=None):
    """
    矩形ROI指定でその領域のみガウスぼかしする（フェザー無し）
    image: 2D numpy.ndarray (グレースケール) または 3ch
    roi: (x, y, w, h) または None（全体ぼかし）
    """
    if isinstance(ksize, int):
        ksize = (ksize, ksize)
    h, w = image.shape[:2]
    result = image.copy()
    if roi is not None:
        x, y, rw, rh = roi
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
        result = cv2.GaussianBlur(image, ksize, sigmaX)
    if output_filename:
        cv2.imwrite(output_filename, result)
        print(f"✅ ぼかし画像を '{output_filename}' に保存しました。")
    return result

def _parse_tiles_arg(s):
    """'1,2,3' -> [1,2,3]。空文字なら空リストを返す。"""
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [int(p) for p in parts]

def _parse_kernel(s):
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if len(parts) == 0:
        return (9,9)
    if len(parts) == 1:
        k = int(parts[0])
        return (k,k)
    return (int(parts[0]), int(parts[1]))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate checkerboards and optionally blur a rectangular ROI for multiple tile sizes.")
    parser.add_argument("--tile", type=int, default=2, help="single tile size (used if --tiles not specified and TILE_SIZES empty)")
    parser.add_argument("--tiles", type=str, default="", help="comma-separated tile sizes, e.g. 2,4,8")
    parser.add_argument("--width", type=int, default=512, help="image width")
    parser.add_argument("--height", type=int, default=512, help="image height")
    parser.add_argument("--kernel", type=str, default="9,9", help="gaussian kernel as WxH or single value, e.g. 7,7 or 7")
    parser.add_argument("--sigma", type=float, default=0.0, help="sigmaX for GaussianBlur (0 = auto)")
    parser.add_argument("--roi", type=str, default="", help="optional roi as x,y,w,h (omit to blur whole image)")
    parser.add_argument("--roi-file", type=str, default="", help="path to JSON file with 'roi' or 'rois' key")
    parser.add_argument("--no-blur", action="store_true", help="generate checkerboards only, skip blurring step")
    parser.add_argument("--open", action="store_true", help="open the last generated file with the OS default viewer")
    args = parser.parse_args()

    # decide tile sizes list; priority: TILE_SIZES (file) > --tiles > --tile
    if TILE_SIZES:
        tile_list = TILE_SIZES
    elif args.tiles:
        tile_list = _parse_tiles_arg(args.tiles)
        if not tile_list:
            tile_list = [args.tile]
    else:
        tile_list = [args.tile]

    KERNEL = _parse_kernel(args.kernel)
    TOTAL_WIDTH = args.width
    TOTAL_HEIGHT = args.height

    generated_files = []

    roi_arg = None
    # 1) CLI --roi が指定されていればそれを使う（形式: x,y,w,h）
    if args.roi:
        try:
            parts = [int(x) for x in args.roi.split(",")]
            if len(parts) != 4:
                raise ValueError("--roi must be x,y,w,h")
            roi_arg = parts
        except Exception as e:
            print("--roi の解析エラー:", e)
            roi_arg = None

    # 2) --roi-file が指定されれば JSON を読み込む（'roi' か 'rois' キーを許容）
    if roi_arg is None and args.roi_file:
        import json
        if os.path.exists(args.roi_file):
            with open(args.roi_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "roi" in data and isinstance(data["roi"], list) and len(data["roi"]) == 4:
                roi_arg = data["roi"]
            elif isinstance(data, dict) and "rois" in data and isinstance(data["rois"], list):
                roi_arg = data["rois"]  # list of rois
            elif isinstance(data, list):
                # 直接配列を渡すケースを許容（rois のみ）
                roi_arg = data
            else:
                # 互換キー
                if isinstance(data, dict) and "ROI" in data:
                    roi_arg = data["ROI"]
        else:
            print(f"指定された ROI ファイルが見つかりません: {args.roi_file}")

    # 3) スクリプト内定義 ROI_IN_FILE を fallback とする（None でなければ）
    if roi_arg is None and ROI_IN_FILE is not None:
        roi_arg = ROI_IN_FILE

    # 4) CLI --roi-size が指定されていれば「幅・高さだけ」から中央配置の ROI を作る
    #    例: --roi-size 200,120 -> roi = [(TOTAL_WIDTH-200)//2, (TOTAL_HEIGHT-120)//2, 200, 120]
    if roi_arg is None and hasattr(args, "roi_size") and args.roi_size:
        try:
            w,h = [int(x) for x in args.roi_size.split(",")]
            if w <= 0 or h <= 0:
                raise ValueError("roi-size must be positive integers")
            cx = (TOTAL_WIDTH - w) // 2
            cy = (TOTAL_HEIGHT - h) // 2
            # クリップして画像内に収める
            cx = max(0, min(cx, TOTAL_WIDTH - w))
            cy = max(0, min(cy, TOTAL_HEIGHT - h))
            roi_arg = [cx, cy, w, h]
        except Exception as e:
            print("--roi-size の解析エラー:", e)
            roi_arg = None

    for TILE_SIZE in tile_list:
        base_filename = f"tilesize{TILE_SIZE}_w{TOTAL_WIDTH}xh{TOTAL_HEIGHT}.png"
        img = generate_checkerboard(TILE_SIZE, TOTAL_WIDTH, TOTAL_HEIGHT, output_filename=base_filename)

        if args.no_blur:
            generated_files.append(base_filename)
            print(f"→ ぼかしをスキップ: {base_filename}")
            continue

        if roi_arg is not None:
            try:
                if len(roi_arg) == 4:
                    rx, ry, rw, rh = roi_arg
                    out_fname = base_filename.replace(".png", f"_roi_{rx}x{ry}_{rw}x{rh}_gauss{KERNEL[0]}x{KERNEL[1]}.png")
                    blurred = apply_gaussian_blur_region(img, ksize=KERNEL, sigmaX=args.sigma, roi=(rx, ry, rw, rh), output_filename=out_fname)
                    generated_files.append(out_fname)
                else:
                    # 複数 ROI の場合
                    for i, single_roi in enumerate(roi_arg):
                        if len(single_roi) == 4:
                            rx, ry, rw, rh = single_roi
                            out_fname = base_filename.replace(".png", f"_roi{i+1}_{rx}x{ry}_{rw}x{rh}_gauss{KERNEL[0]}x{KERNEL[1]}.png")
                            blurred = apply_gaussian_blur_region(img, ksize=KERNEL, sigmaX=args.sigma, roi=single_roi, output_filename=out_fname)
                            generated_files.append(out_fname)
                        else:
                            print(f"⚠️ ROI {i+1} が無効です: {single_roi}")
            except Exception as e:
                print("--roi の解析に失敗しました。形式は x,y,w,h です。", e)
        else:
            out_fname = base_filename.replace(".png", f"_gauss{KERNEL[0]}x{KERNEL[1]}.png")
            blurred = apply_gaussian_blur_region(img, ksize=KERNEL, sigmaX=args.sigma, roi=None, output_filename=out_fname)
            generated_files.append(out_fname)

    print("\n--- 生成完了 ---")
    for f in generated_files:
        print(f"- {f}")

    if args.open and generated_files:
        last = generated_files[-1]
        try:
            os.startfile(last)
        except Exception:
            img = cv2.imread(last, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"'{last}' を開けませんでした。")
            else:
                show_image(img, title=os.path.basename(last))