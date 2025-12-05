import numpy as np
import cv2
import os

# 直接ここにサイズを指定するとコマンドライン引数より優先されます。
# 例: TILE_SIZES = [1,2,3,13,30]
TILE_SIZES = []  # 空にしておくとコマンドライン (--tiles) を使用

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

def _parse_tiles_arg(s):
    """'1,2,3' -> [1,2,3]。空文字なら空リストを返す。"""
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [int(p) for p in parts]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate checkerboard images for multiple tile sizes.")
    parser.add_argument("--tiles", type=str, default="", help="comma-separated tile sizes, e.g. 1,2,3,13,30")
    parser.add_argument("--width", type=int, default=512, help="image width")
    parser.add_argument("--height", type=int, default=512, help="image height")
    parser.add_argument("--open", action="store_true", help="open the last generated file with the OS default viewer")
    args = parser.parse_args()

    # 優先順: ファイル内の TILE_SIZES が非空ならそれを使う。空なら CLI の --tiles を使う。
    if TILE_SIZES:
        tile_list = TILE_SIZES
    else:
        tile_list = _parse_tiles_arg(args.tiles)
        if not tile_list:
            tile_list = [2]  # デフォルト: 2

    TOTAL_WIDTH = args.width
    TOTAL_HEIGHT = args.height

    generated_files = []
    for TILE_SIZE in tile_list:
        output_filename = f"tilesize{TILE_SIZE}_w{TOTAL_WIDTH}xh{TOTAL_HEIGHT}.png"
        img = generate_checkerboard(
            tile_size=TILE_SIZE,
            width=TOTAL_WIDTH,
            height=TOTAL_HEIGHT,
            output_filename=output_filename
        )
        generated_files.append(output_filename)

    print("\n--- 生成されたファイル ---")
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