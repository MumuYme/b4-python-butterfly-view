import numpy as np
import cv2
import os
# from google.colab.patches import cv2_imshow # Colabで画像を表示するための特殊な関数（ローカルでは不要）

def show_image(img, title="image"):
    """ローカルでの表示: まずOpenCVのウィンドウ、ダメならmatplotlib、さらにダメならファイル保存"""
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
    """
    指定されたサイズで市松模様を生成し、PNGで出力するシステム
    """
    # 画像全体を黒(0)で初期化（グレースケール）
    image = np.zeros((height, width), dtype=np.uint8)

    # タイルの行数と列数を計算
    num_rows = height // tile_size
    num_cols = width // tile_size

    # 各タイルにアクセスして色を決定
    for r in range(num_rows):
        for c in range(num_cols):
            # タイルの左上隅のピクセル座標を計算
            y_start = r * tile_size
            x_start = c * tile_size

            # 行インデックス(r)と列インデックス(c)の合計が偶数なら白(255)
            if (r + c) % 2 == 0:
                # タイルの領域を白 (255) で埋める (スライス機能を利用)
                # image[y座標の範囲, x座標の範囲] = 値
                image[y_start : y_start + tile_size, x_start : x_start + tile_size] = 255

    # 画像を指定されたファイル名でPNG形式で出力
    success = cv2.imwrite(output_filename, image)

    if success:
        print(f"✅ 市松模様の画像が '{output_filename}' として出力されました！")
    else:
        print("❌ 画像の出力に失敗しました。")

    return image


# --- 実行 ---
# 1. マスの一辺を 50ピクセル
TILE_SIZE = 50
# 2. 全体の幅を 400ピクセル
TOTAL_WIDTH = 400
# 3. 全体の高さを 300ピクセル
TOTAL_HEIGHT = 300

# --- 変更: 出力ファイル名を指定の形式にする ---
output_filename = f"tilesize{TILE_SIZE}_w{TOTAL_WIDTH}xh{TOTAL_HEIGHT}.png"

generated_img = generate_checkerboard(
    tile_size=TILE_SIZE,
    width=TOTAL_WIDTH,
    height=TOTAL_HEIGHT,
    output_filename=output_filename
)

print("\n--- 生成された画像 ---")
# --- 変更: OSの既定ビューアで開く（Pillow.show と同等の挙動） ---
try:
    os.startfile(output_filename)
except Exception:
    # フォールバックで既存の表示関数を使う
    show_image(generated_img)