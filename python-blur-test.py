from PIL import Image, ImageFilter, ImageDraw, ImageFont
import os

img_path = r"C:\\Users\\mana\\project\\b4-python-butterfly-review\\assets\\black\\checkerboard_squareSize16_squareColor000000.png"

# 画像読み込み（RGBA にして合成を安定させる）
img = Image.open(img_path).convert("RGBA")

# 中央 100x100 の座標を計算
w, h = img.size
size = 400
left = (w - size) // 2
top = (h - size) // 2
right = left + size
bottom = top + size

# 全体をぼかした画像を用意（ぼかし強さは radius を調整）
blur_radius = 6
blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

# マスク作成（白い部分がぼかし領域）
mask = Image.new("L", img.size, 0)
m_draw = ImageDraw.Draw(mask)
m_draw.rectangle((left, top, right, bottom), fill=255)

# マスクで合成（白い部分は blurred、黒い部分は元画像）
result = Image.composite(blurred, img, mask)

# --- ここから最小変更：中央に16pxでNoto Sans JPを載せる ---
# フォント候補（Regular を優先）
font_candidates = [
    r"C:\\Users\\mana\\project\\b4-butterfly-review\\src\\fonts\\NotoSansJP-Regular.ttf",
]
font = None
for fp in font_candidates:
    if os.path.exists(fp):
        try:
            if fp.lower().endswith(".ttc"):
                font = ImageFont.truetype(fp, 16, index=0)
            else:
                font = ImageFont.truetype(fp, 16)
            break
        except Exception:
            continue
if font is None:
    font = ImageFont.load_default()

draw = ImageDraw.Draw(result)
text = "テスト表示"
# 文字サイズを測って中央位置を計算
try:
    # Pillow 8.0+ では textbbox が推奨
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
except AttributeError:
    try:
        # 互換性のためのフォールバック
        tw, th = font.getsize(text)
    except Exception:
        # 最終手段
        mask = font.getmask(text)
        tw, th = mask.size

tx = (w - tw) // 2
ty = (h - th) // 2
draw.text((tx, ty), text, font=font, fill=(255, 255, 255, 255))
# --- ここまで ---

# 保存と表示
out_path = r"C:\\Users\\mana\\project\\b4-python-butterfly-review\\assets\\black\\center_blur_100.png"
result.save(out_path)
print("保存しました:", out_path)
result.show()

