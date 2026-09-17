"""
生成均衡的合成 UI 数据集，用于 Computer Use Model 训练/测试。

特点:
- 6 类动作，每类等量样本 (默认 40/类 = 240)
- 每个样本包含: 合成界面截图 + 自然语言指令 + 动作标签
- 截图内容与动作语义相关 (可学习信号)
- 无需控制真实鼠标/键盘, 可在容器/虚拟机中安全运行

输出:
- synthetic_data/images/*.png
- synthetic_data/dataset.json
- synthetic_data/vocab.json
"""
import os
import json
import random
import colorsys
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = "synthetic_data"
IMG_DIR = os.path.join(OUT_DIR, "images")
W, H = 224, 224
SAMPLES_PER_ACTION = 40
SEED = 1337

ACTIONS = ["click", "type", "scroll", "move", "hotkey", "press"]


def rand_color(rng):
    h = rng.random()
    s = rng.uniform(0.3, 0.8)
    v = rng.uniform(0.6, 1.0)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def get_font(size=12):
    for name in ["arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_click(img, d, rng, font):
    x, y = rng.randint(40, W - 60), rng.randint(40, H - 40)
    d.rectangle([x, y, x + rng.randint(50, 90), y + rng.randint(20, 34)],
                fill=rand_color(rng), outline=(30, 30, 30), width=2)
    d.text((x + 6, y + 6), "BUTTON", fill=(255, 255, 255), font=font)
    d.line([x - 12, y - 12, x - 2, y - 2], fill=(0, 0, 0), width=2)
    return "click", {"x": x + 25, "y": y + 12}


def draw_type(img, d, rng, font):
    x, y = 30, rng.randint(60, 120)
    d.rectangle([x, y, W - 30, y + 40], outline=(60, 60, 60), width=2, fill=(255, 255, 255))
    d.line([x + 5, y + 8, x + 5, y + 32], fill=(0, 0, 0), width=2)
    word = rng.choice(["hello", "data", "model", "test", "train"])
    d.text((x + 12, y + 12), word, fill=(0, 0, 0), font=font)
    return "type", {"text": word}


def draw_scroll(img, d, rng, font):
    for i in range(6):
        yy = 20 + i * 34
        d.rectangle([20, yy, W - 20, yy + 24], fill=rand_color(rng), outline=(80, 80, 80))
    arrow_up = rng.random() < 0.5
    cx = W // 2
    if arrow_up:
        d.polygon([(cx, 20), (cx - 20, 50), (cx + 20, 50)], fill=(0, 0, 0))
    else:
        d.polygon([(cx, H - 20), (cx - 20, H - 50), (cx + 20, H - 50)], fill=(0, 0, 0))
    return "scroll", {"direction": "up" if arrow_up else "down", "amount": rng.randint(1, 5)}


def draw_move(img, d, rng, font):
    x, y = rng.randint(30, W - 30), rng.randint(30, H - 30)
    d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(255, 0, 0))
    d.line([x - 20, y, x + 20, y], fill=(0, 0, 0), width=1)
    d.line([x, y - 20, x, y + 20], fill=(0, 0, 0), width=1)
    return "move", {"x": x, "y": y}


def draw_hotkey(img, d, rng, font):
    d.rectangle([30, 80, W - 30, 140], fill=(255, 245, 170), outline=(120, 120, 0), width=2)
    d.text((38, 100), "selected text", fill=(0, 0, 0), font=font)
    key = rng.choice([("ctrl", "c"), ("ctrl", "v"), ("ctrl", "x")])
    d.text((38, 150), "+".join(key), fill=(0, 0, 120), font=font)
    return "hotkey", {"keys": list(key)}


def draw_press(img, d, rng, font):
    d.rectangle([40, 70, W - 40, 160], fill=(230, 230, 230), outline=(0, 0, 0), width=2)
    key = rng.choice(["enter", "tab", "escape"])
    d.text((60, 100), f"press {key}", fill=(0, 0, 0), font=font)
    return "press", {"key": key}


DRAWERS = {
    "click": draw_click,
    "type": draw_type,
    "scroll": draw_scroll,
    "move": draw_move,
    "hotkey": draw_hotkey,
    "press": draw_press,
}


def make_instruction(action, params):
    if action == "click":
        return f"click the button at {params['x']} {params['y']}"
    if action == "type":
        return f"type {params['text']} in the textbox"
    if action == "scroll":
        return f"scroll {params['direction']} the page"
    if action == "move":
        return f"move the cursor to {params['x']} {params['y']}"
    if action == "hotkey":
        return "press " + " ".join(params["keys"])
    if action == "press":
        return f"press {params['key']}"
    return "unknown"


def main():
    rng = random.Random(SEED)
    os.makedirs(IMG_DIR, exist_ok=True)
    font = get_font(12)

    data = []
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 0

    for action in ACTIONS:
        for _ in range(SAMPLES_PER_ACTION):
            bg = rand_color(rng)
            img = Image.new("RGB", (W, H), bg)
            d = ImageDraw.Draw(img)
            act, params = DRAWERS[action](img, d, rng, font)
            instr = make_instruction(act, params)

            fname = f"{idx:05d}_{action}.png"
            img.save(os.path.join(IMG_DIR, fname))

            for tok in instr.lower().split():
                if tok not in vocab:
                    vocab[tok] = len(vocab)

            data.append({
                "screenshot": os.path.join(IMG_DIR, fname).replace("\\", "/"),
                "instruction": instr,
                "action": act,
                "params": params,
            })
            idx += 1

    with open(os.path.join(OUT_DIR, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_DIR, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"生成完成: {len(data)} 个样本, 词表大小 {len(vocab)}")
    print(f"输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
