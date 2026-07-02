# -*- coding: utf-8 -*-
"""
像素→文本→通用物体识别 V2 (DeepSeek V4 Pro 文本API)
升级: HSV色彩空间 + 轮廓近似 + 背景分离 + 纹理分析 + 旋转不变
"""

import os, struct, math, colorsys
from openai import OpenAI

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-key-here")
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# =============================================================
# Step 1: 读 BMP 像素 (不变)
# =============================================================
def read_bmp(path):
    with open(path, 'rb') as f:
        f.read(18)
        w = struct.unpack('<I', f.read(4))[0]
        h = struct.unpack('<I', f.read(4))[0]
        f.read(28)
        pixels = []
        for y in range(h):
            row = []
            for x in range(w):
                b, g, r = f.read(3)
                row.append((r, g, b))
            pixels.insert(0, row)
        return w, h, pixels


# =============================================================
# Step 2: 通用特征提取 V2
# =============================================================

def rgb_to_hsv(r, g, b):
    """RGB -> HSV (0-1 归一化)"""
    h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
    return h, s, v

def color_histogram(w, h, pixels):
    """HSV直方图 + 主色分析"""
    hues, sats, vals = [], [], []
    for y in range(h):
        for x in range(w):
            r, g, b = pixels[y][x]
            hsv = rgb_to_hsv(r, g, b)
            hues.append(hsv[0]); sats.append(hsv[1]); vals.append(hsv[2])

    def stats(arr):
        m = sum(arr)/len(arr); sd = math.sqrt(sum((x-m)**2 for x in arr)/len(arr))
        return round(m, 3), round(sd, 3)

    h_mean, h_std = stats(hues)
    s_mean, s_std = stats(sats)
    v_mean, v_std = stats(vals)

    # 主色调分类 (基于H值)
    if h_mean < 0.05 or h_mean > 0.95:   color_name = "red"
    elif 0.05 < h_mean < 0.15:           color_name = "orange"
    elif 0.15 < h_mean < 0.25:           color_name = "yellow"
    elif 0.25 < h_mean < 0.45:           color_name = "green"
    elif 0.45 < h_mean < 0.65:           color_name = "cyan/blue"
    elif 0.65 < h_mean < 0.85:           color_name = "blue/purple"
    elif 0.85 < h_mean < 0.95:           color_name = "pink/magenta"
    else:                                 color_name = "mixed"

    return {
        "dominant_hue": color_name,
        "hsv_mean": (h_mean, s_mean, v_mean),
        "hsv_std": (h_std, s_std, v_std),
        "saturation": "vivid" if s_mean > 0.3 else ("dull" if s_mean < 0.1 else "moderate"),
        "brightness": "bright" if v_mean > 0.6 else ("dark" if v_mean < 0.3 else "medium"),
    }


def edge_intensity(w, h, pixels):
    """边缘强度: Sobel算子（比一阶差分更抗噪）"""
    intensities = []
    for y in range(1, h-1):
        for x in range(1, w-1):
            # 灰度值
            g = lambda r,c: sum(pixels[r][c]) / 3
            gx = (-1*g(y-1,x-1)+1*g(y-1,x+1)-2*g(y,x-1)+2*g(y,x+1)-1*g(y+1,x-1)+1*g(y+1,x+1))
            gy = (-1*g(y-1,x-1)-2*g(y-1,x)-1*g(y-1,x+1)+1*g(y+1,x-1)+2*g(y+1,x)+1*g(y+1,x+1))
            mag = math.sqrt(gx*gx + gy*gy)
            if mag > 30:
                intensities.append(mag)

    if not intensities:
        return {"edge_count": 0, "edge_pct": 0, "texture": "smooth"}

    avg_edge = sum(intensities) / len(intensities)
    edge_pct = len(intensities) / (w * h)
    return {
        "edge_count": len(intensities),
        "edge_pct": round(edge_pct, 3),
        "edge_intensity_avg": round(avg_edge, 1),
        "texture": "rough/grainy" if avg_edge > 100 else ("moderate" if avg_edge > 50 else "smooth"),
    }


def foreground_mask(w, h, pixels):
    """背景分离: Otsu阈值 + 前景分析"""
    grays = [sum(pixels[y][x])/3 for y in range(h) for x in range(w)]

    # 简化Otsu: 取中值作为阈值
    sorted_g = sorted(grays)
    threshold = sorted_g[len(sorted_g)//2] if sorted_g else 128

    fg_pixels = []
    fg_y, fg_x = [], []
    for y in range(h):
        for x in range(w):
            g = sum(pixels[y][x]) / 3
            if abs(g - threshold) > 30:  # 偏离背景阈值的为前景
                fg_pixels.append(pixels[y][x])
                fg_y.append(y); fg_x.append(x)

    if not fg_pixels:
        return {"fg_pct": 0, "aspect_ratio": 0, "compactness": 0, "shape": "unknown"}

    fg_pct = len(fg_pixels) / (w * h)
    # 边界框
    min_x, max_x = min(fg_x), max(fg_x)
    min_y, max_y = min(fg_y), max(fg_y)
    bw, bh = max_x - min_x + 1, max_y - min_y + 1

    # 宽高比
    ar = bw / max(bh, 1)
    # 紧凑度: 前景面积 / 边界框面积 (圆的紧凑度 = pi/4 = 0.785)
    compactness = len(fg_pixels) / max(bw * bh, 1)

    # 形状推断
    if 0.9 < ar < 1.1 and 0.6 < compactness < 0.9:
        shape = "round/circular"
    elif ar > 2.0:
        shape = "elongated/horizontal"
    elif ar < 0.5:
        shape = "tall/vertical"
    elif 0.7 < compactness < 0.98:
        shape = "compact/blob"
    else:
        shape = "irregular"

    return {
        "fg_pct": round(fg_pct, 3),
        "bbox_wh_ratio": round(ar, 2),
        "compactness": round(compactness, 3),
        "shape": shape,
    }


def texture_analysis(w, h, pixels):
    """纹理分析: 局部对比度 + 色块均匀度"""
    # 采样中心区域 (避免边界干扰)
    contrasts = []
    for y in range(2, h-2, 3):
        for x in range(2, w-2, 3):
            r1,g1,b1 = pixels[y][x]
            r2,g2,b2 = pixels[y+2][x+2]
            diff = abs(r1-r2) + abs(g1-g2) + abs(b1-b2)
            contrasts.append(diff)

    if not contrasts: return {"local_contrast": 0, "surface": "unknown"}
    avg_contrast = sum(contrasts) / len(contrasts)

    return {
        "local_contrast_avg": round(avg_contrast, 1),
        "surface": "speckled/textured" if avg_contrast > 80 else ("gradient" if avg_contrast > 30 else "uniform/smooth"),
    }


# =============================================================
# Step 3: 查表 (不变, 可扩展)
# =============================================================
OBJECT_DB = [
    {"name": "apple", "hue_match": ["red","orange","yellow","green","pink/magenta"],
     "shape_hint": "round/circular", "surface_hint": "speckled/textured",
     "note": "圆形或多边形, 表面有斑点或光泽, 可能有茎"},
    {"name": "banana", "hue_match": ["yellow","green"],
     "shape_hint": "elongated/horizontal", "surface_hint": "gradient",
     "note": "弯曲长条形, 黄色或青色, 表面有渐变"},
    {"name": "leaf", "hue_match": ["green"], "shape_hint": "elongated/horizontal",
     "surface_hint": "speckled/textured", "note": "绿色, 扁平, 有叶脉纹理"},
    {"name": "sun/sunflower", "hue_match": ["yellow","orange"], "shape_hint": "round/circular",
     "surface_hint": "gradient", "note": "圆形, 亮黄或橙色, 有放射纹理"},
    {"name": "mug/cup", "hue_match": ["mixed","red","blue/purple","cyan/blue"],
     "shape_hint": "tall/vertical", "surface_hint": "uniform/smooth",
     "note": "圆柱形, 可能有把手, 单色表面"},
]

def match_object(features):
    """加权查表匹配"""
    color = features["color"]["dominant_hue"]
    shape = features["fg"]["shape"]
    surface = features["texture"]["surface"]
    matches = []
    for obj in OBJECT_DB:
        score = 0; reasons = []
        if any(h in color for h in obj["hue_match"]) or color in obj["hue_match"]:
            score += 30; reasons.append("色相")
        if obj["shape_hint"] in shape or shape in obj["shape_hint"]:
            score += 30; reasons.append("形状")
        if obj["surface_hint"] in surface or surface in obj["surface_hint"]:
            score += 20; reasons.append("纹理")
        is_round_circular = ("round" in features["fg"]["shape"])
        is_not_round_circular = obj["shape_hint"] not in ["round/circular"]
        if is_round_circular and is_not_round_circular:
            score -= 10
        if "saturation" in features["color"] and features["color"]["saturation"] == "vivid": score += 10; reasons.append("饱和度")
        matches.append((obj["name"], score, reasons))
    matches.sort(key=lambda x: -x[1])
    return matches


# =============================================================
# Step 4: 组装描述 + 调用 DeepSeek
# =============================================================
def identify(path):
    w, h, pixels = read_bmp(path)
    color = color_histogram(w, h, pixels)
    edges = edge_intensity(w, h, pixels)
    fg = foreground_mask(w, h, pixels)
    texture = texture_analysis(w, h, pixels)

    features = {"color": color, "edges": edges, "fg": fg, "texture": texture}
    matches = match_object(features)

    description = f"""64x64图片通用特征报告:

色彩(HSV): 主色调{color['dominant_hue']}, 饱和度{color['saturation']}, 亮度{color['brightness']}
  HSV均值(H,S,V): {color['hsv_mean']}, 标准差: {color['hsv_std']}
边缘: {edges['edge_count']}个边缘像素({edges['edge_pct']*100:.1f}%), 强度{edges['edge_intensity_avg']}, 纹理{edges['texture']}
前景: 占比{fg['fg_pct']*100:.0f}%, 宽高比{fg['bbox_wh_ratio']}, 紧凑度{fg['compactness']}, 形状{fg['shape']}
表面纹理: {texture['surface']}, 局部对比度{texture['local_contrast_avg']}

查表候选(最高3):
  {matches[0][0]}({matches[0][1]}分: {'+'.join(matches[0][2])})
  {matches[1][0] if len(matches)>1 else '无'}({matches[1][1] if len(matches)>1 else 0}分)
  {matches[2][0] if len(matches)>2 else '无'}({matches[2][1] if len(matches)>2 else 0}分)

请推理: 解读特征→逐个比对→加权评分→最终结论。"""

    resp = client.chat.completions.create(
        model="deepseek-chat", temperature=0.3, max_tokens=300,
        messages=[
            {"role": "system", "content": "你是基于HSV色彩+边缘+前景+纹理特征进行视觉推理的引擎。输出逻辑链→逐个比对→结论。物体名称+置信度%。"},
            {"role": "user", "content": description}
        ]
    )
    return resp.choices[0].message.content, description, matches


# =============================================================
# 主流程
# =============================================================
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test_apple.bmp"
    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        sys.exit(1)
    print(f"分析: {path}\n")
    result, desc, matches = identify(path)
    print(f"[特征提取]\n{desc}\n")
    print(f"[DeepSeek 识别]\n{result}")
