# -*- coding: utf-8 -*-
"""
像素→文本 通用物体识别 批量基准测试
用法: python benchmark_vision.py <测试图片文件夹>
      图片命名格式: <物体名>_<编号>.bmp (如 apple_01.bmp)
"""

import os, sys, time
from openai import OpenAI

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-key-here")
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

sys.path.insert(0, os.path.dirname(__file__))
from pixel_to_text_vision import identify

def run_benchmark(image_dir):
    # 收集测试图片 (命名规则: 物体名_编号.bmp)
    tests = []
    for fname in sorted(os.listdir(image_dir)):
        if fname.endswith('.bmp'):
            label = fname.split('_')[0]  # apple, banana, mug, etc.
            tests.append((fname, label, os.path.join(image_dir, fname)))

    if not tests:
        print("未找到 .bmp 文件。命名格式: apple_01.bmp, banana_02.bmp 等")
        return

    print(f"{'='*60}")
    print(f"  批量基准测试: {len(tests)} 张图片")
    print(f"{'='*60}\n")

    results = []
    correct, total = 0, 0
    confusion = {}  # {actual: {predicted: count}}

    for i, (fname, label, path) in enumerate(tests, 1):
        print(f"[{i}/{len(tests)}] {fname} (实际: {label})")
        try:
            result, _, matches = identify(path)
            # 从最终结论提取预测标签
            predicted = "unknown"
            for line in result.split('\n'):
                line = line.strip().lower()
                if ',' in line and any(kw in line for kw in ['apple','banana','mug','sun','leaf']):
                    predicted = line.split(',')[0].strip()
                    break
            if predicted == "unknown" and matches:
                predicted = matches[0][0]  # fallback: 查表最高分

            ok = (predicted == label)
            if ok: correct += 1
            total += 1
            print(f"    预测: {predicted} | {'[OK]' if ok else '[FAIL]'} | 置信度: {matches[0][1]}分")

            results.append({"file": fname, "actual": label, "predicted": predicted, "ok": ok, "top_match": matches[0]})

            # 混淆矩阵
            confusion.setdefault(label, {})
            confusion[label][predicted] = confusion[label].get(predicted, 0) + 1

        except Exception as e:
            print(f"    [ERROR] {e}")
            results.append({"file": fname, "actual": label, "predicted": "error", "ok": False, "error": str(e)})
            total += 1

        time.sleep(0.3)  # API 礼貌间隔

    # ====================
    # 报告
    # ====================
    print(f"\n{'='*60}")
    print(f"  测试报告")
    print(f"{'='*60}")

    accuracy = correct / max(total, 1) * 100
    print(f"\n准确率: {correct}/{total} = {accuracy:.1f}%")

    # 每类指标
    print(f"\n{'物体':<12} {'样本数':<8} {'正确':<8} {'准确率':<10}")
    print("-" * 40)
    all_labels = sorted(set(r["actual"] for r in results))
    for label in all_labels:
        class_results = [r for r in results if r["actual"] == label]
        class_correct = sum(1 for r in class_results if r["ok"])
        class_rate = class_correct / max(len(class_results), 1) * 100
        print(f"{label:<12} {len(class_results):<8} {class_correct:<8} {class_rate:.1f}%")

    # 混淆矩阵
    print(f"\n混淆矩阵 (行=实际, 列=预测):")
    print(f"{'':<12}", end="")
    for label in all_labels:
        print(f"{label:<12}", end="")
    print(f"{'error':<12}")
    for actual in all_labels:
        print(f"{actual:<12}", end="")
        for pred in all_labels:
            cnt = confusion.get(actual, {}).get(pred, 0)
            print(f"{cnt:<12}", end="")
        err_cnt = sum(1 for r in results if r["actual"] == actual and r["predicted"] == "error")
        print(f"{err_cnt:<12}")

    # 错误明细
    failures = [r for r in results if not r["ok"]]
    if failures:
        print(f"\n错误明细 ({len(failures)}条):")
        for f in failures:
            print(f"  {f['file']}: 实际={f['actual']} 预测={f['predicted']}")

    print(f"\n总API调用: {total} | 准确率: {accuracy:.1f}%")
    return results


if __name__ == "__main__":
    import sys
    img_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    run_benchmark(img_dir)
