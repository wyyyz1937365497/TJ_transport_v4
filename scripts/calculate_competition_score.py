#!/usr/bin/env python3
"""
比赛得分计算器

用法：
    python scripts/calculate_competition_score.py --ocr_ai 0.5439 --ocr_baseline 0.5469
"""

import argparse
import json
from pathlib import Path


def load_baseline_results():
    """加载baseline结果"""
    baseline_file = Path("competition_results/baseline_results.json")

    if baseline_file.exists():
        with open(baseline_file, 'r') as f:
            return json.load(f)
    else:
        return None


def compute_competition_score(ocr_ai, ocr_baseline, preliminary=True):
    """
    计算比赛得分

    Args:
        ocr_ai: AI模型的OCR
        ocr_baseline: Baseline的OCR
        preliminary: 是否初赛

    Returns:
        得分字典
    """
    # 计算相对提升
    delta_ocr = (ocr_ai - ocr_baseline) / max(ocr_baseline, 1e-6)

    # 效率得分
    efficiency_score = 100.0 * max(0.0, delta_ocr)

    # 初赛只看效率
    if preliminary:
        stability_score = 0.0
        total_score = efficiency_score
    else:
        # 复赛需要计算稳定性（这里简化）
        stability_score = 0.0
        total_score = efficiency_score

    return {
        'ocr_ai': ocr_ai,
        'ocr_baseline': ocr_baseline,
        'delta_ocr': delta_ocr,
        'ocr_improvement_percent': delta_ocr * 100,
        'efficiency_score': efficiency_score,
        'stability_score': stability_score,
        'total_score': total_score,
        'is_positive': delta_ocr > 0
    }


def print_score_report(score_info):
    """打印得分报告"""
    print("\n" + "=" * 70)
    print("🏆 比赛得分报告")
    print("=" * 70)

    print("\n📊 OCR对比:")
    print(f"  AI模型OCR:  {score_info['ocr_ai']:.4f} ({score_info['ocr_ai']*100:.2f}%)")
    print(f"  Baseline:   {score_info['ocr_baseline']:.4f} ({score_info['ocr_baseline']*100:.2f}%)")
    print(f"  差异:       {score_info['delta_ocr']:+.4f} ({score_info['ocr_improvement_percent']:+.2f}%)")

    print("\n🏅 得分详情:")
    print(f"  效率得分: {score_info['efficiency_score']:.2f}")
    print(f"  稳定性得分: {score_info['stability_score']:.2f}")
    print(f"  总分: {score_info['total_score']:.2f}")

    if score_info['is_positive']:
        print("\n✅ 模型优于baseline，可以提交！")
    else:
        print("\n❌ 模型不优于baseline，得分为0！")
        print("   建议：查看SCORE_ANALYSIS.md获取改进方案")

    print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="比赛得分计算器")
    parser.add_argument('--ocr_ai', type=float, help='AI模型的OCR')
    parser.add_argument('--ocr_baseline', type=float, help='Baseline的OCR')
    parser.add_argument('--from_baseline_file', action='store_true',
                        help='从baseline_results.json读取baseline OCR')
    parser.add_argument('--from_eval_file', type=str,
                        help='从评估结果JSON文件读取AI模型OCR')

    args = parser.parse_args()

    # 加载baseline结果
    if args.from_baseline_file or args.ocr_baseline is None:
        baseline_results = load_baseline_results()
        if baseline_results:
            args.ocr_baseline = baseline_results['ocr']
            print(f"✓ 从baseline_results.json读取baseline OCR: {args.ocr_baseline:.4f}")
        else:
            print("❌ 未找到baseline_results.json，请使用--ocr_baseline指定")
            return

    # 加载评估结果
    if args.from_eval_file or args.ocr_ai is None:
        if args.from_eval_file:
            eval_file = Path(args.from_eval_file)
            if eval_file.exists():
                with open(eval_file, 'r') as f:
                    eval_results = json.load(f)
                    args.ocr_ai = eval_results.get('metrics', {}).get('mean_ocr', 0.0)
                print(f"✓ 从{args.from_eval_file}读取AI OCR: {args.ocr_ai:.4f}")

    # 检查参数
    if args.ocr_ai is None or args.ocr_baseline is None:
        print("❌ 请指定OCR值：")
        print("   方式1: --ocr_ai 0.5439 --ocr_baseline 0.5469")
        print("   方式2: --from_baseline_file --ocr_ai 0.5439")
        print("   方式3: --from_baseline_file --from_eval_file logs/ocr_max/evaluation/stage2_ppo_iter99_results.json")
        return

    # 计算得分
    score_info = compute_competition_score(args.ocr_ai, args.ocr_baseline)

    # 打印报告
    print_score_report(score_info)

    # 保存报告
    output_file = Path("competition_results/score_report.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(score_info, f, indent=2)

    print(f"✓ 得分报告已保存到: {output_file}\n")


if __name__ == '__main__':
    main()
