# -*- coding: utf-8 -*-
"""
招聘数据中文分词 + TF‑IDF + KMeans 聚类脚本
=============================================
交互式：运行时选择列 → 输入关键词筛选 → 自动聚类
"""

import re
import sys
from pathlib import Path

import jieba
import jieba.analyse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ── 配置 ──────────────────────────────────────────────────────────
INPUT_FILE = r"C:\Users\Lenovo\Documents\xwechat_files\wxid_zfltxsj44uiu12_111a\msg\file\2026-07\yx_recruitment.xlsx"
TOP_N_KEYWORDS = 15
K_RANGE = range(2, 11)
RANDOM_STATE = 42

STOP_WORDS = set("""
的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你
会 着 没有 看 好 自己 这 他 她 它 们 那 里 能 与 及 等 或 其 中
之 而 但 为 对 被 把 从 以 让 所 将 向 并 且 如果 因为 所以 可以
这个 那个 这些 那些 什么 怎么 如何 为什么 哪 谁 何时 何地
包括 进行 通过 使用 利用 具备 拥有 具有 负责 协助 参与 完成
相关 以上 以下 左右 其他 以及 及其 方面 工作 岗位 职位 人员
""".split())


# ── 交互式列选择 ───────────────────────────────────────────────────
def pick_column(df: pd.DataFrame) -> str:
    """让用户选择要分析的列"""
    cols = list(df.columns)
    print("\n📋 数据列列表：")
    for i, c in enumerate(cols):
        # 显示该列的前几个非空样本值作为预览
        samples = df[c].dropna().astype(str).str[:60].tolist()[:3]
        preview = " | ".join(samples) if samples else "(空)"
        print(f"  [{i:2d}] {c:30s} → {preview}")
    print(f"  [{len(cols)}] 自定义列名")

    while True:
        try:
            raw = input("\n👉 请选择列的编号（或直接输入列名）：").strip()
            # 先尝试数字
            if raw.isdigit():
                idx = int(raw)
                if 0 <= idx < len(cols):
                    return cols[idx]
                elif idx == len(cols):
                    # 自定义列名
                    custom = input("👉 请输入列名：").strip()
                    if custom in df.columns:
                        return custom
                    else:
                        print(f"❌ 找不到列 '{custom}'，请重新选择")
                        continue
                else:
                    print(f"❌ 编号 {idx} 超出范围 [0-{len(cols)}]")
                    continue
            else:
                # 直接输入的列名
                if raw in df.columns:
                    return raw
                print(f"❌ 找不到列 '{raw}'，请重新选择")
        except (ValueError, IndexError):
            print("❌ 输入无效，请重新选择")


# ── 关键词过滤（多关键词支持）────────────────────────────────────────
def filter_by_keywords(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """让用户输入关键词，只保留包含任意关键词的行"""
    raw = input(f"\n🔍 请输入筛选关键词（多个词用空格隔开，直接回车则不过滤）：").strip()
    if not raw:
        print("   ✅ 不过滤，使用全部数据")
        return df

    keywords = [kw.strip() for kw in raw.split() if kw.strip()]
    if not keywords:
        return df

    pattern = "|".join(re.escape(kw) for kw in keywords)
    mask = df[col].astype(str).str.contains(pattern, case=False, na=False)
    filtered = df[mask].reset_index(drop=True)
    print(f"   ✅ 关键词匹配，筛选出 {len(filtered)} 行（共 {len(df)} 行）")
    return filtered


# ── 读取数据 ──────────────────────────────────────────────────────
def read_data(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        print(f"❌ 文件不存在: {path}")
        sys.exit(1)
    ext = path.suffix.lower()
    engine = "openpyxl" if ext == ".xlsx" else None
    df = pd.read_excel(path, engine=engine)
    print(f"📄 读取成功：{len(df)} 行 × {len(df.columns)} 列")
    return df


# ── 中文分词 ─────────────────────────────────────────────────────
def segment(text: str) -> str:
    text = re.sub(r"[^一-龥a-zA-Z0-9]", " ", str(text))
    words = jieba.lcut(text)
    filtered = [w for w in words if w.strip() and w not in STOP_WORDS and len(w) > 1]
    return " ".join(filtered)


# ── 自动选 K ────────────────────────────────────────────────────
def find_best_k(vectors, k_range: range) -> tuple[int, dict]:
    results = {}
    inertias = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        labels = km.fit_predict(vectors)
        inertias.append(km.inertia_)

        if k >= 2 and len(set(labels)) > 1:
            sil = silhouette_score(vectors, labels)
        else:
            sil = -1

        results[k] = {"inertia": km.inertia_, "silhouette": sil}
        print(f"   K={k:2d}  inertia={km.inertia_:.2f}  silhouette={sil:.4f}")

    if len(inertias) >= 3:
        deltas = np.diff(inertias)
        accel = np.diff(deltas)
        elbow_k = k_range[2 + int(np.argmax(accel))]
    else:
        elbow_k = k_range[-1]

    sil_scores = {k: v["silhouette"] for k, v in results.items()}
    best_sil_k = max(sil_scores, key=sil_scores.get)

    final_k = best_sil_k
    if abs(sil_scores[elbow_k] - sil_scores[best_sil_k]) < 0.05:
        final_k = min(elbow_k, best_sil_k)

    n_samples = vectors.shape[0]
    final_k = min(final_k, max(2, n_samples // 3))

    print(f"\n✅ Elbow推荐 K={elbow_k}，Silhouette推荐 K={best_sil_k}，最终选定 K={final_k}")
    return final_k, results


# ── 关键词提取 ──────────────────────────────────────────────────────
def extract_keywords(corpus: list[str], top_n: int) -> list[tuple[str, float]]:
    if not corpus:
        return []
    vec = TfidfVectorizer(max_features=5000)
    tfidf = vec.fit_transform(corpus)
    avg_weights = tfidf.mean(axis=0).A1
    idx = avg_weights.argsort()[::-1][:top_n]
    return [(vec.get_feature_names_out()[i], round(avg_weights[i], 4)) for i in idx]


# ── 主流程 ─────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  招聘文本聚类 Pipeline（交互版）")
    print("=" * 55)

    df = read_data(INPUT_FILE)

    # 交互选列
    text_col = pick_column(df)
    print(f"   ✅ 选中列: {text_col}")

    # 丢弃空值
    before = len(df)
    df = df.dropna(subset=[text_col]).reset_index(drop=True)
    print(f"  丢弃 {before - len(df)} 行空值，剩余 {len(df)} 行")

    # 关键词筛选（可选）
    df = filter_by_keywords(df, text_col)
    if len(df) < 5:
        print("⚠️  数据量太少（<5 行），聚类意义不大，仍将继续但效果有限")
    if len(df) == 0:
        print("❌ 筛选后无数据，退出")
        return

    # 分词
    print(f"\n✂️  中文分词中……")
    jieba.setLogLevel(20)
    df["_seg"] = df[text_col].apply(segment)
    print(f"  完成 {len(df)} 条文本分词")

    # TF-IDF
    print(f"\n📊 TF-IDF 向量化……")
    vectorizer = TfidfVectorizer(max_features=5000)
    vectors = vectorizer.fit_transform(df["_seg"])
    print(f"  特征维度: {vectors.shape[1]}")

    # 自动选 K
    actual_k_range = range(2, min(11, len(df)))
    if len(actual_k_range) < 1:
        actual_k_range = range(2, 3)
    print(f"\n🔍 自动确定最佳 K（候选 {list(actual_k_range)}）……")
    best_k, metrics = find_best_k(vectors, actual_k_range)

    # 最终聚类
    print(f"\n🤖 运行 KMeans (K={best_k})……")
    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init="auto")
    df["聚类标签"] = km.fit_predict(vectors)

    # 统计
    print("\n📌 聚类分布：")
    counts = df["聚类标签"].value_counts().sort_index()
    for label, cnt in counts.items():
        print(f"   类 {label}: {cnt} 条 ({cnt/len(df)*100:.1f}%)")

    # 关键词提取（基于聚类结果）
    print(f"\n🏷️  每类 Top-{TOP_N_KEYWORDS} 关键词：")
    cluster_keywords = {}
    cluster_names = {}  # 每类的名称（用前3个关键词拼接）
    for label in sorted(df["聚类标签"].unique()):
        corpus = df[df["聚类标签"] == label]["_seg"].tolist()
        keywords = extract_keywords(corpus, TOP_N_KEYWORDS)
        cluster_keywords[int(label)] = keywords
        words_str = ", ".join(f"{w}({s})" for w, s in keywords)
        print(f"   类 {label}: {words_str}")

        # 用前3个关键词生成类名称
        top3 = [w for w, _ in keywords[:3]]
        cluster_names[int(label)] = " / ".join(top3)

    # 给数据框添加"类名称"列
    df["类名称"] = df["聚类标签"].map(cluster_names)

    # 输出 Excel — 把"类名称"和"聚类标签"放到前两列
    base_name = Path(INPUT_FILE).stem
    out_file = f"{base_name}_聚类结果.xlsx"
    cols = [c for c in df.columns if c not in ("_seg", "类名称", "聚类标签")]
    out_df = df[["类名称", "聚类标签"] + cols]
    out_df.to_excel(out_file, index=False, engine="openpyxl")
    print(f"\n💾 输出: {out_file}")

    # 关键词表
    kw_rows = []
    for label, kws in cluster_keywords.items():
        for rank, (word, score) in enumerate(kws, 1):
            kw_rows.append({"聚类标签": label, "排名": rank, "关键词": word, "TFIDF权重": score})
    kw_df = pd.DataFrame(kw_rows)
    kw_out = f"{base_name}_聚类关键词.xlsx"
    kw_df.to_excel(kw_out, index=False, engine="openpyxl")
    print(f"💾 输出: {kw_out}")

    # 类别中心词云汇总（简洁输出）
    print("\n" + "=" * 55)
    print("  🎯 聚类结果摘要")
    print("=" * 55)
    for label in sorted(df["聚类标签"].unique()):
        top_words = [w for w, _ in cluster_keywords[int(label)][:5]]
        cnt = counts[label]
        print(f"   类 {label} ({cnt}条): {' | '.join(top_words)}")

    print("\n  ✅ 完成！")
    print("=" * 55)


if __name__ == "__main__":
    main()
