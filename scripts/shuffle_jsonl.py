# -*- coding: utf-8 -*-
"""
scripts/shuffle_jsonl.py

把一个 JSONL 文件做一次性的全局随机重排。

实现要点：

* 只在内存中保存每行的字节偏移量（offset），不持有任何 JSON payload；
* 随机化偏移顺序后再去原文件 seek + readline 拷贝出来；
* 输出文件每行内容与输入完全一致，仅顺序变化。

对 BC 训练的意义：
exporter 写出的 ``all_schedules_merged.jsonl`` 默认按 schedule_file 分组排列
（每 500 行属于同一个 solver）。如果直接喂给 IterableDataset，shuffle_buffer
有限的情况下 batch 里几乎只来自 1-2 个 solver，会严重破坏 IID 假设。
预先把整文件 shuffle 一次后再训练，可以让训练曲线稳定下来。

用法::

    python scripts/shuffle_jsonl.py \\
        --input data/trajectories/all_schedules_merged.jsonl \\
        --output data/trajectories/all_schedules_shuffled.jsonl \\
        --seed 0
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # ---------- 第一遍：扫描整文件，记录每行起始位置
    offsets: list[int] = []
    with open(args.input, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            offsets.append(pos)

    # 随机打乱 offset 顺序（确定性 seed 便于复现）
    rng = random.Random(args.seed)
    rng.shuffle(offsets)

    # ---------- 第二遍：按 shuffled offset 顺序复制原始字节流
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input, "rb") as f_in, open(args.output, "wb") as f_out:
        for off in offsets:
            f_in.seek(off)
            line = f_in.readline()
            f_out.write(line)
    print(f"[info] shuffled {len(offsets)} lines -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
