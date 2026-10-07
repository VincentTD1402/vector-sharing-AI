"""Demo 05 - Chunking strategy và ảnh hưởng tới retrieval (slide 6, 7, 23, 25).

A. Xem một sổ tay bị cắt ra sao theo từng chiến lược.
B. Thí nghiệm: đổi chunk size / overlap / chiến lược, đo Recall@1, Recall@3, MRR và số token đưa cho LLM.
   - Hit = trong top-k chunk có chunk CHỨA câu trả lời (answer string).

Chạy:  python demos/05_chunking.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.chunking import count_tokens, fixed_size_chunks, recursive_chunks, sentence_chunks
from lib.chunking_eval import K, sweep
from lib.data import load_manual_questions, load_manuals
from lib.viz import banner


def preview() -> None:
    banner("A. Cùng một văn bản, các cách chia khác nhau", "6")
    name, text = next(iter(load_manuals().items()))
    print(f"Tài liệu: {name}  ({count_tokens(text)} token)\n")
    for label, chunks in {
        "fixed 128 token, overlap 0 ": fixed_size_chunks(text, 128, 0),
        "fixed 128 token, overlap 32": fixed_size_chunks(text, 128, 32),
        "sentence (<=128 token)     ": sentence_chunks(text, 128),
        "recursive (<=128 token)    ": recursive_chunks(text, 128),
    }.items():
        print(f"- {label}: {len(chunks)} chunk, token/chunk = {[count_tokens(c) for c in chunks]}")
    first = fixed_size_chunks(text, 128, 0)
    print("\nRanh giới chunk fixed-size thường cắt GIỮA câu:")
    print(f"  ...{first[0][-70:]!r}")
    print(f"  {first[1][:70]!r}...")
    print("\nOverlap giữ lại ngữ cảnh ở ranh giới (chunk sau lặp lại phần cuối chunk trước):")
    ov = fixed_size_chunks(text, 128, 32)
    print(f"  cuối chunk 0: ...{ov[0][-60:]!r}\n  đầu chunk 1 : {ov[1][:60]!r}...")


def experiment() -> None:
    banner(f"B. Thí nghiệm chunk size trên {len(load_manual_questions())} câu hỏi (3 sổ tay, Việt + Anh)", "23, 25")
    print(f"  {'Cấu hình':<28} {'#chunk':>6} {'avg tok':>7} {'R@1':>5} {'R@'+str(K):>5} {'MRR':>5} {'ctx tok (top-'+str(K)+')':>16}")
    for r in sweep():
        print(f"  {r['label']:<28} {r['chunks']:>6} {r['avg_tokens']:>7.0f} {r['recall@1']:>5.2f} "
              f"{r[f'recall@{K}']:>5.2f} {r['mrr']:>5.2f} {r['ctx_tokens']:>16.0f}")
    print("\nCách đọc: R@1/MRR cho biết chunk đúng có lên đầu không; 'ctx tok' là số token phải đưa cho LLM")
    print("(chunk lớn tốn context và nhiễu hơn). Corpus demo chỉ 3 tài liệu nên chênh lệch nhỏ;")
    print("kết luận 'size nào tốt nhất' chỉ có giá trị khi đo trên dữ liệu thật của bạn.")


if __name__ == "__main__":
    preview()
    experiment()
