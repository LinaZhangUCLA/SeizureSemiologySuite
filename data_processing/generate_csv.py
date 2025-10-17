import sys
from pathlib import Path
import pandas as pd
import re
import csv

# 识别 Excel XML 转义：_xNNNN_（大小写均可）
EXCEL_XML_ESC = re.compile(r'_x([0-9A-Fa-f]{4})_', flags=re.IGNORECASE)
# 其它不可见控制字符（保留 \t \n \r 的位置我们后面单独处理）
CTRL_CHARS = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')

def clean_text(s: object, keep_newline=False) -> str:
    """清理 Excel 转义、控制字符、零宽字符等。
    keep_newline=True 时把换行保留为真正的 \n（CSV 会用引号包住能正确显示多行单元格）；
    否则统一压成空格。
    """
    if s is None:
        return ""
    s = str(s)

    # 1) 解码 _xNNNN_（例如 _x000D_/_x000d_）
    def _decode(m):
        ch = chr(int(m.group(1), 16))
        # 回车/换行/制表符单独处理
        if ch in ("\r", "\n", "\t"):
            return ch
        return ch
    s = EXCEL_XML_ESC.sub(_decode, s)

    # 2) 去掉 BOM/零宽字符
    s = (s.replace("\ufeff", "")
           .replace("\u200b", "")
           .replace("\u200c", "")
           .replace("\u200d", "")
           .replace("\u2060", ""))

    # 3) 规范换行与制表符
    if keep_newline:
        # 保留换行，去掉回车，制表符转空格
        s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    else:
        # 全部压成空格
        s = s.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("\t", " ")

    # 4) 去掉其余控制字符
    s = CTRL_CHARS.sub("", s)

    # 5) 折叠多余空白
    if keep_newline:
        # 在保留换行的前提下，逐行折叠空白
        s = "\n".join(re.sub(r"\s+", " ", line).strip() for line in s.splitlines())
    else:
        s = re.sub(r"\s+", " ", s).strip()

    return s


def load_table(path: Path) -> pd.DataFrame:
    # 读入
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    lower_map = {c.lower(): c for c in df.columns}

    def pick(col_key: str) -> str:
        if col_key in df.columns:
            return col_key
        if col_key.lower() in lower_map:
            return lower_map[col_key.lower()]
        raise KeyError(f"找不到列 {col_key}。可用列有：{list(df.columns)}")

    c1 = pick("file_name")
    c2 = pick("Behavioral description")


    out = df[[c1, c2]].copy()
    out[c2] = out[c2].astype(str).map(lambda x: clean_text(x, keep_newline=False))
    out.columns = ["file_name", "report"]

   



    mask = out["file_name"].notna() & out["file_name"].astype(str).str.strip().ne("")
    out = out[mask]

  
    # mask = out["90_file_name"].notna() & out["90_file_name"].astype(str).str.strip().ne("")
    # out = out[mask]

    return out

def main(input_path: str, output_path: str = "new_old_name_map.csv"):
    in_path = Path(input_path)
    out_df = load_table(in_path)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"已保存：{output_path}（{len(out_df)} 行）")

if __name__ == "__main__":
    # 用法：
    # python script.py your_file.csv
    # 或
    # python script.py your_file.xlsx new_old_name_map.csv
    input_path = sys.argv[1] if len(sys.argv) > 1 else "/home/lina/icassp2026/ICASSP2026/evaluation/audio/final_video_df.xlsx"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "/home/lina/ssb/SeizureSemiologyBench/result/ground_truth/task6_annotation.csv"
    main(input_path, output_path)
