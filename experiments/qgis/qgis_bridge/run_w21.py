"""
QGIS W21：自然语言指令 → 工作流参数（表达式），留出句式/取值测泛化

背景：W18/W19 的"选择+导出"工作流参数是写死的（'"area" > 300'）。本轮补上语言层：
    "把面积大于 400 的区域导出"        → '"area" > 400'
    "导出名称是 West 的区域"           → '"name" = 'West''
    "挑出面积不超过 250 的区"          → '"area" <= 250'
    "把不是 North 的区域导出"          → '"name" != 'North''

两条通道对照（沿用 W3 的教训框架）：
  · lexical：实验者手写的算子词表 + 数字/名字抽取（"结构"臂，词表覆盖训练与测试用词）
  · learned：字符级小 Transformer，只在**训练句式**上训练
评测：**已知句式/新取值** vs **未见句式**（新算子措辞）的精确匹配率。

用法:
    python experiments/qgis/qgis_bridge/run_w21.py            # 离线：数据+训练+评测
    python experiments/qgis/qgis_bridge/run_w21.py --e2e      # 追加：真 QGIS 端到端（需 QGIS 在跑）
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
RES = HERE / "w21_results.json"
CKPT = HERE / "w21_model.pt"

# ---------------- 语料 ----------------
NAMES = ["North", "South", "Central", "West", "East"]
AREA_VALS = list(range(100, 951, 50))

# (算子, 训练措辞, 测试措辞)  —— 测试措辞在训练中完全未出现
OPS = [
    ("area", ">",  ["大于", "超过"]),
    ("area", "<",  ["小于", "低于"]),
    ("area", ">=", ["不小于", "至少"]),
    ("area", "<=", ["不大于", "至多"]),
    ("name", "=",  ["是", "叫", "名为"]),
    ("name", "!=", ["不是", "不叫"]),
]

TRAIN_TPL = [
    ("把面积{op}{v}的区域导出", "area"),
    ("导出面积{op}{v}的区域", "area"),
    ("挑出面积{op}{v}的区", "area"),
    ("把名称{nop}{name}的区域导出", "name"),
    ("导出{nop}{name}的区域", "name"),
    ("挑出名称{nop}{name}的区", "name"),
]
TEST_TPL = [                      # 未见句式（同样的语义槽位，不同的句子结构）
    ("凡是面积{op}{v}的区都导出来", "area"),
    ("请导出那些面积{op}{v}的区域", "area"),
    ("名称{nop}{name}的区请导出", "name"),
    ("请挑出名称{nop}{name}的区域导出", "name"),
]


def expr_of(field, op, val):
    if field == "area":
        return f'"{field}" {op} {val}'
    return f'"{field}" {op} \'{val}\''


def gen(split, rng, n):
    rows = []
    for _ in range(n):
        field, op, words = OPS[int(rng.integers(0, len(OPS)))]
        skip = 1 if split == "test" else 0
        op_word = words[skip]
        if field == "area":
            v = int(AREA_VALS[int(rng.integers(0, len(AREA_VALS)))])
            tpls = TEST_TPL if split == "test" else TRAIN_TPL
            tpls = [t for t in tpls if t[1] == "area"]
            tpl = tpls[int(rng.integers(0, len(tpls)))][0]
            instr = tpl.format(op=op_word, v=v)
            target = expr_of(field, op, v)
        else:
            v = NAMES[int(rng.integers(0, len(NAMES)))]
            tpls = TEST_TPL if split == "test" else TRAIN_TPL
            tpls = [t for t in tpls if t[1] == "name"]
            tpl = tpls[int(rng.integers(0, len(tpls)))][0]
            instr = tpl.format(nop=op_word, name=v)
            target = expr_of(field, op, v)
        rows.append((instr, target))
    return rows


# ---------------- 词法基线（"结构"臂） ----------------
LEX = {
    "大于": ">", "超过": ">",
    "小于": "<", "低于": "<",
    "不小于": ">=", "至少": ">=", "不低于": ">=",
    "不大于": "<=", "至多": "<=", "不超过": "<=",
    "不是": "!=", "不叫": "!=",
    "是": "=", "叫": "=", "名为": "=",
}


def lexical(instr):
    # 先长词后短词，避免"不小于"被"小于"抢先
    for w in sorted(LEX, key=len, reverse=True):
        if w in instr:
            op = LEX[w]
            m = re.search(r"\d+", instr)
            if op.startswith(">") or op.startswith("<"):
                if not m:
                    return None
                return f'"area" {op} {int(m.group())}'
            nm = None
            for n in NAMES:
                if n in instr:
                    nm = n
            if nm is None:
                return None
            return f'"name" {op} \'{nm}\''
    return None


# ---------------- 模型 ----------------
class Tiny(nn.Module):
    def __init__(s, vocab, d=128, L=4, H=4, maxlen=64):
        super().__init__()
        s.tok = nn.Embedding(vocab, d)
        s.pos = nn.Embedding(maxlen, d)
        layer = nn.TransformerEncoderLayer(d, H, 4 * d, batch_first=True, dropout=0.0)
        s.enc = nn.TransformerEncoder(layer, L, enable_nested_tensor=False)
        s.head = nn.Linear(d, vocab)
        s.maxlen = maxlen

    def forward(s, x):
        n, T = x.shape
        h = s.tok(x) + s.pos(torch.arange(T, device=x.device))[None]
        m = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        return s.head(s.enc(h, mask=m))


def build_vocab(rows):
    chars = set()
    for a, b in rows:
        chars |= set(a) | set(b)
    chars |= {"\n"}
    itos = ["<pad>", "<eos>"] + sorted(chars)
    return {c: i for i, c in enumerate(itos)}, itos


def encode(instr, target, stoi, maxlen=64):
    # 注意：EOS 必须作为 token id 追加，不能拼 "<eos>" 字面文本
    #（表达式里的 < > 会让这两个字符进词表，字面拼接会被模型学着逐字生成）
    s = instr + "\n" + target
    ids = [stoi.get(c, 1) for c in s][:maxlen - 1] + [1]
    return ids


def batchify(rows, stoi, maxlen=64):
    xs = []
    for a, b in rows:
        ids = encode(a, b, stoi, maxlen)
        xs.append(ids)
    T = max(len(x) for x in xs)
    X = torch.full((len(xs), T), 0, dtype=torch.long)
    for i, x in enumerate(xs):
        X[i, :len(x)] = torch.tensor(x)
    return X


@torch.no_grad()
def predict(model, instr, stoi, itos, maxlen=64):
    model.eval()
    ids = [stoi.get(c, 1) for c in instr + "\n"][:maxlen]
    x = torch.tensor(ids)[None]
    out = []
    for _ in range(maxlen - len(ids)):
        logits = model(x)[0, -1]
        nxt = int(logits.argmax())
        if nxt == 1:                                   # <eos>
            break
        out.append(itos[nxt])
        x = torch.cat([x, torch.tensor([[nxt]])], 1)
    return "".join(out)


def main():
    e2e = "--e2e" in sys.argv
    rng = np.random.default_rng(0)
    train = gen("train", rng, 2400)
    test_seen = gen("test", rng, 0)                    # 占位
    # 三种测试集：已知句式+已知取值 / 已知句式+新取值 / 未见句式
    rng2 = np.random.default_rng(1)
    test_known = gen("train", rng2, 300)
    test_newval = [("把面积大于 %d 的区域导出" % v, expr_of("area", ">", v))
                   for v in (975, 1025, 1100)]          # 名字是封闭词表（5 个区），新取值只测数字
    test_novel = gen("test", rng2, 300)
    allrows = train + test_known + test_novel
    stoi, itos = build_vocab(allrows)
    print(f"语料：train {len(train)}，已知句式测试 {len(test_known)}，未见句式 {len(test_novel)}，"
          f"词表 {len(itos)}")
    print("样例：", train[0], "|", test_novel[0])

    # 词法基线
    res = {}
    for nm, rows in (("known_tpl", test_known), ("new_value", test_newval), ("novel_tpl", test_novel)):
        ok = sum(1 for a, b in rows if lexical(a) == b)
        res[f"lexical_{nm}"] = ok / len(rows)
        print(f"  lexical {nm:9}: {ok}/{len(rows)} = {ok/len(rows):.3f}")

    # 训练（lr 1e-3 + 梯度裁剪，避免发散；按已知句式留出片选最佳）
    torch.manual_seed(0)
    X = batchify(train, stoi)
    model = Tiny(len(itos))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    n = len(train)
    t0 = time.time()
    best = (-1.0, None)
    for ep in range(30):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 64):
            idx = perm[i:i+64]
            xb = X[idx]
            logits = model(xb[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, len(itos)), xb[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(idx)
        if (ep + 1) % 5 == 0:
            acc = sum(1 for a, b in test_known[:100] if predict(model, a, stoi, itos) == b) / 100
            print(f"  epoch {ep+1:2}: loss {tot/n:.4f} 已知句式 acc {acc:.2f}  ({time.time()-t0:.0f}s)")
            if acc > best[0]:
                best = (acc, {k: v.clone() for k, v in model.state_dict().items()})
    if best[1] is not None:
        model.load_state_dict(best[1])

    for nm, rows in (("known_tpl", test_known), ("new_value", test_newval), ("novel_tpl", test_novel)):
        ok = sum(1 for a, b in rows if predict(model, a, stoi, itos) == b)
        res[f"learned_{nm}"] = ok / len(rows)
        print(f"  learned {nm:9}: {ok}/{len(rows)} = {ok/len(rows):.3f}")
        if nm == "novel_tpl":
            for a, b in rows[:5]:
                print(f"     {a!r} -> {predict(model, a, stoi, itos)!r}（应为 {b!r}）")

    torch.save({"sd": model.state_dict(), "stoi": stoi, "itos": itos}, CKPT)
    RES.write_text(json.dumps({"n_train": len(train), "vocab": len(itos), "res": res},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 experiments/qgis/qgis_bridge/w21_results.json 与 w21_model.pt")

    if e2e:
        run_e2e(model, stoi, itos)


# ---------------- 端到端（真 QGIS） ----------------
def run_e2e(model, stoi, itos):
    sys.path.insert(0, str(HERE))
    from client import call
    from run_w18 import select_via_gui_retry, SCENE
    from run_w16 import export_via_gui, features_of
    import numpy as np2

    OUT = HERE / "data" / "w21.geojson"
    # 真值：对 5 个区域求值
    FEATS = {"North": 500, "South": 400, "Central": 300, "West": 150, "East": 120}

    def expected(instr, expr=None):
        e = expr if expr is not None else predict(model, instr, stoi, itos)
        m = re.match(r'"(\w+)"\s*(>=|<=|!=|=|>|<)\s*([\w\']+)', e)
        if not m:
            return e, None
        f, op, v = m.group(1), m.group(2), m.group(3).strip("'")
        def keep(n):
            a = FEATS[n]
            try:
                vv = int(v)
                return {">": a > vv, "<": a < vv, ">=": a >= vv, "<=": a <= vv}.get(op, False)
            except ValueError:
                return {"=": n == v, "!=": n != v}.get(op, False)
        return e, {n for n in FEATS if keep(n)}

    # 每例给**指令语义**的手写真值（不能用模型输出的表达式反推，否则错表达式也能"通过"）
    tests = [
        ("把面积大于400的区域导出", {"North"}),                       # 500>400
        ("挑出面积不超过250的区", {"West", "East"}),                   # 150,120 <= 250
        ("凡是面积至少300的区都导出来", {"North", "South", "Central"}),  # 500,400,300 >= 300
        ("名称不是West的区请导出", {"North", "South", "Central", "East"}),
    ]
    print("\n端到端（真 QGIS；learned 与 lexical 两条 NL→参数 通道对照；真值按指令语义手写）：")
    rows = []
    for arm in ("learned", "lexical"):
        for instr, want in tests:
            expr = predict(model, instr, stoi, itos) if arm == "learned" else lexical(instr)
            call({"op": "reset", "scene": SCENE})
            time.sleep(0.3)
            ok_sel, st_sel, first, tries = select_via_gui_retry(expr, expect_n=len(want))
            if OUT.exists():
                OUT.unlink()
            ok_exp, st_exp, note = export_via_gui(OUT, selected_only=True)
            got = features_of(OUT)
            good = bool(ok_sel and ok_exp and got == want)
            rows.append(dict(arm=arm, instr=instr, expr=expr,
                             want=sorted(want), got=sorted(got), ok=good,
                             sel_ms=st_sel.get("total_ms", 0) if st_sel else 0,
                             exp_ms=st_exp.get("total_ms", 0), note=note))
            print(f"  [{arm:7}] {instr!r}\n     -> {expr!r} 期望 {sorted(want)} "
                  f"实得 {sorted(got)} 成功={good}")
    for arm in ("learned", "lexical"):
        rs = [r for r in rows if r["arm"] == arm]
        print(f"  {arm}: {sum(1 for r in rs if r['ok'])}/{len(rs)}")
    d = json.loads(RES.read_text(encoding="utf-8"))
    d["e2e"] = rows
    RES.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
