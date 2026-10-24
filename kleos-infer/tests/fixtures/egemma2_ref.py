"""Reference embeddings for google/embeddinggemma-2 (text path) from
sentence-transformers, for kleos-infer's parity test.

    HF_HOME=... python egemma2_ref.py out.json
"""
import json, sys, random
import torch
from sentence_transformers import SentenceTransformer

def texts():
    base = [
        "I moved to Lisbon last spring and started a job at a small design studio.",
        "What did Caroline say about her sister's new apartment?",
        "The quarterly report shows margins fell 3% because of shipping costs.",
        "Mitochondria are the powerhouse of the cell.",
        "",
        "a",
        "Café déjà vu — naïve façade, Zürich, São Paulo, Ærøskøbing.",
        "東京で新しい仕事を始めました。来月から大阪に引っ越します。",
        "Я переехал в Берлин в прошлом году.",
        "مرحبا، كيف حالك اليوم؟",
        "emoji test 🚀🔥👩‍💻 and mixed 中文 English العربية",
        "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n",
        "SELECT name, count(*) FROM users GROUP BY name HAVING count(*) > 1;",
        "fn main() { let v: Vec<u32> = (0..10).map(|x| x * x).collect(); println!(\"{v:?}\"); }",
        "    leading and trailing whitespace    ",
        "New\nlines\n\nand\ttabs\there.",
        "UPPERCASE SHOUTING ABOUT THE WEATHER IN DECEMBER",
        "https://example.com/path?query=1&other=two#frag user@example.org",
        "1 2 3 4 5 6 7 8 9 10 3.14159 -42 1e10",
        "The patient reported mild headaches after taking 200mg ibuprofen twice daily.",
    ]
    rng = random.Random(7)
    words = ("memory agent store recall fact update user sister Berlin Lisbon project meeting "
             "deadline report coffee train weekend garden book music river mountain code test "
             "database vector graph entity ontology document page table chart").split()
    for n in (60, 200, 500, 1000, 2000):
        base.append(" ".join(rng.choice(words) for _ in range(n)) + ".")
    para = ("Agents need durable memory. They remember facts, notice when facts change, and recall "
            "what matters when asked. ")
    for reps in (40, 120, 300, 520):
        base.append(para * reps)
    while len(base) < 50:
        i = len(base)
        base.append(f"Sentence number {i}: the {rng.choice(words)} and the {rng.choice(words)} met at {rng.randint(1,12)} o'clock.")
    return base

def main(out):
    m = SentenceTransformer("google/embeddinggemma-2", device="cpu",
                            model_kwargs={"torch_dtype": torch.float32, "attn_implementation": "eager"})
    tok = m.tokenizer if hasattr(m, "tokenizer") else None
    xs = texts()
    res = {"texts": xs, "prompts": {"document": m.prompts["document"], "query": m.prompts["query"]}}
    for kind in ("document", "query"):
        e = m.encode(xs, prompt_name=kind, batch_size=4, convert_to_numpy=True, normalize_embeddings=True)
        res[kind] = [[round(float(v), 7) for v in row] for row in e]
    # token ids exactly as sentence-transformers fed them
    feats = m.preprocess([m.prompts["document"] + t for t in xs[:6]])
    res["ids_document_first6"] = [[int(t) for t, a in zip(r, mk) if a] for r, mk in zip(feats["input_ids"].tolist(), feats["attention_mask"].tolist())]
    res["max_seq_length"] = m.max_seq_length
    json.dump(res, open(out, "w"))
    print("wrote", out, len(xs), "texts; max_seq_length", m.max_seq_length)

if __name__ == "__main__":
    main(sys.argv[1])
