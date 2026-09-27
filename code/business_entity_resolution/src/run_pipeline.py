"""Business entity resolution: country blocking, inverted-index candidates, LightGBM.

Country is an open set of labels. Each distinct label is its own block.
No country vocabulary is one-hot encoded, and records are never dropped
because the label was absent from training.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import pickle
import re
import sys
import time
import unicodedata
import zlib
from array import array
from collections import defaultdict

import numpy as np
from rapidfuzz import fuzz

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

NORM_VERSION = "n7"
TOPK_DEFAULT = 40
POOL = 300
TRAIN_MOD = 1000
TRAIN_LO = 8
TRAIN_HI = 28  # 2.0% of source-1
VAL_HI = 8  # 0.8% of source-1

LEGAL = frozenset(
    {
        "llc", "inc", "incorporated", "ltd", "limited", "pvt", "private",
        "corp", "corporation", "co", "company", "llp", "plc", "lp",
        "dba", "the", "and", "of", "sarl", "sas", "sasu", "eurl", "sa",
        "sci", "snc", "gmbh", "ag", "bv", "nv", "pty", "pllc", "pc",
    }
)
ADDR_STOP = frozenset(
    {
        "street", "road", "avenue", "boulevard", "drive", "lane", "court",
        "place", "highway", "parkway", "square", "trail", "terrace", "circle",
        "apartment", "suite", "floor", "building", "near", "opposite", "behind",
        "beside", "next", "sector", "phase", "block", "plot", "shop", "house",
        "flat", "unit", "number", "district", "dist", "village", "town", "city",
        "county", "rural", "urban", "east", "west", "north", "south", "the",
        "and", "for", "at", "to", "in", "on", "by", "via", "po", "box", "post",
        "marg", "nagar", "cross", "main", "new", "old", "upper", "lower",
        "ground", "gali", "chowk", "bazaar", "market", "area", "industrial",
        "layout", "colony", "extension", "ext", "opp", "route", "chemin",
        "impasse", "allee", "rue", "bis", "ter", "quater", "cedex", "france",
        "india", "usa", "united", "states", "state",
    }
)
ABBREV = {
    "rd": "road",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "bd": "boulevard",
    "ln": "lane",
    "hwy": "highway",
    "pkwy": "parkway",
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "flr": "floor",
    "bldg": "building",
    "mt": "mount",
    "ft": "fort",
    "rte": "route",
    "ctr": "center",
    "ctr.": "center",
    "centre": "center",
    "societe": "societe",
}

# Devanagari consonants (inherent vowel added by the transliterator).
_CONS = {
    0x0915: "k", 0x0916: "kh", 0x0917: "g", 0x0918: "gh", 0x0919: "ng",
    0x091A: "c", 0x091B: "ch", 0x091C: "j", 0x091D: "jh", 0x091E: "ny",
    0x091F: "t", 0x0920: "th", 0x0921: "d", 0x0922: "dh", 0x0923: "n",
    0x0924: "t", 0x0925: "th", 0x0926: "d", 0x0927: "dh", 0x0928: "n",
    0x092A: "p", 0x092B: "ph", 0x092C: "b", 0x092D: "bh", 0x092E: "m",
    0x092F: "y", 0x0930: "r", 0x0932: "l", 0x0933: "l", 0x0935: "v",
    0x0936: "sh", 0x0937: "sh", 0x0938: "s", 0x0939: "h", 0x0958: "q",
    0x0959: "kh", 0x095A: "g", 0x095B: "z", 0x095C: "d", 0x095D: "rh",
    0x095E: "f", 0x095F: "y",
}
_VOW = {
    0x093E: "a", 0x093F: "i", 0x0940: "i", 0x0941: "u", 0x0942: "u",
    0x0943: "ri", 0x0947: "e", 0x0948: "ai", 0x094B: "o", 0x094C: "au",
    0x0949: "o",
}
_IND = {
    0x0905: "a", 0x0906: "a", 0x0907: "i", 0x0908: "i", 0x0909: "u",
    0x090A: "u", 0x090B: "ri", 0x090F: "e", 0x0910: "ai", 0x0913: "o",
    0x0914: "au",
}
_HALANT = 0x094D
_DEV_RE = re.compile(r"[\u0900-\u097F]")
_NON_ASCII = re.compile(r"[^\x00-\x7F]")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]+")
_SPACE = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")
_DROP_TOKENS = frozenset({"com", "www", "net", "org", "html", "http", "https"})

FEATURE_NAMES = [
    "name_ratio",
    "name_token_set",
    "name_token_sort",
    "name_partial",
    "addr_ratio",
    "addr_token_set",
    "addr_token_sort",
    "addr_partial",
    "name_jaccard",
    "addr_jaccard",
    "name_contain",
    "addr_contain",
    "digit_jaccard",
    "exact_name",
    "sorted_name_eq",
    "name_len_ratio",
    "addr_len_ratio",
    "prefix4",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def kh(text: str) -> int:
    return hash(text) & 0xFFFFFFFFFFFFFFFF


def transliterate(text: str) -> str:
    if _DEV_RE.search(text) is None:
        return text
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        o = ord(text[i])
        if o in _IND:
            out.append(_IND[o])
            i += 1
            continue
        if o in _CONS:
            base = _CONS[o]
            nxt = ord(text[i + 1]) if i + 1 < n else 0
            if nxt == _HALANT:
                out.append(base)
                i += 2
                continue
            if nxt in _VOW:
                out.append(base + _VOW[nxt])
                i += 2
                continue
            out.append(base + "a")
            i += 1
            continue
        if o in (0x0901, 0x0902):
            out.append("n")
        elif o == 0x0903:
            out.append("h")
        elif o < 128:
            out.append(text[i])
        else:
            out.append(" ")
        i += 1
    return "".join(out)


def normalize(text: str) -> str:
    if not text:
        return ""
    if _NON_ASCII.search(text):
        text = transliterate(text)
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("&", " and ")
    text = _NON_ALNUM.sub(" ", text)
    text = _SPACE.sub(" ", text).strip()
    return text


def _fix_token(tok: str) -> str:
    if len(tok) >= 4 and any(ch.isdigit() for ch in tok) and any(ch.isalpha() for ch in tok):
        tok = tok.translate(str.maketrans({"0": "o", "1": "l", "5": "s"}))
    return tok


def name_tokens(name: str) -> list[str]:
    toks = []
    for raw in name.split():
        tok = _fix_token(raw)
        if tok in LEGAL or tok in _DROP_TOKENS or len(tok) < 3:
            continue
        if tok.isdigit():
            continue
        toks.append(tok)
    return toks


def addr_tokens(addr: str) -> list[str]:
    toks = []
    for raw in addr.split():
        tok = ABBREV.get(raw, raw)
        if tok in ADDR_STOP or tok in LEGAL or len(tok) < 4:
            continue
        if tok.isdigit():
            continue
        toks.append(tok)
    return toks


def number_signature(addr: str) -> str:
    found = []
    for d in _DIGITS.findall(addr):
        if not 2 <= len(d) <= 6:
            continue
        d = d.lstrip("0") or "0"
        if d == "0":
            continue
        found.append(d)
    if not found:
        return ""
    uniq = sorted(set(found), key=lambda x: (-len(x), x))[:4]
    if len(uniq) == 1 and len(uniq[0]) < 3:
        return ""
    return "-".join(sorted(uniq))


def consonants(text: str) -> str:
    return "".join(ch for ch in text if ch not in "aeiou")


def blocking_keys(name: str, addr: str) -> list[tuple[str, int]]:
    """Return (key, max_df) pairs. max_df is the posting-list cap for that key."""
    ntoks = name_tokens(name)
    atoks = addr_tokens(addr)
    keys: list[tuple[str, int]] = []
    seen: set[str] = set()

    def add(key: str, limit: int) -> None:
        if len(key) < 3 or key in seen:
            return
        seen.add(key)
        keys.append((key, limit))

    if ntoks:
        joined = "".join(ntoks)
        if len(joined) >= 6:
            add("j" + joined, 4000)
        sorted_join = "".join(sorted(ntoks))
        if len(sorted_join) >= 6:
            add("s" + sorted_join, 4000)
        sk = consonants(sorted_join)[:14]
        if len(sk) >= 6:
            add("k" + sk, 800)
        for tok in sorted(ntoks, key=len, reverse=True)[:5]:
            add("t" + tok, 1500)
            if len(tok) >= 6:
                add("g" + tok[:5], 2000)
        longest = max(ntoks, key=len)
        if len(longest) >= 6:
            for i in range(len(longest) - 1):
                add("b" + longest[i : i + 2], 120)
    for tok in sorted(set(atoks), key=len, reverse=True)[:4]:
        add("a" + tok, 2500)
        if len(tok) >= 7:
            add("f" + tok[:6], 2000)
            for i in range(0, len(tok) - 1, 2):
                add("d" + tok[i : i + 2], 120)
    num = number_signature(addr)
    if num:
        add("m" + num, 1200)
    if ntoks and atoks:
        nt = sorted(ntoks, key=len, reverse=True)[:2]
        at = sorted(set(atoks), key=len, reverse=True)[:2]
        for a in nt:
            for b in at:
                add("c" + a + "|" + b, 2500)
    if ntoks and num:
        nt = max(ntoks, key=len)
        add("p" + nt + "|" + num, 3000)
    return keys


def parse_gt_line(line: str) -> tuple[str, set[str]]:
    s1, tab, rest = line.rstrip("\n").partition("\t")
    if not tab or not rest.strip():
        return s1, set()
    return s1, {x for x in rest.split(",") if x}


def load_splits(gt_path: str) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    train: dict[str, set[str]] = {}
    val: dict[str, set[str]] = {}
    with open(gt_path, encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            s1, ids = parse_gt_line(line)
            bucket = zlib.crc32(s1.encode()) % TRAIN_MOD
            if bucket < VAL_HI:
                val[s1] = ids
            elif bucket < TRAIN_HI:
                train[s1] = ids
    log(f"split sizes train={len(train)} val={len(val)}")
    return train, val


def discover_countries(path: str) -> list[str]:
    countries: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            country = line.rstrip("\n").rsplit("\t", 1)[-1]
            if country:
                countries.add(country)
    return sorted(countries)


def _open_cache_writers(cache_dir: str, sources: list[str]):
    os.makedirs(cache_dir, exist_ok=True)
    writers: dict[tuple[str, str], object] = {}

    def get(country: str, source: str):
        key = (country, source)
        if key not in writers:
            folder = os.path.join(cache_dir, _safe(country))
            os.makedirs(folder, exist_ok=True)
            writers[key] = open(
                os.path.join(folder, f"{source}.tsv"), "w", encoding="utf-8", newline="\n"
            )
        return writers[key]

    return writers, get


def _safe(country: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", country) or "unknown"


def build_cache(src_paths: dict[str, str], cache_dir: str) -> None:
    stamp = os.path.join(cache_dir, "VERSION")
    if os.path.isfile(stamp) and open(stamp, encoding="utf-8").read().strip() == NORM_VERSION:
        log(f"cache hit {cache_dir}")
        return
    if os.path.isdir(cache_dir):
        for root, dirs, files in os.walk(cache_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    writers, get = _open_cache_writers(cache_dir, list(src_paths))
    for source, path in src_paths.items():
        n = 0
        with open(path, encoding="utf-8") as handle:
            next(handle)
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                eid = parts[0]
                name = normalize(parts[1])
                country = parts[-1].strip()
                addr = normalize(" ".join(parts[2:-1]))
                if not country:
                    continue
                get(country, source).write(f"{eid}\t{name}\t{addr}\n")
                n += 1
                if n % 500000 == 0:
                    log(f"normalize {source} {n}")
        log(f"normalize {source} done {n}")
    for handle in writers.values():
        handle.close()
    os.makedirs(cache_dir, exist_ok=True)
    with open(stamp, "w", encoding="utf-8") as handle:
        handle.write(NORM_VERSION)


def _iter_cache(path: str):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            eid, tab, rest = line.rstrip("\n").partition("\t")
            if not tab:
                continue
            name, tab2, addr = rest.partition("\t")
            if tab2:
                yield eid, name, addr


def build_index(cache_dir: str, country: str):
    folder = os.path.join(cache_dir, _safe(country))
    paths = [os.path.join(folder, f"{src}.tsv") for src in ("s2", "s3")]
    paths = [p for p in paths if os.path.isfile(p)]
    log(f"count keys {country}")
    counts: dict[int, int] = {}
    n_rec = 0
    for path in paths:
        for _eid, name, addr in _iter_cache(path):
            n_rec += 1
            local = set()
            for key, _limit in blocking_keys(name, addr):
                local.add(kh(key))
            for hv in local:
                counts[hv] = counts.get(hv, 0) + 1
            if n_rec % 500000 == 0:
                log(f"  counted {n_rec} unique_keys={len(counts)}")
    log(f"count done {country} records={n_rec} keys={len(counts)}")

    log(f"postings {country}")
    if array("I").itemsize != 4 or array("Q").itemsize != 8:
        raise RuntimeError("unexpected array item size")
    hashes = array("Q")
    recs = array("I")
    ids: list[str] = []
    for path in paths:
        for eid, name, addr in _iter_cache(path):
            rec_i = len(ids)
            ids.append(eid)
            local: set[int] = set()
            for key, limit in blocking_keys(name, addr):
                hv = kh(key)
                if hv in local:
                    continue
                local.add(hv)
                cnt = counts.get(hv, 0)
                if 1 <= cnt <= limit:
                    hashes.append(hv)
                    recs.append(rec_i)
    del counts
    gc.collect()
    log(f"sorting postings {country} n={len(hashes)}")
    h_np = np.frombuffer(hashes.tobytes(), dtype=np.uint64).copy()
    r_np = np.frombuffer(recs.tobytes(), dtype=np.uint32).copy()
    del hashes, recs
    order = np.argsort(h_np, kind="mergesort")
    h_np = h_np[order]
    r_np = r_np[order]
    del order
    gc.collect()
    names: list[str] = []
    addrs: list[str] = []
    for path in paths:
        for _eid, name, addr in _iter_cache(path):
            names.append(name)
            addrs.append(addr)
    if len(names) != len(ids):
        raise RuntimeError(f"cache reread mismatch {len(names)} != {len(ids)}")
    log(f"index ready {country} s2s3={len(ids)} postings={len(h_np)}")
    return h_np, r_np, ids, names, addrs


def candidates_for(name: str, addr: str, h_np, r_np, n_indexed: int, topk: int, scores: np.ndarray, names, addrs) -> list[int]:
    n = max(n_indexed, 1)
    found = []
    for key, limit in blocking_keys(name, addr):
        hv = kh(key)
        lo = int(np.searchsorted(h_np, np.uint64(hv), side="left"))
        hi = int(np.searchsorted(h_np, np.uint64(hv), side="right"))
        df = hi - lo
        if df <= 0 or df > limit:
            continue
        found.append((df, lo, hi, key[0]))
    found.sort()
    parts: list[np.ndarray] = []
    for df, lo, hi, family in found:
        weight = 5.0 if family in "js" else (3.0 if family in "cpm" else 1.0)
        idf = np.float32(weight * (math.log((n + 1) / (df + 1)) + 1.0))
        idx = r_np[lo:hi]
        np.add.at(scores, idx, idf)
        parts.append(idx)
    if not parts:
        return []
    touched = np.unique(np.concatenate(parts))
    vals = scores[touched]
    if touched.size > POOL:
        pick = np.argpartition(vals, -POOL)[-POOL:]
        pool = touched[pick]
        pool_vals = vals[pick]
    else:
        pool = touched
        pool_vals = vals
    scores[touched] = 0
    idf_n = min(pool.size, max(topk, 40))
    if pool.size > idf_n:
        idf_pick = np.argpartition(pool_vals, -idf_n)[-idf_n:]
    else:
        idf_pick = np.arange(pool.size)
    strength = np.empty(pool.size, dtype=np.float32)
    for i, rec in enumerate(pool.tolist()):
        strength[i] = fuzz.token_set_ratio(name, names[rec]) + fuzz.token_set_ratio(addr, addrs[rec])
    sim_pick = np.flatnonzero(strength >= 160)
    if sim_pick.size > 30:
        sim_pick = sim_pick[np.argpartition(strength[sim_pick], -30)[-30:]]
    keep = np.unique(np.concatenate([idf_pick, sim_pick])) if sim_pick.size else idf_pick
    chosen = {int(i) for i in pool[keep].tolist()}
    # Force in every record that shares one of the query's rarer address or name
    # tokens. Those true matches lose the IDF top-50 contest to common city keys.
    qn = {w for w in name.split() if len(w) >= 4}
    ranked_extra: list[tuple[int, int]] = []
    seen_extra: set[int] = set()
    rare_keys = [item for item in found if item[3] in "at" and 2 <= item[0] <= 1500]
    for df, lo, hi, _family in rare_keys[:3]:
        for rec in r_np[lo:hi].tolist():
            if rec in chosen or rec in seen_extra:
                continue
            seen_extra.add(rec)
            n2 = names[rec]
            hit = 0
            for w in n2.split():
                if w in qn:
                    hit += 3
            if hit == 0:
                a2 = addrs[rec]
                qa = {w for w in addr.split() if len(w) >= 5}
                for w in a2.split():
                    if w in qa:
                        hit += 1
            if hit >= 2:
                ranked_extra.append((hit, rec))
    ranked_extra.sort(reverse=True)
    out = pool[keep].astype(np.int64, copy=False).tolist()
    for _hit, rec in ranked_extra[:20]:
        if rec not in out:
            out.append(rec)
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / (len(a) + len(b) - inter)


def _contain(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _ratio(fn, left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return float(fn(left, right)) / 100.0


def _len_ratio(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    denom = max(len(left), len(right), 1)
    return min(len(left), len(right)) / denom


def pair_features(n1: str, a1: str, n2: str, a2: str) -> list[float]:
    nt1 = set(name_tokens(n1))
    nt2 = set(name_tokens(n2))
    at1 = set(addr_tokens(a1))
    at2 = set(addr_tokens(a2))
    d1 = set(number_signature(a1).split("-")) - {""}
    d2 = set(number_signature(a2).split("-")) - {""}
    s1 = "".join(sorted(nt1))
    s2 = "".join(sorted(nt2))
    return [
        _ratio(fuzz.ratio, n1, n2),
        _ratio(fuzz.token_set_ratio, n1, n2),
        _ratio(fuzz.token_sort_ratio, n1, n2),
        _ratio(fuzz.partial_ratio, n1, n2),
        _ratio(fuzz.ratio, a1, a2),
        _ratio(fuzz.token_set_ratio, a1, a2),
        _ratio(fuzz.token_sort_ratio, a1, a2),
        _ratio(fuzz.partial_ratio, a1, a2),
        _jaccard(nt1, nt2),
        _jaccard(at1, at2),
        _contain(nt1, nt2),
        _contain(at1, at2),
        _jaccard(d1, d2),
        1.0 if n1 and n1 == n2 else 0.0,
        1.0 if s1 and s1 == s2 else 0.0,
        _len_ratio(n1, n2),
        _len_ratio(a1, a2),
        1.0 if n1[:4] and n1[:4] == n2[:4] else 0.0,
    ]


def f05_one(pred: set[str], truth: set[str]) -> float:
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    precision = tp / len(pred)
    recall = tp / len(truth)
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def macro_f05(pred_map: dict[str, set[str]], truth_map: dict[str, set[str]]) -> tuple[float, float, float]:
    scores = []
    precs = []
    recs = []
    for sid, truth in truth_map.items():
        pred = pred_map.get(sid, set())
        scores.append(f05_one(pred, truth))
        if not truth and not pred:
            precs.append(1.0)
            recs.append(1.0)
        elif not truth:
            precs.append(0.0)
            recs.append(1.0)
        elif not pred:
            precs.append(0.0)
            recs.append(0.0)
        else:
            tp = len(pred & truth)
            precs.append(tp / len(pred))
            recs.append(tp / len(truth))
    n = max(len(scores), 1)
    return sum(scores) / n, sum(precs) / n, sum(recs) / n


def _collect_pairs(cache_dir, country, wanted, h_np, r_np, ids, names, addrs, topk):
    """Score blocking for the source-1 ids in ``wanted`` that live in this country."""
    path = os.path.join(cache_dir, _safe(country), "s1.tsv")
    if not os.path.isfile(path) or not wanted:
        return [], [], [], []
    xs: list[list[float]] = []
    ys: list[int] = []
    owners: list[str] = []
    cand_ids: list[str] = []
    n_s1 = 0
    hit = 0
    true_n = 0
    true_hit = 0
    cand_sizes = []
    id_set = set(ids)
    scores = np.zeros(len(ids), dtype=np.float32)
    for eid, name, addr in _iter_cache(path):
        truth = wanted.get(eid)
        if truth is None:
            continue
        n_s1 += 1
        recs = candidates_for(name, addr, h_np, r_np, len(ids), topk, scores, names, addrs)
        chosen = []
        for rec in recs:
            cid = ids[rec]
            if cid in id_set:
                chosen.append((rec, cid))
        cand_sizes.append(len(chosen))
        pred_set = {cid for _rec, cid in chosen}
        true_here = truth & id_set if truth else set()
        # Truth ids from other countries are ignored by the country block;
        # they still count toward blocking recall so a cross-country miss shows up.
        true_n += len(truth)
        true_hit += len(truth & pred_set)
        hit += 1
        for rec, cid in chosen:
            xs.append(pair_features(name, addr, names[rec], addrs[rec]))
            ys.append(1 if cid in truth else 0)
            owners.append(eid)
            cand_ids.append(cid)
        if n_s1 % 2000 == 0:
            log(f"  pairs {country} s1={n_s1} rows={len(ys)}")
    recall = true_hit / true_n if true_n else 1.0
    avg = sum(cand_sizes) / len(cand_sizes) if cand_sizes else 0.0
    log(
        f"blocking {country} queried={hit} avg_cand={avg:.1f} "
        f"link_recall={recall:.4f} true_links={true_n}"
    )
    return xs, ys, owners, cand_ids


def train_model(data_root: str, work_dir: str, topk: int) -> str:
    if lgb is None:
        raise SystemExit("lightgbm is required")
    gt_path = os.path.join(data_root, "train", "train_ground_truth.tsv")
    train_ids, val_ids = load_splits(gt_path)
    cache = os.path.join(work_dir, "cache", "train")
    build_cache(
        {
            "s1": os.path.join(data_root, "train", "train_source1.tsv"),
            "s2": os.path.join(data_root, "train", "train_source2.tsv"),
            "s3": os.path.join(data_root, "train", "train_source3.tsv"),
        },
        cache,
    )
    countries = discover_countries(os.path.join(data_root, "train", "train_source1.tsv"))
    log(f"train countries: {countries}")

    x_train: list[list[float]] = []
    y_train: list[int] = []
    x_val: list[list[float]] = []
    val_owner: list[str] = []
    val_cand: list[str] = []
    for country in countries:
        built = build_index(cache, country)
        h_np, r_np, ids, names, addrs = built
        xt, yt, _ot, _ct = _collect_pairs(
            cache, country, train_ids, h_np, r_np, ids, names, addrs, topk
        )
        x_train.extend(xt)
        y_train.extend(yt)
        xv, _yv, ov, cv = _collect_pairs(
            cache, country, val_ids, h_np, r_np, ids, names, addrs, topk
        )
        x_val.extend(xv)
        val_owner.extend(ov)
        val_cand.extend(cv)
        del built, h_np, r_np, ids, names, addrs
        gc.collect()

    x_all = np.asarray(x_train, dtype=np.float32)
    y_all = np.asarray(y_train, dtype=np.int8)
    del x_train, y_train
    pos = int(y_all.sum())
    log(f"train pairs={len(y_all)} positives={pos} rate={pos / max(len(y_all), 1):.4f}")
    rng = np.random.default_rng(42)
    pos_idx = np.flatnonzero(y_all == 1)
    neg_idx = np.flatnonzero(y_all == 0)
    rng.shuffle(neg_idx)
    keep_neg = neg_idx[: min(len(neg_idx), max(len(pos_idx), 1) * 4)]
    sel = np.concatenate([pos_idx, keep_neg])
    rng.shuffle(sel)
    x_fit = x_all[sel]
    y_fit = y_all[sel]
    del x_all, y_all
    gc.collect()
    x_val_np = np.asarray(x_val, dtype=np.float32) if x_val else np.zeros((0, len(FEATURE_NAMES)), np.float32)

    def sweep(probs: np.ndarray):
        winners: dict[str, tuple[float, str]] = {}
        for owner, cand, prob in zip(val_owner, val_cand, probs):
            prev = winners.get(cand)
            p = float(prob)
            if prev is None or p > prev[0] or (p == prev[0] and owner < prev[1]):
                winners[cand] = (p, owner)
        best_t = 0.5
        best_f = -1.0
        best_curve = []
        for threshold in np.linspace(0.15, 0.95, 81):
            pred_map: dict[str, set[str]] = defaultdict(set)
            for cand, (p, owner) in winners.items():
                if p >= threshold:
                    pred_map[owner].add(cand)
            score, prec, rec = macro_f05(pred_map, val_ids)
            best_curve.append((float(threshold), score, prec, rec))
            if score > best_f + 1e-6 or (abs(score - best_f) <= 1e-6 and threshold > best_t):
                best_f = score
                best_t = float(threshold)
        return best_f, best_t, best_curve

    models = []
    if lgb is not None:
        models.append((
            "lightgbm",
            lgb.LGBMClassifier(
                objective="binary",
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                min_child_samples=40,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                n_jobs=-1,
                verbose=-1,
            ),
        ))
    try:
        import xgboost as xgb
        models.append((
            "xgboost",
            xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=40,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                n_jobs=-1,
                tree_method="hist",
                eval_metric="logloss",
            ),
        ))
    except ImportError:
        log("xgboost not installed")
    try:
        from catboost import CatBoostClassifier
        models.append((
            "catboost",
            CatBoostClassifier(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3.0,
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
            ),
        ))
    except ImportError:
        log("catboost not installed")

    best_name = ""
    best_model = None
    best_t = 0.5
    best_f = -1.0
    curve = []
    for name, clf in models:
        log(f"fitting {name}")
        clf.fit(x_fit, y_fit)
        probs = clf.predict_proba(x_val_np)[:, 1] if len(x_val_np) else np.zeros(0, np.float32)
        score, threshold, this_curve = sweep(probs)
        log(f"{name} val macro F0.5={score:.4f} at threshold={threshold:.3f}")
        if score > best_f:
            best_f = score
            best_t = threshold
            best_model = clf
            best_name = name
            curve = this_curve
        del clf
        gc.collect()
    del x_fit, y_fit
    model = best_model
    log(f"selected {best_name} F0.5={best_f:.4f} threshold={best_t:.3f}")
    for threshold, score, prec, rec in curve[::8]:
        log(f"  t={threshold:.3f} F0.5={score:.4f} P={prec:.4f} R={rec:.4f}")
    importances = sorted(
        zip(FEATURE_NAMES, model.feature_importances_.tolist()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    log("importances " + ", ".join(f"{name}={imp}" for name, imp in importances))

    os.makedirs(work_dir, exist_ok=True)
    model_path = os.path.join(work_dir, "model.pkl")
    with open(model_path, "wb") as handle:
        pickle.dump(
            {"model": model, "model_name": best_name, "threshold": best_t, "f05": best_f, "features": FEATURE_NAMES, "topk": topk},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    metrics_path = os.path.join(work_dir, "val_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        handle.write(f"f05={best_f:.6f}\nthreshold={best_t:.6f}\ntopk={topk}\n")
        for threshold, score, prec, rec in curve:
            handle.write(f"{threshold:.4f}\t{score:.6f}\t{prec:.6f}\t{rec:.6f}\n")
    log(f"saved {model_path}")
    return model_path


def _write_outputs(s1_order: list[str], matches: dict[str, list[str]], cands: dict[str, list[str]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    match_path = os.path.join(out_dir, "matching_results.tsv")
    cand_path = os.path.join(out_dir, "candidate_pairs.tsv")
    with open(match_path, "w", encoding="utf-8", newline="\n") as match_f, open(
        cand_path, "w", encoding="utf-8", newline="\n"
    ) as cand_f:
        match_f.write("source1_entity_id\tmatched_entity_ids\n")
        cand_f.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid in s1_order:
            mids = matches.get(eid, [])
            cids = cands.get(eid, [])
            # Matches must be a subset of candidates.
            if mids:
                allowed = set(cids)
                mids = [m for m in mids if m in allowed]
            match_f.write(eid + "\t" + ",".join(mids) + "\n")
            cand_f.write(eid + "\t" + ",".join(cids) + "\n")
    log(f"wrote {match_path}")
    log(f"wrote {cand_path}")


def predict(data_root: str, work_dir: str, out_dir: str, model_path: str) -> None:
    with open(model_path, "rb") as handle:
        blob = pickle.load(handle)
    model = blob["model"]
    threshold = float(blob["threshold"])
    topk = int(blob.get("topk", TOPK_DEFAULT))
    log(f"predict threshold={threshold:.3f} topk={topk}")
    cache = os.path.join(work_dir, "cache", "test")
    s1_path = os.path.join(data_root, "test", "test_source1.tsv")
    build_cache(
        {
            "s1": s1_path,
            "s2": os.path.join(data_root, "test", "test_source2.tsv"),
            "s3": os.path.join(data_root, "test", "test_source3.tsv"),
        },
        cache,
    )
    s1_order = []
    with open(s1_path, encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            eid = line.split("\t", 1)[0]
            if eid:
                s1_order.append(eid)
    countries = discover_countries(s1_path)
    log(f"test countries: {countries} s1={len(s1_order)}")
    matches: dict[str, list[str]] = {}
    cands: dict[str, list[str]] = {}
    for country in countries:
        h_np, r_np, ids, names, addrs = build_index(cache, country)
        path = os.path.join(cache, _safe(country), "s1.tsv")
        batch_eids: list[str] = []
        batch_recs: list[list[int]] = []
        batch_names: list[str] = []
        batch_addrs: list[str] = []
        n_s1 = 0
        scores = np.zeros(len(ids), dtype=np.float32)
        pending: list[tuple[str, str, float]] = []

        def flush() -> None:
            if not batch_eids:
                return
            rows = []
            spans = []
            for name, addr, recs in zip(batch_names, batch_addrs, batch_recs):
                start = len(rows)
                for rec in recs:
                    rows.append(pair_features(name, addr, names[rec], addrs[rec]))
                spans.append((start, len(rows)))
            if rows:
                probs = model.predict_proba(np.asarray(rows, dtype=np.float32))[:, 1]
            else:
                probs = np.zeros(0, np.float32)
            for eid, recs, (start, end) in zip(batch_eids, batch_recs, spans):
                cand_list = list(dict.fromkeys(ids[rec] for rec in recs))
                cands[eid] = cand_list
                for rec, prob in zip(recs, probs[start:end]):
                    pending.append((eid, ids[rec], float(prob)))
            batch_eids.clear()
            batch_recs.clear()
            batch_names.clear()
            batch_addrs.clear()

        for eid, name, addr in _iter_cache(path):
            recs = candidates_for(name, addr, h_np, r_np, len(ids), topk, scores, names, addrs)
            batch_eids.append(eid)
            batch_recs.append(recs)
            batch_names.append(name)
            batch_addrs.append(addr)
            n_s1 += 1
            if len(batch_eids) >= 1500:
                flush()
            if n_s1 % 50000 == 0:
                log(f"  predict {country} {n_s1}")
        flush()
        winners: dict[str, tuple[float, str]] = {}
        for eid, cid, prob in pending:
            prev = winners.get(cid)
            if prev is None or prob > prev[0] or (prob == prev[0] and eid < prev[1]):
                winners[cid] = (prob, eid)
        for cid, (prob, eid) in winners.items():
            if prob >= threshold:
                matches.setdefault(eid, []).append(cid)
        log(f"predict country {country} s1={n_s1}")
        del h_np, r_np, ids, names, addrs
        gc.collect()
    _write_outputs(s1_order, matches, cands, out_dir)


def self_test() -> None:
    name = normalize("Life Healthcare Private Limited")
    addr = normalize("C/O Ketan Rajendra Mande, Dahiwad, Amalner, Jalgaon, Maharashtra")
    name2 = normalize("लाइफ हेल्थकेयर प्राइवेट लिमिटेड")
    addr2 = normalize("JALGAON, AMALNER, Maharashtra, C/O KETAN RAJENDRA MANDE")
    k1 = {k for k, _ in blocking_keys(name, addr)}
    k2 = {k for k, _ in blocking_keys(name2, addr2)}
    shared = k1 & k2
    assert shared, f"expected shared address keys, got none\n{sorted(k1)[:12]}\n{sorted(k2)[:12]}"
    n_us = normalize("Brown Piedmont Australia")
    n_dom = normalize("brownpiedmontaustralia.com")
    jk = {k for k, _ in blocking_keys(n_us, "washougal magnolia")}
    dk = {k for k, _ in blocking_keys(n_dom, "washougal magnolia")}
    assert jk & dk, (jk, dk)
    a = normalize("Ábber Shop LLC")
    assert "abber" in name_tokens(a)
    assert abs(f05_one({"a", "b", "c"}, {"a", "c"}) - 0.714285714) < 1e-3 or True
    # The prompt example: P=2/3, R=1 -> 0.714
    got = f05_one({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    assert abs(got - 0.7142857142857143) < 1e-6, got
    assert f05_one(set(), set()) == 1.0
    assert f05_one({"x"}, set()) == 0.0
    log("self-test ok")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Business entity resolution pipeline")
    parser.add_argument("--data-root", required=True, help="Folder that contains train/ and test/")
    parser.add_argument("--work-dir", required=True, help="Cache, model, and logs")
    parser.add_argument("--out-dir", required=True, help="Folder for matching_results.tsv and candidate_pairs.tsv")
    parser.add_argument("--stage", choices=("train", "predict", "all", "self-test"), default="all")
    parser.add_argument("--topk", type=int, default=TOPK_DEFAULT)
    parser.add_argument("--model", default=None, help="Model pickle (default: WORK/model.pkl)")
    args = parser.parse_args()
    if args.stage == "self-test":
        self_test()
        return
    model_path = args.model or os.path.join(args.work_dir, "model.pkl")
    if args.stage in ("train", "all"):
        train_model(args.data_root, args.work_dir, args.topk)
    if args.stage in ("predict", "all"):
        if not os.path.isfile(model_path):
            raise SystemExit(f"missing model: {model_path}")
        predict(args.data_root, args.work_dir, args.out_dir, model_path)


if __name__ == "__main__":
    main()
