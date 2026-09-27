import csv
import os
import sys
from collections import defaultdict
from difflib import SequenceMatcher

def token_set_ratio(s1, s2):
    """Approximate token set ratio using SequenceMatcher on token sets."""
    if not s1 or not s2:
        return 0.0
    tokens1 = set(s1.lower().split())
    tokens2 = set(s2.lower().split())
    if not tokens1 or not tokens2:
        return 0.0
    inter = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(inter) / len(union)

def norm(text):
    return text.strip().lower() if text else ""

def first_tokens(text, n=2):
    toks = norm(text).split()
    return " ".join(toks[:n])

def build_blocks(df_iter, id_col, name_col, addr_col, country_col):
    """Yield (block_key, entity_id, name, addr, country) for each row."""
    for row in df_iter:
        eid = row[id_col]
        name = row[name_col]
        addr = row[addr_col]
        country = row[country_col]
        block_key = f"{norm(country)}|{first_tokens(name, 2)}"
        yield block_key, eid, name, addr, country

def read_tsv(path):
    with open(path, "r", newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            yield row

def main():
    base = "dataset/student_resource/dataset"
    test_dir = os.path.join(base, "test")
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    src1_path = os.path.join(test_dir, "test_source1.tsv")
    src2_path = os.path.join(test_dir, "test_source2.tsv")
    src3_path = os.path.join(test_dir, "test_source3.tsv")

    # Build blocks for source2 and source3
    blocks = defaultdict(list)   # block_key -> list of (eid, name, addr, country)
    for path in (src2_path, src3_path):
        for block_key, eid, name, addr, country in build_blocks(read_tsv(path),
                                                                "entity_id","business_name","business_address","country"):
            blocks[block_key].append((eid, name, addr, country))

    # For each source1 entity, find candidates in same block, score them
    cand_out = os.path.join(out_dir, "candidate_pairs.tsv")
    match_out = os.path.join(out_dir, "matching_results.tsv")

    with open(src1_path, "r", newline='', encoding='utf-8') as f1, \
         open(cand_out, "w", newline='', encoding='utf-8') as fc, \
         open(match_out, "w", newline='', encoding='utf-8') as fm:

        reader1 = csv.DictReader(f1, delimiter='\t')
        writer_c = csv.DictWriter(fc, fieldnames=["source1_entity_id","candidate_entity_ids"], delimiter='\t')
        writer_m = csv.DictWriter(fm, fieldnames=["source1_entity_id","matched_entity_ids"], delimiter='\t')
        writer_c.writeheader()
        writer_m.writeheader()

        for row1 in reader1:
            eid1 = row1["entity_id"]
            name1 = row1["business_name"]
            addr1 = row1["business_address"]
            country1 = row1["country"]
            block_key = f"{norm(country1)}|{first_tokens(name1, 2)}"

            candidates = blocks.get(block_key, [])
            cand_ids = []
            match_ids = []

            for eid2, name2, addr2, country2 in candidates:
                # quick filter: same country (already by block)
                # compute similarity
                name_sim = token_set_ratio(name1, name2)
                addr_sim = token_set_ratio(addr1, addr2)
                # combined score weighted
                score = 0.7 * name_sim + 0.3 * addr_sim
                if score >= 0.55:          # threshold tuned roughly
                    cand_ids.append(eid2)
                    if score >= 0.75:      # higher threshold for final match
                        match_ids.append(eid2)

            # dedupe preserving order
            def dedup(lst):
                seen = set()
                out = []
                for x in lst:
                    if x not in seen:
                        seen.add(x)
                        out.append(x)
                return out

            cand_ids = dedup(cand_ids)
            match_ids = dedup(match_ids)

            writer_c.writerow({"source1_entity_id": eid1,
                               "candidate_entity_ids": ",".join(cand_ids)})
            writer_m.writerow({"source1_entity_id": eid1,
                               "matched_entity_ids": ",".join(match_ids)})

    print("Improved outputs written to", out_dir)

if __name__ == "__main__":
    main()