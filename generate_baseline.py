import pandas as pd
import os

# paths
train_dir = "dataset/student_resource/dataset/train"
test_dir = "dataset/student_resource/dataset/test"
out_dir = "output"

# read test sources
src1 = pd.read_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t", dtype=str)
src2 = pd.read_csv(os.path.join(test_dir, "test_source2.tsv"), sep="\t", dtype=str)
src3 = pd.read_csv(os.path.join(test_dir, "test_source3.tsv"), sep="\t", dtype=str)

# simple blocking: same country and first token of name
def first_token(name):
    if pd.isna(name):
        return ""
    return str(name).strip().split()[0].lower()

src2["block_key"] = src2["country"].fillna("") + "_" + src2["business_name"].apply(first_token)
src3["block_key"] = src3["country"].fillna("") + "_" + src3["business_name"].apply(first_token)

# build dict from block_key to list of entity_ids
candidates_map = {}
for _, row in src2.iterrows():
    candidates_map.setdefault(row["block_key"], []).append(row["entity_id"])
for _, row in src3.iterrows():
    candidates_map.setdefault(row["block_key"], []).append(row["entity_id"])

# generate candidate pairs for each source1 entity
cand_rows = []
match_rows = []
for _, row in src1.iterrows():
    key = row["country"].fillna("") + "_" + first_token(row["business_name"])
    cand_ids = candidates_map.get(key, [])
    # dedupe
    cand_ids = list(dict.fromkeys(cand_ids))
    cand_str = ",".join(cand_ids)
    cand_rows.append({"source1_entity_id": row["entity_id"], "candidate_entity_ids": cand_str})
    # for baseline matching, we output empty matches
    match_rows.append({"source1_entity_id": row["entity_id"], "matched_entity_ids": ""})

cand_df = pd.DataFrame(cand_rows, columns=["source1_entity_id", "candidate_entity_ids"])
match_df = pd.DataFrame(match_rows, columns=["source1_entity_id", "matched_entity_ids"])

cand_df.to_csv(os.path.join(out_dir, "candidate_pairs.tsv"), sep="\t", index=False)
match_df.to_csv(os.path.join(out_dir, "matching_results.tsv"), sep="\t", index=False)

print("Generated baseline outputs in", out_dir)