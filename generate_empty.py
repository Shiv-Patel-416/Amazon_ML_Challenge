import csv
import os

test_dir = "dataset/student_resource/dataset/test"
out_dir = "output"
src1_path = os.path.join(test_dir, "test_source1.tsv")
cand_path = os.path.join(out_dir, "candidate_pairs.tsv")
match_path = os.path.join(out_dir, "matching_results.tsv")

os.makedirs(out_dir, exist_ok=True)

with open(src1_path, "r", newline='', encoding='utf-8') as f_in, \
     open(cand_path, "w", newline='', encoding='utf-8') as f_cand, \
     open(match_path, "w", newline='', encoding='utf-8') as f_match:

    reader = csv.DictReader(f_in, delimiter='\t')
    writer_cand = csv.DictWriter(f_cand, fieldnames=["source1_entity_id", "candidate_entity_ids"], delimiter='\t')
    writer_match = csv.DictWriter(f_match, fieldnames=["source1_entity_id", "matched_entity_ids"], delimiter='\t')
    writer_cand.writeheader()
    writer_match.writeheader()

    for row in reader:
        eid = row["entity_id"]
        writer_cand.writerow({"source1_entity_id": eid, "candidate_entity_ids": ""})
        writer_match.writerow({"source1_entity_id": eid, "matched_entity_ids": ""})

print("Generated empty baseline outputs")