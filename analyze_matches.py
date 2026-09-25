"""Quick analysis of matched pairs to understand noise patterns."""
import csv, os, sys
sys.stdout.reconfigure(encoding='utf-8')
base = 'dataset/student_resource/dataset/train'

# Load a sample of ground truth 
gt = {}
with open(os.path.join(base, 'train_ground_truth.tsv'), encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for i, row in enumerate(reader):
        if i >= 20:
            break
        m = row.get('matched_entity_ids', '').strip()
        if m:
            gt[row['source1_entity_id']] = m.split(',')

# Load S1 records for these IDs
s1_ids = set(gt.keys())
s23_ids = set()
for v in gt.values():
    s23_ids.update(v)

s1_data = {}
with open(os.path.join(base, 'train_source1.tsv'), encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        if row['entity_id'] in s1_ids:
            s1_data[row['entity_id']] = row

# Load matched S2/S3 records
s23_data = {}
for src in ['train_source2.tsv', 'train_source3.tsv']:
    with open(os.path.join(base, src), encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row['entity_id'] in s23_ids:
                s23_data[row['entity_id']] = row

# Show some examples
for s1_id, matched_ids in list(gt.items())[:8]:
    s1 = s1_data.get(s1_id, {})
    print("S1:", s1_id)
    print("  Name:", s1.get('business_name', ''))
    print("  Addr:", s1.get('business_address', ''))
    print("  Country:", s1.get('country', ''))
    for mid in matched_ids[:3]:
        s23 = s23_data.get(mid, {})
        print("  ->", mid, ":", s23.get('business_name', 'N/A'), "|", s23.get('business_address', 'N/A'), "|", s23.get('country', 'N/A'))
    print()
