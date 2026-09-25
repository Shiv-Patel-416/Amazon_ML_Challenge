"""Analyze blocking failures to improve recall."""
import csv, os, sys, re, unicodedata
from collections import defaultdict, Counter
sys.stdout.reconfigure(encoding='utf-8')

base = 'dataset/student_resource/dataset/train'

# Import solution preprocessing
sys.path.insert(0, '.')
from solution import clean_name, clean_address, get_name_key_tokens, char_ngrams, extract_numbers, has_non_ascii

# Load small sample
import random
random.seed(42)

# Load ground truth (first 5000)
gt = {}
with open(os.path.join(base, 'train_ground_truth.tsv'), encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for i, row in enumerate(reader):
        if i >= 5000:
            break
        s1_id = row['source1_entity_id']
        m = (row.get('matched_entity_ids', '') or '').strip()
        if m:
            gt[s1_id] = set(m.split(','))
        else:
            gt[s1_id] = set()

# Get relevant IDs
s1_ids = set(gt.keys())
s23_ids = set()
for v in gt.values():
    s23_ids.update(v)

# Load records
s1_data = {}
with open(os.path.join(base, 'train_source1.tsv'), encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        if row['entity_id'] in s1_ids:
            s1_data[row['entity_id']] = row
            s1_data[row['entity_id']]['name_clean'] = clean_name(row['business_name'])
            s1_data[row['entity_id']]['addr_clean'] = clean_address(row['business_address'])
            s1_data[row['entity_id']]['country_clean'] = row['country'].lower().strip()

s23_data = {}
for src in ['train_source2.tsv', 'train_source3.tsv']:
    with open(os.path.join(base, src), encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row['entity_id'] in s23_ids:
                s23_data[row['entity_id']] = row
                s23_data[row['entity_id']]['name_clean'] = clean_name(row['business_name'])
                s23_data[row['entity_id']]['addr_clean'] = clean_address(row['business_address'])
                s23_data[row['entity_id']]['country_clean'] = row['country'].lower().strip()

# Analyze what blocking would catch
def simulate_blocking(s1_rec, s23_rec):
    """Check if any blocking strategy would connect these records."""
    reasons = []
    
    country_match = s1_rec['country_clean'] == s23_rec['country_clean']
    if not country_match:
        return [], 'country_mismatch'
    
    # Token overlap
    t1 = get_name_key_tokens(s1_rec['name_clean'])
    t2 = get_name_key_tokens(s23_rec['name_clean'])
    common_tokens = t1 & t2
    if common_tokens:
        reasons.append(f'token_overlap:{common_tokens}')
    
    # Prefix match
    n1 = s1_rec['name_clean']
    n2 = s23_rec['name_clean']
    if n1[:3] == n2[:3] and len(n1) >= 3:
        reasons.append('prefix_match')
    
    # Number overlap
    nums1 = extract_numbers(s1_rec['addr_clean'])
    nums2 = extract_numbers(s23_rec['addr_clean'])
    common_nums = {n for n in nums1 & nums2 if len(n) >= 3}
    if common_nums:
        reasons.append(f'num_overlap:{common_nums}')
    
    # N-gram overlap
    n1_ascii = ''.join(c for c in n1 if ord(c) < 128)
    n2_ascii = ''.join(c for c in n2 if ord(c) < 128)
    ng1 = char_ngrams(n1_ascii, 3)
    ng2 = char_ngrams(n2_ascii, 3)
    if ng1 and ng2:
        ngram_overlap = len(ng1 & ng2) / max(len(ng1 | ng2), 1)
        if ngram_overlap > 0.1:
            reasons.append(f'ngram_overlap:{ngram_overlap:.2f}')
    
    # Cross-script (transliteration case)
    s1_non_ascii = has_non_ascii(s1_rec['name_clean'])
    s23_non_ascii = has_non_ascii(s23_rec['name_clean'])
    if s1_non_ascii != s23_non_ascii:
        reasons.append('cross_script')
    
    return reasons, 'found' if reasons else 'no_overlap'

# Analyze
missed_examples = []
caught = 0
missed = 0
miss_reasons = Counter()

for s1_id, true_matches in gt.items():
    if s1_id not in s1_data:
        continue
    s1_rec = s1_data[s1_id]
    
    for mid in true_matches:
        if mid not in s23_data:
            continue
        s23_rec = s23_data[mid]
        
        reasons, status = simulate_blocking(s1_rec, s23_rec)
        
        if status == 'found':
            caught += 1
        elif status == 'country_mismatch':
            missed += 1
            miss_reasons['country_mismatch'] += 1
        else:
            missed += 1
            miss_reasons['no_overlap'] += 1
            if len(missed_examples) < 15:
                missed_examples.append((s1_id, mid, s1_rec, s23_rec))

total = caught + missed
print(f"Blocking analysis: {caught}/{total} caught ({100*caught/total:.1f}%)")
print(f"Missed: {missed} ({100*missed/total:.1f}%)")
print(f"Miss reasons: {dict(miss_reasons)}")

print("\n--- Missed examples ---")
for s1_id, mid, s1_rec, s23_rec in missed_examples:
    print(f"\nS1: {s1_id}")
    print(f"  Name: {s1_rec['business_name']}")
    print(f"  Name_clean: {s1_rec['name_clean']}")
    print(f"  Addr: {s1_rec['business_address']}")
    print(f"  Country: {s1_rec['country']}")
    print(f"  -> {mid}")
    print(f"  Name: {s23_rec['business_name']}")
    print(f"  Name_clean: {s23_rec['name_clean']}")
    print(f"  Addr: {s23_rec['business_address']}")
    print(f"  Country: {s23_rec['country']}")
