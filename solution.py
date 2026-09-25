"""
Business Entity Resolution - Optimized Pure Python Solution
============================================================
Designed for scale: 1.7M S1 × 10M S2+S3 with pure Python.

Architecture:
  1. Load data by country to manage memory
  2. Multi-strategy inverted index blocking
  3. Score-based candidate ranking
  4. Fast feature computation (set-based, avoiding O(n*m) edit distance)
  5. Rule-based matching with precision-focused thresholds

Usage:
  python solution.py                    # Full pipeline (train validation + test)
  python solution.py --test-only        # Test prediction only
  python solution.py --validate-only    # Validate on training sample
"""

import os
import sys
import csv
import re
import time
import math
import unicodedata
import argparse
import gc
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = "dataset/student_resource/dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
OUT_DIR = "output"

os.makedirs(OUT_DIR, exist_ok=True)

# Blocking parameters
MAX_CANDIDATES = 20          # Max candidates per S1 entity
MAX_BLOCK_SIZE = 1000        # Skip overly common blocking keys

# ============================================================================
# Text Normalization
# ============================================================================

SUFFIX_MAP = {
    'corporation': 'corp', 'incorporated': 'inc', 'limited': 'ltd',
    'company': 'co', 'private': 'pvt', 'pvt.': 'pvt',
    'associates': 'assoc', 'association': 'assoc',
    'international': 'intl', 'national': 'natl',
    'services': 'svc', 'service': 'svc',
    'enterprise': 'ent', 'enterprises': 'ent',
    'industries': 'ind', 'industry': 'ind',
    'solutions': 'soln', 'solution': 'soln',
    'technologies': 'tech', 'technology': 'tech',
    'manufacturing': 'mfg', 'management': 'mgmt',
    'development': 'dev', 'developers': 'dev',
    'construction': 'constr', 'consultants': 'consult',
    'consultant': 'consult', 'consulting': 'consult',
    'foundation': 'fdn', 'laboratories': 'labs',
    'laboratory': 'labs', 'lab': 'labs',
    'societe': 'ste', 'etablissements': 'ets',
}

NAME_STOP = frozenset({'inc', 'llc', 'ltd', 'corp', 'co', 'pvt', 'llp',
                        'the', 'of', 'and', 'a', 'an', 'sarl', 'sas', 'sa',
                        'dba', 'gmbh', 'ag', 'plc', 'nv', 'bv', 'pty', 'pte',
                        'for', 'in', 'at', 'on', 'by', 'to', 'is', 'it',
                        'or', 'not', 'no', 'with', 'from'})

ADDR_ABBREV = {
    'street': 'st', 'road': 'rd', 'avenue': 'ave', 'boulevard': 'blvd',
    'drive': 'dr', 'lane': 'ln', 'court': 'ct', 'place': 'pl',
    'circle': 'cir', 'highway': 'hwy', 'parkway': 'pkwy', 'terrace': 'ter',
    'apartment': 'apt', 'suite': 'ste', 'building': 'bldg', 'floor': 'fl',
    'north': 'n', 'south': 's', 'east': 'e', 'west': 'w',
    'northeast': 'ne', 'northwest': 'nw', 'southeast': 'se', 'southwest': 'sw',
    'township': 'twp', 'heights': 'hts', 'springs': 'spgs',
    'mount': 'mt', 'saint': 'st', 'fort': 'ft',
}

ADDR_NOISE = frozenset({'near', 'opposite', 'opp', 'behind', 'beside', 'next',
                         'above', 'null', 'po', 'box', 'unit', 'apt', 'ste',
                         'fl', 'no', 'kh', 'plot', 'door'})

# US State full → abbreviation
US_STATES = {
    'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar',
    'california': 'ca', 'colorado': 'co', 'connecticut': 'ct', 'delaware': 'de',
    'florida': 'fl', 'georgia': 'ga', 'hawaii': 'hi', 'idaho': 'id',
    'illinois': 'il', 'indiana': 'in', 'iowa': 'ia', 'kansas': 'ks',
    'kentucky': 'ky', 'louisiana': 'la', 'maine': 'me', 'maryland': 'md',
    'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn',
    'mississippi': 'ms', 'missouri': 'mo', 'montana': 'mt', 'nebraska': 'ne',
    'nevada': 'nv', 'new hampshire': 'nh', 'new jersey': 'nj',
    'new mexico': 'nm', 'new york': 'ny', 'north carolina': 'nc',
    'north dakota': 'nd', 'ohio': 'oh', 'oklahoma': 'ok', 'oregon': 'or',
    'pennsylvania': 'pa', 'rhode island': 'ri', 'south carolina': 'sc',
    'south dakota': 'sd', 'tennessee': 'tn', 'texas': 'tx', 'utah': 'ut',
    'vermont': 'vt', 'virginia': 'va', 'washington': 'wa',
    'west virginia': 'wv', 'wisconsin': 'wi', 'wyoming': 'wy',
    'district of columbia': 'dc',
}

# Full state name → abbreviation for Indian states
INDIAN_STATES = {
    'andhra pradesh': 'ap', 'arunachal pradesh': 'ar', 'assam': 'as',
    'bihar': 'br', 'chhattisgarh': 'cg', 'goa': 'ga', 'gujarat': 'gj',
    'haryana': 'hr', 'himachal pradesh': 'hp', 'jharkhand': 'jh',
    'karnataka': 'ka', 'kerala': 'kl', 'madhya pradesh': 'mp',
    'maharashtra': 'mh', 'manipur': 'mn', 'meghalaya': 'ml',
    'mizoram': 'mz', 'nagaland': 'nl', 'odisha': 'od', 'orissa': 'od',
    'punjab': 'pb', 'rajasthan': 'rj', 'sikkim': 'sk',
    'tamil nadu': 'tn', 'telangana': 'tg', 'tripura': 'tr',
    'uttar pradesh': 'up', 'uttarakhand': 'uk', 'west bengal': 'wb',
    'delhi': 'dl', 'new delhi': 'dl',
}

# French region name normalization
FRENCH_REGIONS = {
    'nouvelle aquitaine': 'naq', 'ile de france': 'idf',
    'auvergne rhone alpes': 'ara', 'occitanie': 'occ',
    'hauts de france': 'hdf', 'grand est': 'gest',
    'provence alpes cote d azur': 'paca', 'bretagne': 'bre',
    'normandie': 'nor', 'pays de la loire': 'pdl',
    'centre val de loire': 'cvl', 'bourgogne franche comte': 'bfc',
    'corse': 'cor',
}


def normalize_unicode(text):
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in text if unicodedata.category(c) != 'Mn')


def clean_name(name):
    if not name:
        return ""
    name = str(name).lower().strip()
    name = normalize_unicode(name)
    # Remove URLs
    name = re.sub(r'https?://\S+', '', name)
    name = re.sub(r'www\.\S+', '', name)
    # Remove after pipe
    name = name.split('|')[0].strip()
    # Normalize & to and
    name = name.replace(' & ', ' and ').replace('&', ' and ')
    # Remove special chars
    name = re.sub(r'[^\w\s]', ' ', name)
    # Normalize suffixes
    tokens = name.split()
    out = [SUFFIX_MAP.get(t, t) for t in tokens]
    return ' '.join(out).strip()


def clean_address(addr):
    if not addr:
        return ""
    addr = str(addr).lower().strip()
    addr = normalize_unicode(addr)
    addr = re.sub(r'[^\w\s,]', ' ', addr)
    tokens = addr.split()
    out = []
    for t in tokens:
        r = ADDR_ABBREV.get(t, t)
        if r and r not in ADDR_NOISE:
            out.append(r)
    result = ' '.join(out)
    # Normalize state names
    for full, abbr in US_STATES.items():
        result = result.replace(full, abbr)
    for full, abbr in INDIAN_STATES.items():
        result = result.replace(full, abbr)
    return re.sub(r'\s+', ' ', result).strip()


def get_significant_tokens(name):
    """Get significant name tokens (length > 2, not stop words)."""
    if not name:
        return frozenset()
    return frozenset(t for t in name.split() if len(t) > 2 and t not in NAME_STOP)


def get_ascii_significant_tokens(name):
    """Get only ASCII significant tokens."""
    if not name:
        return frozenset()
    return frozenset(t for t in name.split() 
                     if len(t) > 2 and t not in NAME_STOP and t.isascii())


def extract_numbers(text):
    """Extract numbers ≥3 digits from text."""
    if not text:
        return frozenset()
    return frozenset(n for n in re.findall(r'\d+', text) if len(n) >= 3)


def char_ngrams(s, n=3):
    """Character n-grams."""
    if not s or len(s) < n:
        return frozenset()
    return frozenset(s[i:i+n] for i in range(len(s) - n + 1))


def has_non_ascii(text):
    return bool(text) and any(ord(c) > 127 for c in text)


def jaccard(s1, s2):
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union else 0.0


def overlap(s1, s2):
    if not s1 or not s2:
        return 0.0
    inter = len(s1 & s2)
    mn = min(len(s1), len(s2))
    return inter / mn if mn else 0.0


def dice(s1, s2):
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    inter = len(s1 & s2)
    return 2 * inter / (len(s1) + len(s2))


# ============================================================================
# Record Processing
# ============================================================================

def process_record(row):
    """Preprocess a raw CSV row into an enriched record dict."""
    rec = {
        'entity_id': row['entity_id'],
        'country': (row.get('country', '') or '').lower().strip(),
    }
    
    raw_name = row.get('business_name', '') or ''
    raw_addr = row.get('business_address', '') or ''
    
    rec['name_clean'] = clean_name(raw_name)
    rec['addr_clean'] = clean_address(raw_addr)
    rec['name_tokens'] = get_significant_tokens(rec['name_clean'])
    rec['name_ascii_tokens'] = get_ascii_significant_tokens(rec['name_clean'])
    rec['addr_numbers'] = extract_numbers(rec['addr_clean'])
    rec['name_ngrams'] = char_ngrams(''.join(c for c in rec['name_clean'] if c.isascii()), 3)
    rec['addr_tokens'] = frozenset(t for t in rec['addr_clean'].split() 
                                    if len(t) > 2 and t not in ADDR_NOISE)
    rec['is_non_ascii'] = has_non_ascii(rec['name_clean'])
    
    return rec


# ============================================================================
# Blocking Engine
# ============================================================================

class BlockingEngine:
    """Multi-strategy inverted index for blocking."""
    
    def __init__(self, max_block_size=MAX_BLOCK_SIZE):
        self.max_block_size = max_block_size
        # Indices: key -> set of entity_ids
        self.name_token_idx = defaultdict(set)      # name token -> ids
        self.name_prefix_idx = defaultdict(set)      # first 3 chars -> ids
        self.addr_number_idx = defaultdict(set)      # address number -> ids
        self.name_ngram_idx = defaultdict(set)       # char 3-gram -> ids
        self.addr_token_idx = defaultdict(set)       # address token -> ids
        self.first_word_idx = defaultdict(set)       # first word of name -> ids
        
    def add(self, rec):
        eid = rec['entity_id']
        name = rec['name_clean']
        
        # Name tokens
        for t in rec['name_tokens']:
            s = self.name_token_idx[t]
            if len(s) < self.max_block_size:
                s.add(eid)
        
        # Name prefix (first 3 chars)
        if len(name) >= 3:
            self.name_prefix_idx[name[:3]].add(eid)
        
        # First word
        words = name.split()
        if words and len(words[0]) > 2:
            s = self.first_word_idx[words[0]]
            if len(s) < self.max_block_size:
                s.add(eid)
        
        # Address numbers
        for n in rec['addr_numbers']:
            s = self.addr_number_idx[n]
            if len(s) < self.max_block_size * 2:
                s.add(eid)
        
        # Name character n-grams
        for ng in rec['name_ngrams']:
            s = self.name_ngram_idx[ng]
            if len(s) < self.max_block_size * 2:
                s.add(eid)
        
        # Address tokens (for cross-script matching)
        for t in rec['addr_tokens']:
            s = self.addr_token_idx[t]
            if len(s) < self.max_block_size * 2:
                s.add(eid)
    
    def query(self, rec, max_candidates=MAX_CANDIDATES):
        """Find candidate entity IDs for a given S1 record."""
        scores = Counter()
        
        # Name tokens (weight 4)
        for t in rec['name_tokens']:
            block = self.name_token_idx.get(t)
            if block and len(block) < self.max_block_size:
                for cid in block:
                    scores[cid] += 4
        
        # First word (weight 3)
        words = rec['name_clean'].split()
        if words and len(words[0]) > 2:
            block = self.first_word_idx.get(words[0])
            if block and len(block) < self.max_block_size:
                for cid in block:
                    scores[cid] += 3
        
        # Name prefix (weight 2)
        name = rec['name_clean']
        if len(name) >= 3:
            block = self.name_prefix_idx.get(name[:3])
            if block:
                for cid in block:
                    scores[cid] += 2
        
        # Address numbers (weight 3)
        for n in rec['addr_numbers']:
            block = self.addr_number_idx.get(n)
            if block and len(block) < self.max_block_size * 2:
                for cid in block:
                    scores[cid] += 3
        
        # Name n-grams (weight 1)
        for ng in rec['name_ngrams']:
            block = self.name_ngram_idx.get(ng)
            if block and len(block) < self.max_block_size * 2:
                for cid in block:
                    scores[cid] += 1
        
        # Address tokens (weight 2) - important for cross-script cases
        for t in rec['addr_tokens']:
            block = self.addr_token_idx.get(t)
            if block and len(block) < self.max_block_size * 2:
                for cid in block:
                    scores[cid] += 2
        
        if not scores:
            return []
        
        # Return top candidates
        return [cid for cid, _ in scores.most_common(max_candidates)]
    
    def stats(self):
        return {
            'name_token': len(self.name_token_idx),
            'name_prefix': len(self.name_prefix_idx),
            'first_word': len(self.first_word_idx),
            'addr_number': len(self.addr_number_idx),
            'name_ngram': len(self.name_ngram_idx),
            'addr_token': len(self.addr_token_idx),
        }


# ============================================================================
# String Similarity (Levenshtein - only used on ~20 candidates, not during blocking)
# ============================================================================

def levenshtein_ratio(s1, s2):
    """Normalized Levenshtein similarity (0=different, 1=identical)."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    len1, len2 = len(s1), len(s2)
    max_len = max(len1, len2)
    
    # Quick reject: if length difference is too large, they can't be similar
    if abs(len1 - len2) / max_len > 0.6:
        return 0.0
    
    # Compute edit distance with early termination
    if len1 < len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1
    
    prev = list(range(len2 + 1))
    for i in range(len1):
        curr = [i + 1]
        for j in range(len2):
            cost = 0 if s1[i] == s2[j] else 1
            curr.append(min(prev[j+1] + 1, curr[j] + 1, prev[j] + cost))
        prev = curr
    
    return 1.0 - prev[len2] / max_len


def token_sort_ratio(s1, s2):
    """Sort tokens and compute Levenshtein ratio."""
    t1 = ' '.join(sorted(s1.split()))
    t2 = ' '.join(sorted(s2.split()))
    return levenshtein_ratio(t1, t2)


def token_set_ratio(s1, s2):
    """Token set similarity: compare shared + remainder tokens."""
    if s1 == s2:
        return 1.0
    t1 = set(s1.split())
    t2 = set(s2.split())
    if not t1 or not t2:
        return 0.0
    common = sorted(t1 & t2)
    rest1 = sorted(t1 - t2)
    rest2 = sorted(t2 - t1)
    
    if not common:
        return levenshtein_ratio(s1, s2)
    
    s_common = ' '.join(common)
    s_1 = (s_common + ' ' + ' '.join(rest1)).strip()
    s_2 = (s_common + ' ' + ' '.join(rest2)).strip()
    
    return max(
        levenshtein_ratio(s_common, s_1),
        levenshtein_ratio(s_common, s_2),
        levenshtein_ratio(s_1, s_2)
    )


# ============================================================================
# Matching Engine
# ============================================================================

def compute_match_score(s1, s23):
    """
    Compute match confidence score between S1 and S2/S3 records.
    Returns (is_match, confidence).
    Uses set-based features for quick screening + Levenshtein for precision.
    """
    # ---- Fast set-based features (no expensive computation) ----
    name_tok_jaccard = jaccard(s1['name_tokens'], s23['name_tokens'])
    name_tok_overlap = overlap(s1['name_tokens'], s23['name_tokens'])
    name_tok_dice = dice(s1['name_tokens'], s23['name_tokens'])
    
    name_ng_jaccard = jaccard(s1['name_ngrams'], s23['name_ngrams'])
    
    addr_tok_jaccard = jaccard(s1['addr_tokens'], s23['addr_tokens'])
    addr_tok_overlap = overlap(s1['addr_tokens'], s23['addr_tokens'])
    
    addr_num_jaccard = jaccard(s1['addr_numbers'], s23['addr_numbers'])
    addr_num_overlap = overlap(s1['addr_numbers'], s23['addr_numbers'])
    
    s1_words = s1['name_clean'].split()
    s23_words = s23['name_clean'].split()
    first_word_match = (len(s1_words) > 0 and len(s23_words) > 0 and 
                        s1_words[0] == s23_words[0] and len(s1_words[0]) > 2)
    
    cross_script = s1['is_non_ascii'] != s23['is_non_ascii']
    
    # ---- Cross-script matching (transliteration cases) ----
    if cross_script:
        # Names are in different scripts - can't compare names at all
        # Rely entirely on address similarity - be strict
        addr_score = max(addr_tok_jaccard, addr_tok_overlap * 0.85)
        num_score = addr_num_overlap
        
        # Need strong address evidence for cross-script matches
        if addr_score > 0.60 and num_score > 0.40:
            return True, 0.5 * addr_score + 0.5 * num_score
        if addr_score > 0.75:
            return True, addr_score
        return False, addr_score * 0.3
    
    # ---- Quick rejection: if both name AND address are very dissimilar, skip ----
    if name_ng_jaccard < 0.08 and addr_tok_jaccard < 0.15 and not first_word_match:
        return False, 0.0
    
    # ---- Compute Levenshtein features (more expensive but precise) ----
    name1 = s1['name_clean']
    name2 = s23['name_clean']
    addr1 = s1['addr_clean']
    addr2 = s23['addr_clean']
    
    name_lev = levenshtein_ratio(name1, name2)
    name_tok_sort = token_sort_ratio(name1, name2)
    name_tok_set = token_set_ratio(name1, name2)
    
    addr_lev = levenshtein_ratio(addr1, addr2)
    addr_tok_sort = token_sort_ratio(addr1, addr2)
    
    # Best name similarity
    best_name = max(name_lev, name_tok_sort, name_tok_set)
    best_addr = max(addr_lev, addr_tok_sort, addr_tok_jaccard)
    
    # ---- Precision-focused matching rules ----
    
    # Rule 1: Very strong name match + some address signal
    if best_name >= 0.88 and best_addr >= 0.30:
        return True, 0.6 * best_name + 0.4 * best_addr
    
    # Rule 2: Strong name + decent address
    if best_name >= 0.75 and best_addr >= 0.45:
        return True, 0.5 * best_name + 0.5 * best_addr
    
    # Rule 3: Good name + strong address
    if best_name >= 0.60 and best_addr >= 0.60:
        return True, 0.4 * best_name + 0.6 * best_addr
    
    # Rule 4: Strong token set match (handles word reordering + missing suffixes)
    if name_tok_set >= 0.85 and addr_tok_jaccard >= 0.30:
        return True, 0.5 * name_tok_set + 0.3 * addr_tok_jaccard + 0.2 * addr_num_overlap
    
    # Rule 5: Moderate name + strong address numbers (very discriminative)
    if best_name >= 0.50 and addr_num_overlap >= 0.65 and addr_tok_overlap >= 0.45:
        return True, 0.3 * best_name + 0.3 * addr_tok_overlap + 0.4 * addr_num_overlap
    
    # Rule 6: First word match + strong n-gram similarity (catches typos)
    if first_word_match and name_ng_jaccard >= 0.45 and best_addr >= 0.40:
        return True, 0.4 * name_ng_jaccard + 0.3 * best_addr + 0.3
    
    # Combined score with strict threshold
    combined = (0.30 * best_name + 0.25 * best_addr + 
                0.15 * name_ng_jaccard + 0.15 * addr_num_overlap +
                0.15 * (1.0 if first_word_match else 0.0))
    
    if combined >= 0.60:
        return True, combined
    
    return False, combined


# ============================================================================
# Pipeline
# ============================================================================

def load_records_by_country(filepath):
    """Load records grouped by country."""
    by_country = defaultdict(list)
    count = 0
    with open(filepath, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rec = process_record(row)
            by_country[rec['country']].append(rec)
            count += 1
            if count % 1000000 == 0:
                print(f"    Loaded {count // 1000000}M records...")
    print(f"    Total: {count} records, {len(by_country)} countries")
    return by_country, count


def run_pipeline_for_country(s1_records, s23_records, country):
    """Run the full blocking + matching pipeline for one country."""
    print(f"\n  --- {country.upper()}: {len(s1_records)} S1, {len(s23_records)} S2+S3 ---")
    
    # Build blocking index
    t0 = time.time()
    engine = BlockingEngine()
    for rec in s23_records:
        engine.add(rec)
    t1 = time.time()
    print(f"    Index built in {t1-t0:.1f}s | {engine.stats()}")
    
    # Build S2/S3 lookup
    s23_lookup = {rec['entity_id']: rec for rec in s23_records}
    
    # Generate candidates and match
    candidates = {}
    matches = {}
    total_cands = 0
    total_matches = 0
    
    report_every = max(len(s1_records) // 10, 1)
    
    for i, s1_rec in enumerate(s1_records):
        if (i + 1) % report_every == 0:
            print(f"    Progress: {i+1}/{len(s1_records)} "
                  f"({100*(i+1)//len(s1_records)}%) | "
                  f"cands: {total_cands}, matches: {total_matches}")
        
        s1_id = s1_rec['entity_id']
        
        # Blocking
        cand_ids = engine.query(s1_rec)
        candidates[s1_id] = cand_ids
        total_cands += len(cand_ids)
        
        # Matching
        matched = []
        for cid in cand_ids:
            s23_rec = s23_lookup.get(cid)
            if s23_rec is None:
                continue
            is_match, confidence = compute_match_score(s1_rec, s23_rec)
            if is_match:
                matched.append(cid)
        
        matches[s1_id] = matched
        total_matches += len(matched)
    
    t2 = time.time()
    avg_cands = total_cands / max(len(s1_records), 1)
    avg_matches = total_matches / max(len(s1_records), 1)
    print(f"    Done in {t2-t1:.1f}s | avg cands: {avg_cands:.1f}, "
          f"avg matches: {avg_matches:.2f}")
    
    return candidates, matches


def evaluate(matches, gt_dict):
    """Compute macro-averaged F0.5."""
    scores = []
    for s1_id, true_set in gt_dict.items():
        pred_set = set(matches.get(s1_id, []))
        if not true_set and not pred_set:
            scores.append(1.0)
        elif not true_set:
            scores.append(0.0)
        elif not pred_set:
            scores.append(0.0)
        else:
            tp = len(pred_set & true_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f05 = (1.25 * p * r) / (0.25 * p + r) if (p + r) > 0 else 0.0
            scores.append(f05)
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_blocking(candidates, gt_dict, s23_available):
    """Compute blocking recall."""
    found = 0
    total = 0
    for s1_id, true_set in gt_dict.items():
        cand_set = set(candidates.get(s1_id, []))
        available_true = true_set & s23_available
        found += len(available_true & cand_set)
        total += len(available_true)
    return found / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--train-sample', type=int, default=30000)
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("=" * 70)
    print("BUSINESS ENTITY RESOLUTION — PURE PYTHON")
    print("=" * 70)
    
    # ======================================================================
    # VALIDATION ON TRAINING DATA
    # ======================================================================
    if not args.test_only:
        print("\n" + "=" * 70)
        print("PHASE 1: VALIDATION ON TRAINING DATA")
        print("=" * 70)
        
        # Load ground truth
        print("\n  Loading ground truth...")
        gt_dict = {}
        with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                s1_id = row['source1_entity_id']
                matched = (row.get('matched_entity_ids', '') or '').strip()
                gt_dict[s1_id] = set(matched.split(',')) if matched else set()
        print(f"    {len(gt_dict)} entries")
        
        # Sample S1 IDs for validation
        import random
        random.seed(42)
        all_s1_ids = list(gt_dict.keys())
        sample_size = min(args.train_sample, len(all_s1_ids))
        sample_s1_ids = set(random.sample(all_s1_ids, sample_size))
        
        # Get relevant S2/S3 IDs
        gt_s23_ids = set()
        for s1_id in sample_s1_ids:
            gt_s23_ids.update(gt_dict[s1_id])
        
        # Load S1 sample
        print(f"\n  Loading S1 (sampling {sample_size})...")
        s1_by_country = defaultdict(list)
        with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                if row['entity_id'] in sample_s1_ids:
                    rec = process_record(row)
                    s1_by_country[rec['country']].append(rec)
        
        # Load S2+S3 (sample + all ground truth matches)
        S23_SAMPLE = 200000
        print(f"\n  Loading S2+S3 (target ~{S23_SAMPLE})...")
        all_s23 = []
        gt_s23 = []
        for src in ["train_source2.tsv", "train_source3.tsv"]:
            filepath = os.path.join(TRAIN_DIR, src)
            with open(filepath, encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    if row['entity_id'] in gt_s23_ids:
                        gt_s23.append(row)
                    elif len(all_s23) < S23_SAMPLE:
                        all_s23.append(row)
        
        # Combine: ground truth matches + sample
        remaining = S23_SAMPLE - len(gt_s23)
        if remaining > 0:
            all_s23 = gt_s23 + all_s23[:remaining]
        else:
            all_s23 = gt_s23
        print(f"    Loaded {len(all_s23)} S2+S3 records")
        
        # Process by country
        s23_by_country = defaultdict(list)
        s23_ids_available = set()
        for row in all_s23:
            rec = process_record(row)
            s23_by_country[rec['country']].append(rec)
            s23_ids_available.add(rec['entity_id'])
        
        del all_s23, gt_s23
        gc.collect()
        
        # Run pipeline per country
        all_candidates = {}
        all_matches = {}
        
        for country in sorted(set(s1_by_country.keys()) | set(s23_by_country.keys())):
            s1_recs = s1_by_country.get(country, [])
            s23_recs = s23_by_country.get(country, [])
            
            if not s1_recs:
                continue
            
            cands, matches = run_pipeline_for_country(s1_recs, s23_recs, country)
            all_candidates.update(cands)
            all_matches.update(matches)
        
        # Evaluate
        gt_sample = {s1_id: gt_dict[s1_id] for s1_id in sample_s1_ids}
        
        blocking_recall = evaluate_blocking(all_candidates, gt_sample, s23_ids_available)
        f05 = evaluate(all_matches, gt_sample)
        
        print(f"\n  {'='*50}")
        print(f"  VALIDATION RESULTS")
        print(f"  Blocking recall: {blocking_recall:.4f}")
        print(f"  F0.5 score: {f05:.4f}")
        print(f"  {'='*50}")
        
        if args.validate_only:
            elapsed = time.time() - start_time
            print(f"\n  Done in {elapsed/60:.1f} minutes")
            return
        
        # Clean up
        del s1_by_country, s23_by_country, all_candidates, all_matches, gt_dict
        gc.collect()
    
    # ======================================================================
    # TEST PREDICTION
    # ======================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: TEST PREDICTION")
    print("=" * 70)
    
    # We process by country to manage memory
    all_test_candidates = {}
    all_test_matches = {}
    all_s1_ids = []  # Preserve order
    
    # First, load S1 to know what countries we need
    print("\n  Loading test S1...")
    s1_by_country = defaultdict(list)
    s1_order = []
    with open(os.path.join(TEST_DIR, "test_source1.tsv"), encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rec = process_record(row)
            s1_by_country[rec['country']].append(rec)
            s1_order.append(rec['entity_id'])
    
    all_s1_ids = s1_order
    countries = sorted(s1_by_country.keys())
    print(f"    {len(s1_order)} entities, countries: {countries}")
    
    # Process each country
    for country in countries:
        s1_recs = s1_by_country[country]
        
        # Load S2+S3 for this country only
        print(f"\n  Loading S2+S3 for {country}...")
        s23_recs = []
        for src in ["test_source2.tsv", "test_source3.tsv"]:
            filepath = os.path.join(TEST_DIR, src)
            with open(filepath, encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    c = (row.get('country', '') or '').lower().strip()
                    if c == country:
                        s23_recs.append(process_record(row))
        
        print(f"    Loaded {len(s23_recs)} S2+S3 records for {country}")
        
        # Run pipeline
        cands, matches = run_pipeline_for_country(s1_recs, s23_recs, country)
        all_test_candidates.update(cands)
        all_test_matches.update(matches)
        
        # Clean up
        del s23_recs
        gc.collect()
    
    # Ensure matched IDs are subset of candidates
    for s1_id in all_test_matches:
        cand_set = set(all_test_candidates.get(s1_id, []))
        for mid in all_test_matches[s1_id]:
            if mid not in cand_set:
                all_test_candidates.setdefault(s1_id, []).append(mid)
    
    # Write output
    print("\n  Writing output...")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    match_path = os.path.join(OUT_DIR, "matching_results.tsv")
    with open(match_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['source1_entity_id', 'matched_entity_ids'])
        for eid in all_s1_ids:
            matched = all_test_matches.get(eid, [])
            writer.writerow([eid, ','.join(matched) if matched else ''])
    print(f"    Written: {match_path}")
    
    cand_path = os.path.join(OUT_DIR, "candidate_pairs.tsv")
    with open(cand_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['source1_entity_id', 'candidate_entity_ids'])
        for eid in all_s1_ids:
            cands = all_test_candidates.get(eid, [])
            writer.writerow([eid, ','.join(cands) if cands else ''])
    print(f"    Written: {cand_path}")
    
    # Stats
    total_matches = sum(len(v) for v in all_test_matches.values())
    total_cands = sum(len(v) for v in all_test_candidates.values())
    with_matches = sum(1 for v in all_test_matches.values() if v)
    
    print(f"\n  Summary:")
    print(f"    S1 entities: {len(all_s1_ids)}")
    print(f"    Total candidates: {total_cands} (avg {total_cands/len(all_s1_ids):.1f})")
    print(f"    Total matches: {total_matches} (avg {total_matches/len(all_s1_ids):.2f})")
    print(f"    S1 with matches: {with_matches}")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"DONE in {elapsed/60:.1f} minutes")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
