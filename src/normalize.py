"""
Text Normalization Engine for Business Entity Resolution.
Provides deterministic, Unicode-aware, multilingual representations for business names,
addresses, and countries.
"""

import re
import unicodedata
from typing import List, Set, Optional, Tuple, Dict, Any

# ---------------------------------------------------------
# Configurable Suffix and Pattern Definitions
# ---------------------------------------------------------

# Legal suffixes across US, India, France, and international business entities.
# Sorted by length (descending) so multi-word suffixes match before single words.
LEGAL_SUFFIXES_RAW = [
    # Multi-word suffixes
    "private limited", "pvt limited", "pvt ltd", "pvt. ltd.", "pvt. ltd", 
    "private ltd", "co. ltd.", "co. ltd", "co ltd", "company limited",
    "limited liability company", "societe a responsabilite limitee",
    "societe par actions simplifiee", "societe civile immobiliere",
    "joint stock company",
    # Single-word suffixes & abbreviations
    "incorporated", "corporation", "limited", "private", "company",
    "enterprises", "enterprise", "solutions", "technologies", "services", 
    "holdings", "group",
    "inc.", "inc", "corp.", "corp", "llc.", "llc", "l.l.c.", "l.l.c", 
    "ltd.", "ltd", "pvt.", "pvt", "llp.", "llp", "plc.", "plc", "lp.", "lp",
    # French entity types
    "sarl", "s.a.r.l.", "s.a.r.l", "sas", "s.a.s.", "s.a.s", "sasu", "s.a.s.u.", 
    "sci", "s.c.i.", "sa", "s.a.", "eurl", "gie", "snc"
]

# Standardized regex pattern for removing legal suffixes at end of string or standalone
_SUFFIX_PATTERN_PARTS = [
    re.escape(s).replace(r"\ ", r"\s+").replace(r"\.", r"\.?") 
    for s in sorted(LEGAL_SUFFIXES_RAW, key=len, reverse=True)
]
LEGAL_SUFFIX_RE = re.compile(
    r'(?:^|\s+|[,(\[])(?:' + '|'.join(_SUFFIX_PATTERN_PARTS) + r')(?:$|\s+|[,)\]])',
    re.IGNORECASE
)

# Common domain extensions
DOMAIN_EXTENSIONS = (
    ".com", ".org", ".net", ".in", ".co.in", ".co", ".io", 
    ".ai", ".fr", ".biz", ".info", ".edu", ".gov", ".us"
)
DOMAIN_RE = re.compile(
    r'(?:https?://)?(?:www\.)?([a-zA-Z0-9_\-]+)\.(?:com|org|net|co\.in|co|in|fr|biz|info|io|ai|us|gov|edu)\b',
    re.IGNORECASE
)

# Noise brackets and symbols discovered in EDA
NOISE_PREFIX_RE = re.compile(r'^[@#<>\s\[\]]+')
NOISE_SYMBOLS_RE = re.compile(r'[\[\]#<>]')

# Postal code regex (generic 5 or 6 digit codes)
POSTAL_CODE_RE = re.compile(r'\b(\d{5,6})\b')

# Address numbers regex (building, plot, street, flat numbers)
# Captures: 123, 123A, 448A, 1/2, 3/115, 5-78, etc.
NUMBER_TOKEN_RE = re.compile(r'\b\d+(?:[/\-]\d+)?[a-zA-Z]?\b')
ORDINAL_RE = re.compile(r'(?<=\d)(?:st|nd|rd|th)\b', re.IGNORECASE)

# Acronym dot collapsing regex: L.L.C. -> LLC, P.V.T. -> PVT
ACRONYM_DOTS_RE = re.compile(r'(?<=\b[a-zA-Z])\.(?=[a-zA-Z]\b|\s|$)')

# Common address street-type abbreviations for standardization
STREET_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "dr": "drive",
    "blvd": "boulevard",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "pkwy": "parkway",
    "hwy": "highway",
    "sq": "square",
    "fl": "floor",
    "bldg": "building",
    "apt": "apartment",
    "ste": "suite",
    "no": "number",
    # French address terms
    "r": "rue",
    "bd": "boulevard",
    "all": "allee",
    "av": "avenue",
}

# ---------------------------------------------------------
# Core Unicode & Cleaning Functions
# ---------------------------------------------------------

def normalize_unicode(text: Optional[str]) -> str:
    """
    Applies NFKD decomposition, strips Latin diacritics (accents),
    while strictly keeping Indic vowels and non-Latin scripts intact.
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize('NFKD', text)
    res = []
    for c in decomposed:
        if unicodedata.combining(c):
            # Strip combining marks ONLY if applied to a Latin character (e.g. French accents)
            if res and 'LATIN' in unicodedata.name(res[-1], ''):
                continue
        res.append(c)
    return unicodedata.normalize('NFC', ''.join(res))


def clean_text(text: Optional[str]) -> str:
    """
    Standardizes whitespace, normalizes symbols (& -> and),
    collapses acronym dots, strips extraneous brackets and noise.
    Preserves all Unicode letters, numbers, and Indic vowel marks (Mn/Mc).
    """
    if not text:
        return ""
    # 1. Unicode decomposition with selective Latin diacritic folding
    s = normalize_unicode(text)
    # 2. Casefold for multilingual safety
    s = s.casefold()
    # 3. Collapse acronym dots before punctuation stripping: L.L.C. -> LLC
    s = ACRONYM_DOTS_RE.sub("", s)
    # 4. Noise symbols like [[LLC]], ##2476, << Team Ecole
    s = NOISE_SYMBOLS_RE.sub(" ", s)
    # 5. Ampersand normalization
    s = re.sub(r'\s*&\s*', ' and ', s)
    # 6. Preserve Letters (L), Marks (M - essential for Indic vowels), Numbers (N), and Whitespace
    # Replaces all punctuation and connectors (like _) with spaces
    char_list = []
    for c in s:
        cat = unicodedata.category(c)
        if cat[0] in ('L', 'M', 'N') or c in (' ', '\t', '\n'):
            char_list.append(c)
        else:
            char_list.append(' ')
    s = ''.join(char_list)
    # 7. Collapse multiple whitespaces
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# ---------------------------------------------------------
# Business Name Normalization Representations
# ---------------------------------------------------------

def normalize_domain_or_handle(text: Optional[str]) -> str:
    """
    Extracts core business name from domain names (e.g. strategicpraetorian.com -> strategicpraetorian)
    and social handles (@smartraj -> smartraj, @apex_labs -> apex labs).
    """
    if not text:
        return ""
    s = text.strip()
    # Handle domain
    m = DOMAIN_RE.search(s)
    if m:
        core = m.group(1)
        return clean_text(core.replace("_", " ").replace("-", " "))
    # Handle social handle
    if s.startswith("@"):
        core = s[1:].strip()
        return clean_text(core.replace("_", " ").replace("-", " "))
    return clean_text(s)


def strip_legal_suffixes(text: Optional[str]) -> str:
    """
    Removes recognized legal entity suffixes (e.g. Inc, LLC, Pvt Ltd, SARL, SAS)
    using boundary-aware token matching. Retains original words if only suffix exists.
    """
    if not text:
        return ""
    cleaned = clean_text(text)
    # Apply regex iteratively up to 2 times to handle compound suffixes (e.g. Co Ltd)
    prev = None
    curr = cleaned
    for _ in range(2):
        prev = curr
        curr = LEGAL_SUFFIX_RE.sub(" ", f" {curr} ").strip()
        curr = re.sub(r'\s+', ' ', curr)
        if not curr:  # Do not reduce string to empty if it was only a suffix
            return prev
    return curr if curr else cleaned


def to_alphanumeric(text: Optional[str]) -> str:
    """
    Returns compressed alphanumeric-only string: lowercase, no spaces, no punctuation.
    """
    if not text:
        return ""
    cleaned = clean_text(text)
    # Filter to only alphanumeric characters (Category L, M, N)
    return ''.join(c for c in cleaned if unicodedata.category(c)[0] in ('L', 'M', 'N'))


def tokenize(text: Optional[str]) -> List[str]:
    """
    Deterministic Unicode tokenization: splits on non-alphanumeric separators.
    Preserves all alphanumeric tokens (including short tokens like AI, UK, 3D).
    """
    if not text:
        return []
    cleaned = clean_text(text)
    return [t for t in cleaned.split() if t]


def to_sorted_tokens(text: Optional[str], remove_suffixes: bool = False) -> str:
    """
    Produces word-order invariant representation by sorting unique tokens.
    Example: 'XX Apex Nippon' and 'XX Nippon Apex' both yield 'apex nippon xx'.
    """
    if not text:
        return ""
    target = strip_legal_suffixes(text) if remove_suffixes else clean_text(text)
    toks = tokenize(target)
    if not toks:
        return ""
    return " ".join(sorted(toks))


def normalize_business_name(raw_name: Optional[str]) -> Dict[str, Any]:
    """
    Generates all standardized representations for a business name record.
    Preserves raw value while generating clean, suffix-stripped, sorted, and alphanumeric views.
    """
    raw = raw_name or ""
    nfkd = normalize_unicode(raw)
    casefold_val = raw.casefold()
    clean_val = clean_text(raw)
    no_suffix_val = strip_legal_suffixes(raw)
    domain_val = normalize_domain_or_handle(raw)
    alpha_val = to_alphanumeric(no_suffix_val or clean_val)
    tokens_list = tokenize(no_suffix_val or clean_val)
    sorted_tokens_val = to_sorted_tokens(raw, remove_suffixes=True)

    return {
        "name_original": raw,
        "name_nfkd": nfkd,
        "name_casefold": casefold_val,
        "name_clean": clean_val,
        "name_no_legal_suffix": no_suffix_val,
        "name_domain": domain_val,
        "name_alphanumeric": alpha_val,
        "name_tokens": tokens_list,
        "name_sorted_tokens": sorted_tokens_val
    }


# ---------------------------------------------------------
# Address Normalization & Extraction Representations
# ---------------------------------------------------------

def extract_numbers(address: Optional[str]) -> List[str]:
    """
    Extracts all numerical identifiers (house number, plot number, building number).
    Normalizes ordinals: 2Nd -> 2, 1st -> 1, 3rd -> 3.
    """
    if not address:
        return []
    # Strip ordinals so 2nd Main matches 2 Main
    normalized_ordinals = ORDINAL_RE.sub("", address)
    return NUMBER_TOKEN_RE.findall(normalized_ordinals)


def extract_postal_code(address: Optional[str]) -> Optional[str]:
    """
    Extracts generic postal code (5 to 6 digits, e.g. US 5-digit ZIP, India 6-digit PIN, France 5-digit).
    Picks the last matching 5-6 digit token, which typically corresponds to the postal code.
    """
    if not address:
        return None
    matches = POSTAL_CODE_RE.findall(address)
    return matches[-1] if matches else None


def normalize_address(raw_address: Optional[str]) -> Dict[str, Any]:
    """
    Generates all standardized representations and extracted components for an address.
    """
    if raw_address is None or raw_address.strip() == "":
        return {
            "address_original": "",
            "address_clean": "",
            "address_alphanumeric": "",
            "address_tokens": [],
            "address_numbers": [],
            "postal_code": None,
            "is_address_missing": True
        }

    raw = raw_address.strip()
    clean_val = clean_text(raw)
    
    # Standardize common street abbreviations
    toks = clean_val.split()
    standardized_toks = [STREET_ABBREVIATIONS.get(t, t) for t in toks]
    std_clean = " ".join(standardized_toks)
    
    alpha_val = to_alphanumeric(std_clean)
    num_tokens = extract_numbers(raw)
    pin = extract_postal_code(raw)

    return {
        "address_original": raw,
        "address_clean": std_clean,
        "address_alphanumeric": alpha_val,
        "address_tokens": standardized_toks,
        "address_numbers": num_tokens,
        "postal_code": pin,
        "is_address_missing": False
    }


# ---------------------------------------------------------
# Country Normalization
# ---------------------------------------------------------

def normalize_country(raw_country: Optional[str]) -> Dict[str, str]:
    """
    Generic country normalization without hardcoded whitelist.
    Handles any current or future country string (US, India, France, etc.).
    """
    raw = raw_country or ""
    clean_c = normalize_unicode(raw).strip().upper()
    return {
        "country_original": raw,
        "country_normalized": clean_c
    }


# ---------------------------------------------------------
# Full Record Normalizer
# ---------------------------------------------------------

def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes an entire business entity record containing:
    entity_id, business_name, business_address, country.
    """
    name_dict = normalize_business_name(record.get("business_name"))
    addr_dict = normalize_address(record.get("business_address"))
    c_dict = normalize_country(record.get("country"))

    res = {
        "entity_id": record.get("entity_id", "")
    }
    res.update(name_dict)
    res.update(addr_dict)
    res.update(c_dict)
    return res
