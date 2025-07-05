import json
import re
#Standardized test type categories
STANDARD_TYPES = {
    "cognitive": ["ability", "aptitude", "g+", "verify"],
    "personality": ["personality", "behavior", "behaviour"],
    "technical": ["knowledge", "skills", "developer", "engineering"],
    "behavioral": ["competencies", "competency"],
    "situational": ["situational judgement", "biodata"],
    "simulation": ["simulation"],
}
def extract_duration(raw):
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    try:
        match = re.search(r"(\d{1,3})\s*(min|minutes)?", str(raw).lower())
        return int(match.group(1)) if match else None
    except:
        return None
def map_test_types(raw):
    if not raw:
        return []
    raw = raw.lower()
    matched_types = set()
    for standard, keywords in STANDARD_TYPES.items():
        if any(keyword in raw for keyword in keywords):
            matched_types.add(standard)
    return sorted(list(matched_types))
def clean_metadata(input_path="shl_metadata_index.json", output_path="shl_metadata_index_cleaned.json"):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cleaned_count = 0
    for item in data:
        raw_duration = item.get("Duration", "")
        raw_type = item.get("Test Type", "")
        cleaned_duration = extract_duration(raw_duration)
        cleaned_types = map_test_types(raw_type)
        item["Duration"] = cleaned_duration if cleaned_duration is not None else None
        item["Test Type"] = cleaned_types if cleaned_types else []
        if cleaned_duration or cleaned_types:
            cleaned_count += 1
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Cleaned metadata saved to {output_path}")
    print(f"Total records cleaned: {cleaned_count} / {len(data)}")
if __name__ == "__main__":
    clean_metadata()
