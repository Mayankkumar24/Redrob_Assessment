import json

file_path = r"c:\Users\Neeraj\Downloads\redrob-ranker\flagged_profiles_HARD_only.json"

cert_count = 0
career_count = 0
expert_count = 0
other_count = 0

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for profile in data:
            reasons = profile.get("honeypot_reasons", [])
            if "certification" in str(reasons):
                cert_count += 1
            elif "career_history" in str(reasons):
                career_count += 1
            elif "expert" in str(reasons):
                expert_count += 1
            else:
                other_count += 1
        

    print(f"Certificate reasons count: {cert_count}")
    print(f"Career history reasons count: {career_count}")
    print(f"Expert proficiency reasons count: {expert_count}")
    print(f"Other reasons count: {other_count}")
except Exception as e:
    print(f"Error: {e}")
