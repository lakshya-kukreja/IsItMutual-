import requests

url = "http://localhost:9696"

# A participant with every feature filled in
full_participant = {
    # Profile
    "gender": 0, "age": 22, "age_o": 24,
    "race": 2, "race_o": 4, "samerace": 0,
    "field_cd": 1, "goal": 2, "date": 5, "go_out": 3,
    # What they value in a partner
    "attr1_1": 30, "sinc1_1": 15, "intel1_1": 20,
    "fun1_1": 20, "amb1_1": 5, "shar1_1": 10,
    # Impressions after the date
    "attr": 8, "sinc": 7, "intel": 8,
    "fun": 9, "amb": 6, "shar": 7,
    "like": 8, "prob": 7, "met": 2
}

# Only the most important post-date ratings (the rest are imputed)
partial_participant = {
    "like": 9, "attr": 9, "fun": 9, "prob": 8
}

# Low ratings
low_participant = {
    "like": 2, "attr": 3, "fun": 2, "prob": 1
}


def predict(name, participant):
    response = requests.post(f"{url}/predict", json=participant)
    print(f"\n{name}")
    print("Status:", response.status_code)
    print("Response:", response.json())
    return response


# 1. Health check
health = requests.get(f"{url}/")
print("Health check:", health.json())
assert health.status_code == 200

# 2. Valid requests
r_full = predict("Full participant", full_participant)
r_partial = predict("Partial participant", partial_participant)
r_low = predict("Low-rating participant", low_participant)

for r in (r_full, r_partial, r_low):
    result = r.json()
    assert r.status_code == 200
    assert 0.0 <= result["mutual_match_probability"] <= 1.0
    assert result["mutual_match"] == (
        result["mutual_match_probability"] >= result["threshold"]
    )

# High ratings should score higher than low ratings
assert (
    r_partial.json()["mutual_match_probability"]
    > r_low.json()["mutual_match_probability"]
)

# 3. Invalid input (text instead of a number) must be rejected
r_bad = predict("Invalid input", {"like": "abc"})
assert r_bad.status_code == 422

print("\nAll tests passed.")