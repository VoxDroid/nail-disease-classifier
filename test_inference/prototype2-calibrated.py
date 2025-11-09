import pandas as pd
from collections import defaultdict
import numpy as np
from numpy.linalg import pinv

# Load the CSV file
df = pd.read_csv("StatisticalDataset.csv")
df = df.fillna(value=pd.NA)

# Parse the data into a dictionary
feature_to_diseases = defaultdict(list)
for _, row in df.iterrows():
    f = row['Nail Feature']
    d = row['Associated Disease/Condition']
    p_fd = row['P(Nail | Disease) 0-1%']
    p_d = row['P(Disease) 0-1%']
    if pd.isna(p_d):
        if d == 'No systemic disease':
            p_d = 1.0
        else:
            p_d = 0.0
            continue  # Skip if no P(Disease)
    p_female = row['P(Disease) Sex_Female 0-1'] if not pd.isna(row['P(Disease) Sex_Female 0-1']) else None
    p_male = row['P(Disease) Sex_Male 0-1'] if not pd.isna(row['P(Disease) Sex_Male 0-1']) else None
    age_mean = row['Age (Mean)'] if not pd.isna(row['Age (Mean)']) else None
    age_low = row['Age_Low'] if not pd.isna(row['Age_Low']) else None
    age_high = row['Age_High'] if not pd.isna(row['Age_High']) else None
    feature_to_diseases[f].append({
        'disease': d,
        'p_fd': p_fd,
        'p_d': p_d,
        'p_female': p_female,
        'p_male': p_male,
        'age_low': age_low,
        'age_high': age_high,
        'age_mean': age_mean
    })

# Define the order of labels matching the confusion matrix
labels = [
    "Melanonychia",
    "Beau's Lines",
    "Blue Finger (Cyanosis)",
    "Clubbing",
    "Healthy Nail",
    "Koilonychia",
    "Muehrcke's Lines",
    "Onychogryphosis",
    "Pitting",
    "Terry's Nails"
]

# Confidence scores #1 (to be replaced with the model output dun sa system)
# confidence_str = {
#     'Melanonychia': 0.01,
#     "Beau's Lines": 0.03,
#     'Blue Finger (Cyanosis)': 0.04,
#     'Clubbing': 0.00,
#     'Healthy Nail': 0.28,
#     'Koilonychia': 99.62,
#     "Muehrcke's Lines": 0.00,
#     'Onychogryphosis': 0.00,
#     'Pitting': 0.00,
#     "Terry's Nails": 0.00
# }

confidence_str = {
    'Melanonychia': 0.00,
    "Beau's Lines": 0.00,
    'Blue Finger (Cyanosis)': 0.00,
    'Clubbing': 5.00,
    'Healthy Nail': 95.00,
    'Koilonychia': 0.00,
    "Muehrcke's Lines": 0.00,
    'Onychogryphosis': 0.00,
    'Pitting': 0.00,
    "Terry's Nails": 0.00
}

# Convert to probabilities (0-1)
confidence = {k: v / 100 for k, v in confidence_str.items()}

# Extract q vector from confidence dict in label order
q = np.array([confidence.get(label, 0.0) for label in labels])

# Confusion matrix counts (rows: true, columns: predicted)
counts = np.array([
    [34, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 22, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 25, 1, 1, 0, 0, 0, 0, 2],
    [0, 3, 0, 33, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 30, 0, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 14, 0, 1, 2, 0],
    [1, 1, 0, 0, 0, 0, 14, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 33, 0, 0],
    [0, 3, 0, 1, 0, 0, 0, 0, 28, 0],
    [0, 0, 0, 0, 5, 0, 0, 1, 0, 36]
])

# Normalize to get Conf: P(pred | true)
row_sums = counts.sum(axis=1, keepdims=True)
conf = counts / row_sums

# Adjust q to estimate true probabilities using pseudoinverse
adjusted_p = pinv(conf) @ q
adjusted_p = np.maximum(adjusted_p, 0.0)  # Clip negative values
if adjusted_p.sum() > 0:
    adjusted_p /= adjusted_p.sum()  # Normalize to sum to 1

# Convert back to dict
adjusted_confidence = {labels[i]: adjusted_p[i] for i in range(len(labels))}

# User prompt
sex = input("Enter your sex (male/female): ").lower()
age = float(input("Enter your age: "))

# Compute posteriors using adjusted_confidence
disease_to_post = defaultdict(float)
for f, entries in feature_to_diseases.items():
    p_f_image = adjusted_confidence.get(f, 0)
    if p_f_image == 0:
        continue
    unnorm = {}
    sum_unnorm = 0.0
    for entry in entries:
        d = entry['disease']
        p_fd = entry['p_fd']
        if pd.isna(p_fd) or p_fd == 0:
            continue
        p_d = entry['p_d']
        p_female = entry['p_female']
        p_male = entry['p_male']
        if p_female is None and p_male is None:
            effective_p_d = p_d
        else:
            if sex == 'female':
                p_sex = p_female if p_female is not None else 0.0
            else:
                p_sex = p_male if p_male is not None else 0.0
            if p_female is not None and p_male is not None:
                sum_p = p_female + p_male
                if abs(sum_p - 1) < 0.05:
                    is_p_sex_d = True
                else:
                    is_p_sex_d = False
            else:
                is_p_sex_d = False
            if is_p_sex_d:
                effective_p_d = p_d * p_sex
            else:
                effective_p_d = p_sex
        if effective_p_d == 0:
            continue
        # Age adjustment
        p_age_d = 1.0
        low = entry['age_low']
        high = entry['age_high']
        if low is not None and high is not None:
            if high > low:
                if low <= age <= high:
                    p_age_d = 1.0 / (high - low)
                else:
                    p_age_d = 0.0
            else:
                if age == low:
                    p_age_d = 1.0
                else:
                    p_age_d = 0.0
        if p_age_d == 0:
            continue
        effective_prior = effective_p_d * p_age_d
        unnorm_d = p_fd * effective_prior
        unnorm[d] = unnorm_d
        sum_unnorm += unnorm_d
    if sum_unnorm > 0:
        for d, u in unnorm.items():
            p_d_f = u / sum_unnorm
            disease_to_post[d] += p_d_f * p_f_image

# Normalize the final posteriors
total_post = sum(disease_to_post.values())
if total_post > 0:
    for d in disease_to_post:
        disease_to_post[d] /= total_post

# Sort by probability descending
sorted_diseases = sorted(disease_to_post.items(), key=lambda x: x[1], reverse=True)

# Output
print("\nPotential Systemic Diseases (Probabilities):")
for d, p in sorted_diseases:
    print(f"{d}: {p * 100:.2f}%")