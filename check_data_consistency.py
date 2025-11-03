import pandas as pd
import numpy as np

# Load the dataset
print("Loading dataset...")
data = pd.read_csv('rwanda_early_sexual_debut_dataset.csv')
print(f"Dataset loaded: {data.shape[0]:,} rows, {data.shape[1]} columns\n")

print("="*80)
print("COMPREHENSIVE DATA CONSISTENCY CHECKS")
print("="*80)

# ===== 1. LOGICAL INCONSISTENCIES =====
print("\n" + "="*80)
print("1. LOGICAL INCONSISTENCIES")
print("="*80)

# Check 1.1: Age at first birth < Age at first sex
inconsistent_ages = data[
    (data['v525'] > 0) & (data['v525'] < 50) &  # Has valid age at first sex
    (data['v531'].notna()) & (data['v531'] > 0) & (data['v531'] < 50) &  # Has valid age at first birth
    (data['v531'] < data['v525'])  # Birth happened before first sex
]
print(f"\n1.1 Age at first birth < Age at first sex: {len(inconsistent_ages)} cases")
if len(inconsistent_ages) > 0:
    print("     ⚠️  WARNING: These women gave birth before having sex (biologically impossible)")
    print(f"     Sample cases (first 5):")
    print(inconsistent_ages[['caseid', 'v525', 'v531', 'v012']].head())

# Check 1.2: Age at first sex or birth > Current age
inconsistent_sex_age = data[
    (data['v525'] > 0) & (data['v525'] < 50) & 
    (data['v525'] > data['v012'])
]
print(f"\n1.2 Age at first sex > Current age: {len(inconsistent_sex_age)} cases")
if len(inconsistent_sex_age) > 0:
    print("     ⚠️  WARNING: First sex occurred in the future")
    print(f"     Sample cases:")
    print(inconsistent_sex_age[['caseid', 'v525', 'v012']].head())

inconsistent_birth_age = data[
    (data['v531'].notna()) & (data['v531'] > 0) & (data['v531'] < 50) &
    (data['v531'] > data['v012'])
]
print(f"\n1.3 Age at first birth > Current age: {len(inconsistent_birth_age)} cases")
if len(inconsistent_birth_age) > 0:
    print("     ⚠️  WARNING: First birth occurred in the future")

# Check 1.4: Women who gave birth but never had sex
gave_birth_no_sex = data[
    (data['v201'] > 0) &  # Has children
    ((data['v525'] == 0) | (data['v525'].isna()) | (data['v525'] >= 50))  # Never had sex
]
print(f"\n1.4 Women with children but no recorded sexual debut: {len(gave_birth_no_sex)} cases")
if len(gave_birth_no_sex) > 0:
    print("     ⚠️  WARNING: Biologically impossible - may indicate data entry errors")
    print(f"     Sample cases:")
    print(gave_birth_no_sex[['caseid', 'v525', 'v531', 'v201', 'v012']].head())

# Check 1.5: Never had sex but is currently pregnant
never_sex_pregnant = data[
    ((data['v525'] == 0) | (data['v525'].isna()) | (data['v525'] >= 50)) &
    (data['v213'] == 1)  # Currently pregnant
]
print(f"\n1.5 Never had sex but currently pregnant: {len(never_sex_pregnant)} cases")
if len(never_sex_pregnant) > 0:
    print("     ⚠️  WARNING: Biologically impossible")

# Check 1.6: Age at first cohabitation before age at first sex
inconsistent_union = data[
    (data['v525'] > 0) & (data['v525'] < 50) &
    (data['v511'].notna()) & (data['v511'] > 0) & (data['v511'] < 50) &
    (data['v511'] < data['v525'])
]
print(f"\n1.6 Age at first union < Age at first sex: {len(inconsistent_union)} cases")
if len(inconsistent_union) > 0:
    print("     ℹ️  NOTE: Union before sex (culturally possible but unusual)")

# ===== 2. OUT-OF-RANGE VALUES =====
print("\n" + "="*80)
print("2. OUT-OF-RANGE VALUES")
print("="*80)

# Check 2.1: Age at first sex
extreme_sex_age_low = data[(data['v525'] > 0) & (data['v525'] < 10)]
extreme_sex_age_high = data[(data['v525'] > 40) & (data['v525'] < 50)]
print(f"\n2.1 Age at first sex:")
print(f"     - Extremely young (<10 years): {len(extreme_sex_age_low)} cases")
if len(extreme_sex_age_low) > 0:
    print(f"       Ages: {sorted(extreme_sex_age_low['v525'].unique())}")
print(f"     - Very late (>40 years): {len(extreme_sex_age_high)} cases")

# Check 2.2: Age at first birth
extreme_birth_age_low = data[(data['v531'] > 0) & (data['v531'] < 12) & (data['v531'] < 50)]
extreme_birth_age_high = data[(data['v531'] > 45) & (data['v531'] < 50)]
print(f"\n2.2 Age at first birth:")
print(f"     - Very young (<12 years): {len(extreme_birth_age_low)} cases")
if len(extreme_birth_age_low) > 0:
    print(f"       Ages: {sorted(extreme_birth_age_low['v531'].unique())}")
print(f"     - Very late (>45 years): {len(extreme_birth_age_high)} cases")

# Check 2.3: Current age distribution
age_outliers = data[(data['v012'] < 15) | (data['v012'] > 49)]
print(f"\n2.3 Current age outside expected range (15-49): {len(age_outliers)} cases")

# ===== 3. DERIVED VARIABLE CONSISTENCY =====
print("\n" + "="*80)
print("3. DERIVED VARIABLE CONSISTENCY CHECKS")
print("="*80)

# Check 3.1: Early sexual debut flag consistency
has_sex = (data['v525'] > 0) & (data['v525'] < 50)
early_debut_check = data[has_sex].copy()
early_debut_check['computed_early'] = (early_debut_check['v525'] < 18).astype(int)
mismatched_early = early_debut_check[
    early_debut_check['early_sexual_debut'].notna() &
    (early_debut_check['early_sexual_debut'] != early_debut_check['computed_early'])
]
print(f"\n3.1 Early sexual debut flag mismatch: {len(mismatched_early)} cases")

# Check 3.2: Sexual debut category consistency
category_mismatch = data[has_sex].copy()
category_mismatch['computed_category'] = 'Normal/Late debut (≥18)'
category_mismatch.loc[category_mismatch['v525'] < 18, 'computed_category'] = 'Early debut (<18)'
category_issues = category_mismatch[
    category_mismatch['sexual_debut_category'] != category_mismatch['computed_category']
]
print(f"\n3.2 Sexual debut category mismatch: {len(category_issues)} cases")

# Check 3.3: Women marked as "Never had sex" but have v525 values
never_had_sex_with_age = data[
    (data['sexual_debut_category'] == 'Never had sex') &
    (data['v525'] > 0) & (data['v525'] < 50)
]
print(f"\n3.3 'Never had sex' but has age at first sex: {len(never_had_sex_with_age)} cases")

# ===== 4. MISSING DATA PATTERNS =====
print("\n" + "="*80)
print("4. SUSPICIOUS MISSING DATA PATTERNS")
print("="*80)

# Check 4.1: Women with children but missing age at first birth
children_no_birth_age = data[
    (data['v201'] > 0) &
    ((data['v531'].isna()) | (data['v531'] == 0) | (data['v531'] >= 50))
]
print(f"\n4.1 Women with children but missing age at first birth: {len(children_no_birth_age)} cases")
print(f"     ({len(children_no_birth_age)/len(data)*100:.2f}% of dataset)")

# Check 4.2: Missing both sex and birth ages for women with children
missing_both = data[
    (data['v201'] > 0) &
    ((data['v525'] == 0) | (data['v525'].isna()) | (data['v525'] >= 50)) &
    ((data['v531'].isna()) | (data['v531'] == 0) | (data['v531'] >= 50))
]
print(f"\n4.2 Women with children but missing both sexual debut AND first birth age: {len(missing_both)} cases")

# ===== 5. MARRIAGE/PARTNERSHIP INCONSISTENCIES =====
print("\n" + "="*80)
print("5. MARRIAGE/PARTNERSHIP INCONSISTENCIES")
print("="*80)

# Check 5.1: Never married but has children
never_married_children = data[
    (data['v501'] == 0) &  # Never married
    (data['v201'] > 0)  # Has children
]
print(f"\n5.1 Never married but has children: {len(never_married_children)} cases")
print(f"     ℹ️  NOTE: This is possible (out-of-wedlock births)")

# Check 5.2: Currently married/in union but partner age missing
married_no_partner_age = data[
    (data['v501'].isin([1, 2])) &  # Married or living together
    (data['v701'].isna() | (data['v701'] == 0))
]
print(f"\n5.2 Currently married/in union but partner age missing: {len(married_no_partner_age)} cases")

# ===== 6. STATISTICAL OUTLIERS =====
print("\n" + "="*80)
print("6. STATISTICAL OUTLIERS")
print("="*80)

# Check 6.1: Very large intervals between sex and birth
has_both = (
    (data['v525'] > 0) & (data['v525'] < 50) &
    (data['v531'].notna()) & (data['v531'] > 0) & (data['v531'] < 50)
)
data_with_both = data[has_both].copy()
data_with_both['sex_birth_gap'] = data_with_both['v531'] - data_with_both['v525']

large_gaps = data_with_both[data_with_both['sex_birth_gap'] > 15]
print(f"\n6.1 Interval between first sex and first birth >15 years: {len(large_gaps)} cases")
if len(large_gaps) > 0:
    print(f"     Mean gap: {data_with_both['sex_birth_gap'].mean():.1f} years")
    print(f"     Max gap: {data_with_both['sex_birth_gap'].max():.0f} years")

# Check 6.2: Very young mothers (potential child abuse cases)
very_young_mothers = data[(data['v531'] > 0) & (data['v531'] < 14)]
print(f"\n6.2 First birth before age 14: {len(very_young_mothers)} cases")
if len(very_young_mothers) > 0:
    print(f"     ⚠️  WARNING: May indicate child sexual abuse")
    print(f"     Ages at first birth: {sorted(very_young_mothers['v531'].unique())}")

# ===== 7. WEIGHT AND SAMPLING ISSUES =====
print("\n" + "="*80)
print("7. SAMPLING WEIGHT ISSUES")
print("="*80)

# Check 7.1: Missing or zero weights
missing_weights = data[data['v005'].isna() | (data['v005'] == 0)]
print(f"\n7.1 Missing or zero sampling weights: {len(missing_weights)} cases")

# Check 7.2: Extreme weights
if data['v005'].notna().any():
    weight_stats = data['v005'].describe()
    extreme_weights = data[
        (data['v005'] < weight_stats['25%'] - 3 * (weight_stats['75%'] - weight_stats['25%'])) |
        (data['v005'] > weight_stats['75%'] + 3 * (weight_stats['75%'] - weight_stats['25%']))
    ]
    print(f"\n7.2 Extreme sampling weights (>3 IQR from quartiles): {len(extreme_weights)} cases")

# ===== SUMMARY =====
print("\n" + "="*80)
print("SUMMARY OF CRITICAL ISSUES")
print("="*80)

critical_issues = []
if len(inconsistent_ages) > 0:
    critical_issues.append(f"• {len(inconsistent_ages)} cases: Birth before first sex")
if len(gave_birth_no_sex) > 0:
    critical_issues.append(f"• {len(gave_birth_no_sex)} cases: Children but no sexual debut recorded")
if len(never_sex_pregnant) > 0:
    critical_issues.append(f"• {len(never_sex_pregnant)} cases: Pregnant but never had sex")
if len(extreme_birth_age_low) > 0:
    critical_issues.append(f"• {len(extreme_birth_age_low)} cases: First birth before age 12")
if len(mismatched_early) > 0:
    critical_issues.append(f"• {len(mismatched_early)} cases: Early debut flag mismatch")

if len(critical_issues) > 0:
    print("\n⚠️  CRITICAL ISSUES FOUND:")
    for issue in critical_issues:
        print(issue)
else:
    print("\n✓ No critical logical inconsistencies found!")

print("\n" + "="*80)
print("DATA QUALITY ASSESSMENT COMPLETE")
print("="*80)

# Export problematic cases for manual review
if len(inconsistent_ages) > 0 or len(gave_birth_no_sex) > 0:
    print("\n📊 Exporting problematic cases for manual review...")
    
    all_problems = pd.concat([
        inconsistent_ages.assign(issue_type='Birth before first sex'),
        gave_birth_no_sex.assign(issue_type='Children but no sex recorded'),
        very_young_mothers.assign(issue_type='Very young mother (<14)')
    ], ignore_index=True)
    
    all_problems[['caseid', 'v525', 'v531', 'v012', 'v201', 'issue_type']].to_csv(
        'data_inconsistencies_flagged.csv', 
        index=False
    )
    print("✓ Saved to: data_inconsistencies_flagged.csv")

print("\nRecommendations:")
print("1. Review flagged cases manually to determine if they are data entry errors")
print("2. Consider excluding biologically impossible cases from analysis")
print("3. Document all data cleaning decisions for reproducibility")
print("4. Use sample weights (v005) in all statistical analyses")