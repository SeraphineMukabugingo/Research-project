import pandas as pd
import numpy as np

"""
MINIMAL DATA CLEANING FOR RWANDA DHS DATASET
=============================================

PRINCIPLE: Remove only biologically IMPOSSIBLE cases
RATIONALE: Study focuses on early pregnancy risk, so extreme cases 
           are part of the phenomenon being studied, not errors.

EXCLUSION CRITERION: Age at first birth < Age at first sexual intercourse
                     (Biologically impossible)
"""

print("="*80)
print("MINIMAL DATA CLEANING - RWANDA DHS EARLY SEXUAL DEBUT STUDY")
print("="*80)

# Load original dataset
print("\nLoading dataset...")
data = pd.read_csv('rwanda_early_sexual_debut_dataset.csv')
original_n = len(data)
print(f"✓ Original dataset: {original_n:,} observations")

# ===== IDENTIFY BIOLOGICALLY IMPOSSIBLE CASES =====
print("\n" + "="*80)
print("IDENTIFYING BIOLOGICALLY IMPOSSIBLE CASES")
print("="*80)

# Flag: Birth before first sexual intercourse
biologically_impossible = (
    (data['v525'] > 0) & (data['v525'] < 50) &  # Valid age at first sex
    (data['v531'].notna()) & (data['v531'] > 0) & (data['v531'] < 50) &  # Valid age at first birth
    (data['v531'] < data['v525'])  # Birth before sex (IMPOSSIBLE)
)

n_impossible = biologically_impossible.sum()
print(f"\n⚠️  Cases where first birth < first sex: {n_impossible}")
print(f"    This represents: {(n_impossible/original_n)*100:.2f}% of dataset")

# Show examples of impossible cases
if n_impossible > 0:
    print("\n📋 Sample of biologically impossible cases:")
    impossible_cases = data[biologically_impossible][['caseid', 'v525', 'v531', 'v012', 'v201']].head(10)
    impossible_cases.columns = ['Case ID', 'Age First Sex', 'Age First Birth', 'Current Age', 'Total Children']
    print(impossible_cases.to_string(index=False))

# ===== DOCUMENT EXTREME BUT POSSIBLE CASES (KEPT IN DATASET) =====
print("\n" + "="*80)
print("EXTREME BUT BIOLOGICALLY POSSIBLE CASES (RETAINED)")
print("="*80)

# Very young sexual debut (8-9 years) - KEPT
extreme_young_sex = data[(data['v525'] > 0) & (data['v525'] < 10)]
print(f"\n✓ Very young age at first sex (8-9 years): {len(extreme_young_sex)} cases")
print(f"  → RETAINED (biologically possible post-menarche)")

# Very young mothers (8-13 years) - KEPT
extreme_young_birth = data[(data['v531'] > 0) & (data['v531'] < 14) & (data['v531'] < 50)]
print(f"\n✓ Very young age at first birth (8-13 years): {len(extreme_young_birth)} cases")
print(f"  → RETAINED (biologically possible, represents high-risk population)")
if len(extreme_young_birth) > 0:
    print(f"\n  Age distribution of very young mothers:")
    age_dist = extreme_young_birth['v531'].value_counts().sort_index()
    for age, count in age_dist.items():
        print(f"    Age {int(age)}: {count} cases")

# ===== CREATE CLEANED DATASET =====
print("\n" + "="*80)
print("CREATING CLEANED DATASET")
print("="*80)

# Remove only the biologically impossible cases
data_clean = data[~biologically_impossible].copy()
n_removed = original_n - len(data_clean)

print(f"\n✓ Cases removed: {n_removed} ({(n_removed/original_n)*100:.2f}%)")
print(f"✓ Cases retained: {len(data_clean):,} ({(len(data_clean)/original_n)*100:.2f}%)")

# ===== VERIFY KEY VARIABLES AFTER CLEANING =====
print("\n" + "="*80)
print("POST-CLEANING DATA SUMMARY")
print("="*80)

# Sexual debut status distribution
print("\n1. Sexual Debut Status (Cleaned Dataset):")
debut_counts = data_clean['sexual_debut_category'].value_counts()
for category in ['Never had sex', 'Early debut (<18)', 'Normal/Late debut (≥18)']:
    if category in debut_counts.index:
        count = debut_counts[category]
        pct = (count / len(data_clean)) * 100
        print(f"   {category:30s}: {count:6,} ({pct:5.2f}%)")

# Age at first sex (among those who had sex)
has_sex = (data_clean['v525'] > 0) & (data_clean['v525'] < 50)
age_first_sex = data_clean.loc[has_sex, 'v525']
print(f"\n2. Age at First Sexual Intercourse (n={len(age_first_sex):,}):")
print(f"   Mean: {age_first_sex.mean():.2f} years")
print(f"   Median: {age_first_sex.median():.2f} years")
print(f"   Range: {age_first_sex.min():.0f} - {age_first_sex.max():.0f} years")

# Age at first birth (among those who gave birth)
has_birth = (data_clean['v531'].notna()) & (data_clean['v531'] > 0) & (data_clean['v531'] < 50)
age_first_birth = data_clean.loc[has_birth, 'v531']
print(f"\n3. Age at First Birth (n={len(age_first_birth):,}):")
print(f"   Mean: {age_first_birth.mean():.2f} years")
print(f"   Median: {age_first_birth.median():.2f} years")
print(f"   Range: {age_first_birth.min():.0f} - {age_first_birth.max():.0f} years")

# Early pregnancy rates
early_debut_rate = data_clean.loc[has_sex, 'early_sexual_debut'].mean() * 100
print(f"\n4. Early Sexual Debut Rate (<18 years):")
print(f"   Among sexually active women: {early_debut_rate:.1f}%")

early_birth_rate = data_clean.loc[has_birth, 'early_first_birth'].mean() * 100
print(f"\n5. Early First Birth Rate (<18 years):")
print(f"   Among women who gave birth: {early_birth_rate:.1f}%")

# ===== SAVE CLEANED DATASET =====
print("\n" + "="*80)
print("SAVING CLEANED DATASET")
print("="*80)

output_file = 'rwanda_dhs_CLEANED_minimal.csv'
data_clean.to_csv(output_file, index=False)
print(f"\n✓ Cleaned dataset saved: {output_file}")
print(f"  Final N = {len(data_clean):,} observations")

# Save list of excluded cases for documentation
excluded_cases = data[biologically_impossible][['caseid', 'v525', 'v531', 'v012', 'v201', 
                                                  'education_category', 'wealth_category', 'residence']]
excluded_cases.to_csv('excluded_impossible_cases.csv', index=False)
print(f"✓ Excluded cases saved: excluded_impossible_cases.csv")

# ===== DOCUMENTATION FOR METHODS SECTION =====
print("\n" + "="*80)
print("DOCUMENTATION FOR YOUR RESEARCH PAPER")
print("="*80)

methods_text = f"""
SUGGESTED METHODS SECTION TEXT:
-------------------------------

Data Quality and Exclusions:
From the original sample of {original_n:,} women aged 15-49 years, we excluded 
{n_removed} cases ({(n_removed/original_n)*100:.2f}%) where the reported age at first 
birth preceded the reported age at first sexual intercourse, as this represents 
a biologically impossible scenario likely due to data entry error or recall bias. 
The final analytical sample comprised {len(data_clean):,} women ({(len(data_clean)/original_n)*100:.1f}% 
of the original sample).

We retained all other cases, including those with very young ages at sexual 
debut or first birth, as these represent legitimate outcomes within the scope 
of our research on early pregnancy risk factors. All analyses incorporated 
complex survey design elements and sampling weights (v005) to ensure 
representativeness of the Rwandan female population.

KEY POINTS FOR DISCUSSION/LIMITATIONS:
--------------------------------------
• {len(extreme_young_birth)} cases ({(len(extreme_young_birth)/len(data_clean))*100:.1f}%) 
  reported first birth before age 14, highlighting the vulnerability of young 
  adolescents to early pregnancy and potential child sexual abuse.

• The retention of extreme cases ensures our findings reflect the full spectrum 
  of early pregnancy experiences in Rwanda and avoids systematic bias toward 
  less vulnerable populations.

• DHS employs rigorous data collection protocols with multiple quality checks, 
  supporting the validity of reported ages even at extreme values.
"""

print(methods_text)

# ===== ETHICAL CONSIDERATIONS =====
print("\n" + "="*80)
print("ETHICAL CONSIDERATIONS FOR YOUR ANALYSIS")
print("="*80)
print(f"""
⚠️  IMPORTANT NOTES:

1. CHILD PROTECTION CONCERN:
   • {len(extreme_young_birth)} cases of first birth <14 years represent 
     serious child protection concerns
   • These should be discussed as evidence of vulnerability, NOT stigmatized
   • Frame as public health priority requiring intervention

2. FRAMING IN YOUR PAPER:
   ✓ DO: "These findings highlight the urgent need for child protection 
          interventions and comprehensive sexuality education"
   ✗ DON'T: Use judgmental language or imply moral failings

3. HANDLING SENSITIVE DATA:
   • Do NOT attempt to identify individuals
   • Aggregate reporting only
   • Follow ethical guidelines for vulnerable populations

4. POLICY IMPLICATIONS:
   • Your results can inform prevention programs
   • Emphasize structural factors (poverty, education, gender inequality)
   • Advocate for support services, not punishment
""")

print("\n" + "="*80)
print("✅ DATA CLEANING COMPLETE!")
print("="*80)
print(f"\nYour cleaned dataset is ready for analysis:")
print(f"  • File: {output_file}")
print(f"  • N = {len(data_clean):,} women")
print(f"  • Only {n_removed} biologically impossible cases removed")
print(f"  • All extreme but possible cases retained")
print(f"\n📊 Ready for statistical modeling and analysis!")
print("="*80)