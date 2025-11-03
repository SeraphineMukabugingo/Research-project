import pandas as pd
import numpy as np
from datetime import datetime

# ===== CREATE COMPREHENSIVE DATA DICTIONARY =====
print("="*80)
print("RWANDA DHS 2019-20: EARLY SEXUAL DEBUT & EARLY PREGNANCY DATASET")
print("DATA DICTIONARY")
print("="*80)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Dataset: RWIR81FL (Individual Recode)")
print("="*80 + "\n")

# Define the data dictionary structure
data_dict = {
    'variable_name': [],
    'variable_label': [],
    'variable_type': [],
    'value_labels': [],
    'notes': [],
    'source': [],
    'section': []
}

def add_variable(name, label, var_type, values, notes, source, section):
    """Helper function to add variables to the dictionary"""
    data_dict['variable_name'].append(name)
    data_dict['variable_label'].append(label)
    data_dict['variable_type'].append(var_type)
    data_dict['value_labels'].append(values)
    data_dict['notes'].append(notes)
    data_dict['source'].append(source)
    data_dict['section'].append(section)

# ==================== SECTION 1: IDENTIFICATION & WEIGHTS ====================
section = "1. Identification & Weights"

add_variable(
    'caseid', 
    'Unique case identifier', 
    'String',
    'Unique ID for each respondent',
    'Primary key for merging datasets',
    'DHS Original',
    section
)

add_variable(
    'v001', 
    'Cluster number', 
    'Numeric',
    '1-xxx (cluster ID)',
    'Geographic cluster identifier for spatial analysis',
    'DHS Original',
    section
)

add_variable(
    'v005', 
    'Sample weight', 
    'Numeric',
    'Integer values (typically 6-7 digits)',
    'CRITICAL: Must be normalized (divide by 1,000,000) for proper weighted analysis',
    'DHS Original',
    section
)

add_variable(
    'sample_weight', 
    'Normalized sample weight', 
    'Numeric (Float)',
    'Continuous values around 1.0',
    'v005 divided by 1,000,000. Use this for all weighted analyses',
    'Derived',
    section
)

add_variable(
    'v024', 
    'Region/Province', 
    'Numeric',
    '1=Kigali City, 2=South, 3=West, 4=North, 5=East',
    'Administrative region of residence',
    'DHS Original',
    section
)

add_variable(
    'v025', 
    'Type of place of residence', 
    'Numeric',
    '1=Urban, 2=Rural',
    'Urban-rural classification',
    'DHS Original',
    section
)

add_variable(
    'residence', 
    'Type of residence (labeled)', 
    'Categorical',
    'Urban, Rural',
    'Labeled version of v025',
    'Derived',
    section
)

add_variable(
    'v102', 
    'Type of place of residence (detailed)', 
    'Numeric',
    '1=Capital/large city, 2=Small city, 3=Town, 4=Countryside',
    'More detailed urban-rural classification',
    'DHS Original',
    section
)

# ==================== SECTION 2: PRIMARY OUTCOMES ====================
section = "2. Primary Outcomes"

add_variable(
    'v525', 
    'Age at first sexual intercourse', 
    'Numeric',
    '1-49=Age in years, 0=Never had sex, 96=At first union, 97=Inconsistent, 98=Don\'t know, 99=Missing',
    'PRIMARY OUTCOME: Age when first had sexual intercourse. Valid ages: 1-49',
    'DHS Original',
    section
)

add_variable(
    'sexual_debut_category', 
    'Sexual debut status (3 categories)', 
    'Categorical',
    'Never had sex, Early debut (<18), Normal/Late debut (≥18)',
    'DERIVED PRIMARY OUTCOME: Categorizes sexual debut timing',
    'Derived',
    section
)

add_variable(
    'early_sexual_debut', 
    'Early sexual debut indicator', 
    'Binary',
    '0=Debut at ≥18 years, 1=Debut at <18 years, NaN=Never had sex',
    'Binary indicator for early sexual debut (only for sexually active women)',
    'Derived',
    section
)

add_variable(
    'very_early_debut', 
    'Very early sexual debut indicator', 
    'Binary',
    '0=Debut at ≥15 years, 1=Debut at <15 years, NaN=Never had sex',
    'Binary indicator for very early sexual debut (only for sexually active women)',
    'Derived',
    section
)

add_variable(
    'v531', 
    'Age at first birth', 
    'Numeric',
    '1-49=Age in years, 0=Never gave birth, 97=Inconsistent, 98=Don\'t know, 99=Missing',
    'PRIMARY OUTCOME: Age when first child was born. Valid ages: 1-49',
    'DHS Original',
    section
)

add_variable(
    'early_first_birth', 
    'Early first birth indicator', 
    'Binary',
    '0=First birth at ≥18 years, 1=First birth at <18 years, NaN=Never gave birth',
    'Binary indicator for early childbearing (only for mothers)',
    'Derived',
    section
)

add_variable(
    'teen_pregnancy', 
    'Teen pregnancy indicator', 
    'Binary',
    '0=First birth at ≥20 years, 1=First birth at <20 years, NaN=Never gave birth',
    'Binary indicator for teenage pregnancy (only for mothers)',
    'Derived',
    section
)

add_variable(
    'v012', 
    'Current age of respondent', 
    'Numeric',
    '15-49 (years)',
    'Age at time of survey',
    'DHS Original',
    section
)

add_variable(
    'v013', 
    'Age in 5-year groups', 
    'Numeric',
    '1=15-19, 2=20-24, 3=25-29, 4=30-34, 5=35-39, 6=40-44, 7=45-49',
    'DHS standard age grouping',
    'DHS Original',
    section
)

add_variable(
    'age_group', 
    'Age group (labeled)', 
    'Categorical',
    '15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49',
    'Labeled version of age groups',
    'Derived',
    section
)

add_variable(
    'v201', 
    'Total children ever born', 
    'Numeric',
    '0-20+ (number of children)',
    'Total number of children born alive to respondent',
    'DHS Original',
    section
)

add_variable(
    'ever_given_birth', 
    'Ever given birth indicator', 
    'Binary',
    '0=No children, 1=Has given birth',
    'Binary indicator for childbearing experience',
    'Derived',
    section
)

add_variable(
    'v218', 
    'Number of living children', 
    'Numeric',
    '0-20+ (number of children)',
    'Number of children currently alive',
    'DHS Original',
    section
)

add_variable(
    'v213', 
    'Currently pregnant', 
    'Numeric',
    '0=No/unsure, 1=Yes',
    'Pregnancy status at time of survey',
    'DHS Original',
    section
)

# ==================== SECTION 3: EDUCATION ====================
section = "3. Education"

add_variable(
    'v106', 
    'Highest educational level', 
    'Numeric',
    '0=No education, 1=Primary, 2=Secondary, 3=Higher',
    'Highest level of education attended',
    'DHS Original',
    section
)

add_variable(
    'education_category', 
    'Education level (labeled)', 
    'Categorical',
    'No education, Primary, Secondary, Higher',
    'Labeled version of v106',
    'Derived',
    section
)

add_variable(
    'v107', 
    'Highest year of education', 
    'Numeric',
    '0-20+ (years completed)',
    'Highest year completed at that level',
    'DHS Original',
    section
)

add_variable(
    'v133', 
    'Education in single years', 
    'Numeric',
    '0-20+ (total years)',
    'Total years of education completed',
    'DHS Original',
    section
)

add_variable(
    'v149', 
    'Educational attainment', 
    'Numeric',
    '0=No education, 1=Incomplete primary, 2=Complete primary, 3=Incomplete secondary, 4=Complete secondary, 5=Higher',
    'More detailed education categorization',
    'DHS Original',
    section
)

# ==================== SECTION 4: SOCIOECONOMIC STATUS ====================
section = "4. Socioeconomic Status"

add_variable(
    'v190', 
    'Wealth index quintile', 
    'Numeric',
    '1=Poorest, 2=Poorer, 3=Middle, 4=Richer, 5=Richest',
    'Household wealth index in quintiles (composite measure)',
    'DHS Original',
    section
)

add_variable(
    'wealth_category', 
    'Wealth quintile (labeled)', 
    'Categorical',
    'Poorest, Poorer, Middle, Richer, Richest',
    'Labeled version of v190',
    'Derived',
    section
)

add_variable(
    'v191', 
    'Wealth index factor score', 
    'Numeric (Float)',
    'Continuous score (typically -100000 to 300000)',
    'Continuous wealth score before categorization',
    'DHS Original',
    section
)

add_variable(
    'v190a', 
    'Wealth index for urban/rural', 
    'Numeric',
    '1=Poorest, 2=Poorer, 3=Middle, 4=Richer, 5=Richest',
    'Wealth index calculated separately for urban and rural areas',
    'DHS Original',
    section
)

# ==================== SECTION 5: MARRIAGE & UNION STATUS ====================
section = "5. Marriage & Union Status"

add_variable(
    'v501', 
    'Current marital status', 
    'Numeric',
    '0=Never married, 1=Married, 2=Living together, 3=Widowed, 4=Divorced, 5=Separated',
    'Current marital/union status',
    'DHS Original',
    section
)

add_variable(
    'marital_status', 
    'Marital status (labeled)', 
    'Categorical',
    'Never married, Married, Living together, Widowed, Divorced, Separated',
    'Labeled version of v501',
    'Derived',
    section
)

add_variable(
    'v502', 
    'Currently/formerly/never in union', 
    'Numeric',
    '0=Never in union, 1=Currently in union, 2=Formerly in union',
    'Simplified union status',
    'DHS Original',
    section
)

add_variable(
    'v511', 
    'Age at first cohabitation/union', 
    'Numeric',
    '1-49=Age in years, 0=Never in union, 97=Inconsistent, 98=Don\'t know',
    'Age when started living with first partner',
    'DHS Original',
    section
)

add_variable(
    'v512', 
    'Years since first cohabitation', 
    'Numeric',
    '0-35+ (years)',
    'Duration since first union',
    'DHS Original',
    section
)

# ==================== SECTION 6: CULTURAL & SOCIAL FACTORS ====================
section = "6. Cultural & Social Factors"

add_variable(
    'v130', 
    'Religion', 
    'Numeric',
    '1=Catholic, 2=Protestant, 3=Adventist, 4=Muslim, 5=Traditional, 6=No religion, 96=Other',
    'Religious affiliation',
    'DHS Original',
    section
)

add_variable(
    'religion', 
    'Religion (labeled)', 
    'Categorical',
    'Catholic, Protestant, Adventist, Muslim, Traditional, No religion, Other',
    'Labeled version of v130',
    'Derived',
    section
)

add_variable(
    'v131', 
    'Ethnicity', 
    'Numeric',
    'Country-specific ethnic group codes',
    'Ethnic group affiliation',
    'DHS Original',
    section
)

# ==================== SECTION 7: CONTRACEPTIVE KNOWLEDGE & USE ====================
section = "7. Contraceptive Knowledge & Use"

add_variable(
    'v301', 
    'Knowledge of any contraceptive method', 
    'Numeric',
    '0=Knows no method, 1=Knows only traditional, 2=Knows modern method',
    'Level of contraceptive knowledge',
    'DHS Original',
    section
)

add_variable(
    'v302', 
    'Ever use of any contraceptive method', 
    'Numeric',
    '0=Never used, 1=Used in past, 2=Current user',
    'Ever use of any contraceptive method',
    'DHS Original',
    section
)

add_variable(
    'v302a', 
    'Ever use of modern contraceptive method', 
    'Numeric',
    '0=Never used, 1=Used in past, 2=Current user',
    'Ever use of modern methods only',
    'DHS Original',
    section
)

add_variable(
    'v312', 
    'Current contraceptive method', 
    'Numeric',
    '0=Not using, 1-10+=Specific method codes',
    'Current contraceptive method being used',
    'DHS Original',
    section
)

add_variable(
    'v313', 
    'Current use by method type', 
    'Numeric',
    '0=Not using, 1=Modern method, 2=Traditional method',
    'Current contraceptive use categorized by type',
    'DHS Original',
    section
)

add_variable(
    'v375a', 
    'Desire for more children', 
    'Numeric',
    '1=Wants another, 2=Undecided, 3=Wants no more, 4=Sterilized, 5=Declared infecund',
    'Future fertility preferences',
    'DHS Original',
    section
)

# ==================== SECTION 8: EMPLOYMENT & AUTONOMY ====================
section = "8. Employment & Women's Autonomy"

add_variable(
    'v714', 
    'Currently working', 
    'Numeric',
    '0=No, 1=Yes',
    'Employment status at time of survey',
    'DHS Original',
    section
)

add_variable(
    'v715', 
    'Occupation', 
    'Numeric',
    '0=Not working, 1-98=Occupation codes',
    'Type of occupation (if working)',
    'DHS Original',
    section
)

add_variable(
    'v731', 
    'Worked in last 12 months', 
    'Numeric',
    '0=Not worked, 1=In past year, 2=Currently working, 3=Have a job but on leave',
    'Work status in past year',
    'DHS Original',
    section
)

add_variable(
    'v743a', 
    'Who decides on respondent\'s healthcare', 
    'Numeric',
    '1=Respondent alone, 2=Respondent and partner, 3=Partner alone, 4=Someone else, 5=Other, 6=Not applicable',
    'Decision-making autonomy for healthcare',
    'DHS Original',
    section
)

add_variable(
    'v743b', 
    'Who decides on large household purchases', 
    'Numeric',
    '1=Respondent alone, 2=Respondent and partner, 3=Partner alone, 4=Someone else, 5=Other, 6=Not applicable',
    'Decision-making autonomy for major purchases',
    'DHS Original',
    section
)

add_variable(
    'v743d', 
    'Who decides on visits to family/relatives', 
    'Numeric',
    '1=Respondent alone, 2=Respondent and partner, 3=Partner alone, 4=Someone else, 5=Other, 6=Not applicable',
    'Decision-making autonomy for family visits',
    'DHS Original',
    section
)

add_variable(
    'v744a', 
    'Wife beating justified if burns food', 
    'Numeric',
    '0=No, 1=Yes',
    'Attitude toward domestic violence',
    'DHS Original',
    section
)

add_variable(
    'v744b', 
    'Wife beating justified if argues with husband', 
    'Numeric',
    '0=No, 1=Yes',
    'Attitude toward domestic violence',
    'DHS Original',
    section
)

add_variable(
    'v744e', 
    'Wife beating justified if refuses sex', 
    'Numeric',
    '0=No, 1=Yes',
    'Attitude toward domestic violence',
    'DHS Original',
    section
)

# ==================== SECTION 9: PARTNER CHARACTERISTICS ====================
section = "9. Partner Characteristics"

add_variable(
    'v701', 
    'Partner\'s age', 
    'Numeric',
    '15-95+ (years)',
    'Current/most recent partner\'s age',
    'DHS Original',
    section
)

add_variable(
    'v704', 
    'Partner lives in household', 
    'Numeric',
    '0=No, 1=Yes',
    'Whether partner currently lives with respondent',
    'DHS Original',
    section
)

add_variable(
    'v730', 
    'Age difference with partner', 
    'Numeric',
    '-25 to +50 (years; negative=partner younger)',
    'Partner\'s age minus respondent\'s age',
    'DHS Original',
    section
)

add_variable(
    'v751', 
    'Partner\'s education level', 
    'Numeric',
    '0=No education, 1=Primary, 2=Secondary, 3=Higher, 8=Don\'t know',
    'Partner\'s highest educational level',
    'DHS Original',
    section
)

# ==================== SECTION 10: MEDIA EXPOSURE ====================
section = "10. Media Exposure"

add_variable(
    'v157', 
    'Frequency of reading newspaper/magazine', 
    'Numeric',
    '0=Not at all, 1=Less than once a week, 2=At least once a week, 3=Almost every day',
    'Media exposure - print',
    'DHS Original',
    section
)

add_variable(
    'v158', 
    'Frequency of listening to radio', 
    'Numeric',
    '0=Not at all, 1=Less than once a week, 2=At least once a week, 3=Almost every day',
    'Media exposure - radio',
    'DHS Original',
    section
)

add_variable(
    'v159', 
    'Frequency of watching television', 
    'Numeric',
    '0=Not at all, 1=Less than once a week, 2=At least once a week, 3=Almost every day',
    'Media exposure - television',
    'DHS Original',
    section
)

# ==================== SECTION 11: HIV/AIDS KNOWLEDGE ====================
section = "11. HIV/AIDS Knowledge"

add_variable(
    'v754bp', 
    'Know condoms prevent AIDS', 
    'Numeric',
    '0=No, 1=Yes',
    'Knowledge of condom use for HIV prevention',
    'DHS Original',
    section
)

add_variable(
    'v754cp', 
    'Know limiting partners prevents AIDS', 
    'Numeric',
    '0=No, 1=Yes',
    'Knowledge of partner reduction for HIV prevention',
    'DHS Original',
    section
)

add_variable(
    'v754dp', 
    'Know abstaining prevents AIDS', 
    'Numeric',
    '0=No, 1=Yes',
    'Knowledge of abstinence for HIV prevention',
    'DHS Original',
    section
)

# ==================== SECTION 12: SEXUAL BEHAVIOR ====================
section = "12. Sexual Behavior"

add_variable(
    'v527', 
    'Time since last sexual intercourse', 
    'Numeric',
    '100-599=Days, 600-1295=Months (subtract 600), 1300-1395=Years (subtract 1300), 0=Never had sex, 995=Inconsistent',
    'Duration since last sex (complex coding)',
    'DHS Original',
    section
)

add_variable(
    'v528', 
    'Time since last intercourse (grouped)', 
    'Numeric',
    '0=Never, 1=Active in last 4 weeks, 2-8=Time categories',
    'Grouped version of v527',
    'DHS Original',
    section
)

add_variable(
    'v535', 
    'Recent sexual activity', 
    'Numeric',
    '0=Not active, 1=Active in last 4 weeks',
    'Recent sexual activity indicator',
    'DHS Original',
    section
)

add_variable(
    'v536', 
    'Recent sexual activity (detailed)', 
    'Numeric',
    'Various codes for timing and circumstances',
    'Detailed recent sexual activity',
    'DHS Original',
    section
)

# ==================== SECTION 13: HOUSEHOLD CHARACTERISTICS ====================
section = "13. Household Characteristics"

add_variable(
    'v113', 
    'Has electricity', 
    'Numeric',
    '0=No, 1=Yes',
    'Household has electricity',
    'DHS Original',
    section
)

add_variable(
    'v115', 
    'Has television', 
    'Numeric',
    '0=No, 1=Yes',
    'Household has television',
    'DHS Original',
    section
)

add_variable(
    'v116', 
    'Has radio', 
    'Numeric',
    '0=No, 1=Yes',
    'Household has radio',
    'DHS Original',
    section
)

add_variable(
    'v119', 
    'Has telephone/mobile phone', 
    'Numeric',
    '0=No, 1=Yes',
    'Household has telephone or mobile phone',
    'DHS Original',
    section
)

add_variable(
    'v120', 
    'Has bicycle', 
    'Numeric',
    '0=No, 1=Yes',
    'Household has bicycle',
    'DHS Original',
    section
)

add_variable(
    'v121', 
    'Has motorcycle/scooter', 
    'Numeric',
    '0=No, 1=Yes',
    'Household has motorcycle or scooter',
    'DHS Original',
    section
)

add_variable(
    'v122', 
    'Has car/truck', 
    'Numeric',
    '0=No, 1=Yes',
    'Household has car or truck',
    'DHS Original',
    section
)

add_variable(
    'v127', 
    'Main floor material', 
    'Numeric',
    '10-39=Various floor types (earth to polished)',
    'Type of flooring in dwelling',
    'DHS Original',
    section
)

add_variable(
    'v128', 
    'Main wall material', 
    'Numeric',
    '10-39=Various wall types (cane to cement)',
    'Type of walls in dwelling',
    'DHS Original',
    section
)

add_variable(
    'v129', 
    'Main roof material', 
    'Numeric',
    '10-39=Various roof types (thatch to tiles)',
    'Type of roofing in dwelling',
    'DHS Original',
    section
)

# ==================== SECTION 14: FIRST BIRTH DETAILS ====================
section = "14. Birth History Details"

add_variable(
    'bidx_01', 
    'Birth index of first child', 
    'Numeric',
    '1-20 (birth order)',
    'Index number of first-born child in birth history',
    'DHS Original',
    section
)

add_variable(
    'b2_01', 
    'Year of first birth', 
    'Numeric',
    '1970-2020 (year)',
    'Calendar year of first birth',
    'DHS Original',
    section
)

add_variable(
    'b3_01', 
    'Date of first birth (CMC)', 
    'Numeric',
    'Century Month Code',
    'Date of first birth in CMC format (months since Jan 1900)',
    'DHS Original',
    section
)

add_variable(
    'b11_01', 
    'Preceding birth interval', 
    'Numeric',
    '0-600+ (months)',
    'Months between first and second birth',
    'DHS Original',
    section
)

add_variable(
    'v209', 
    'Interval between marriage and first birth', 
    'Numeric',
    '0-600+ (months)',
    'Duration from first union to first birth',
    'DHS Original',
    section
)

# ==================== SECTION 15: PREGNANCY & BIRTH HISTORY ====================
section = "15. Pregnancy & Birth History"

add_variable(
    'v202', 
    'Number of pregnancies', 
    'Numeric',
    '0-20+ (pregnancies)',
    'Total number of pregnancies (including current)',
    'DHS Original',
    section
)

add_variable(
    'v203', 
    'Currently pregnant', 
    'Numeric',
    '0=No/unsure, 1=Yes',
    'Current pregnancy status',
    'DHS Original',
    section
)

add_variable(
    'v204', 
    'Pregnancy wanted', 
    'Numeric',
    '1=Then, 2=Later, 3=Not at all',
    'Whether current/last pregnancy was wanted (timing)',
    'DHS Original',
    section
)

add_variable(
    'v205', 
    'Preferred waiting time', 
    'Numeric',
    '1-998 (months), 994=Wanted soon, 995=Wanted then',
    'How long wanted to wait for pregnancy',
    'DHS Original',
    section
)

add_variable(
    'v206', 
    'Sons who have died', 
    'Numeric',
    '0-20+ (number)',
    'Number of male children who died',
    'DHS Original',
    section
)

add_variable(
    'v207', 
    'Daughters who have died', 
    'Numeric',
    '0-20+ (number)',
    'Number of female children who died',
    'DHS Original',
    section
)

add_variable(
    'v208', 
    'Births in last 5 years', 
    'Numeric',
    '0-5+ (births)',
    'Number of births in 5 years before survey',
    'DHS Original',
    section
)

# ==================== SECTION 16: FAMILY PLANNING MESSAGES ====================
section = "16. Exposure to Family Planning Messages"

add_variable(
    'v384a', 
    'Heard family planning on radio', 
    'Numeric',
    '0=No, 1=Yes',
    'Heard FP message on radio in last few months',
    'DHS Original',
    section
)

add_variable(
    'v384b', 
    'Heard family planning on TV', 
    'Numeric',
    '0=No, 1=Yes',
    'Heard FP message on TV in last few months',
    'DHS Original',
    section
)

add_variable(
    'v384c', 
    'Heard family planning in newspaper/magazine', 
    'Numeric',
    '0=No, 1=Yes',
    'Saw FP message in print media in last few months',
    'DHS Original',
    section
)

# ==================== CREATE DATAFRAME ====================
dict_df = pd.DataFrame(data_dict)

# Sort by section and then by variable name
dict_df = dict_df.sort_values(['section', 'variable_name']).reset_index(drop=True)

# ==================== PRINT FORMATTED DATA DICTIONARY ====================
print("\n" + "="*80)
print("DETAILED DATA DICTIONARY")
print("="*80 + "\n")

current_section = None
for idx, row in dict_df.iterrows():
    # Print section header when it changes
    if row['section'] != current_section:
        current_section = row['section']
        print("\n" + "="*80)
        print(f"{current_section}")
        print("="*80 + "\n")
    
    print(f"Variable: {row['variable_name']}")
    print(f"Label:    {row['variable_label']}")
    print(f"Type:     {row['variable_type']}")
    print(f"Values:   {row['value_labels']}")
    print(f"Notes:    {row['notes']}")
    print(f"Source:   {row['source']}")
    print("-" * 80 + "\n")

# ==================== SAVE DATA DICTIONARY ====================
# Save as CSV
dict_df.to_csv('rwanda_dhs_data_dictionary.csv', index=False, encoding='utf-8')
print("\n✓ Data dictionary saved as: rwanda_dhs_data_dictionary.csv")

# Save as Excel with better formatting
try:
    with pd.ExcelWriter('rwanda_dhs_data_dictionary.xlsx', engine='openpyxl') as writer:
        dict_df.to_excel(writer, sheet_name='Data Dictionary', index=False)
        
        # Get workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Data Dictionary']
        
        # Set column widths
        worksheet.column_dimensions['A'].width = 25  # variable_name
        worksheet.column_dimensions['B'].width = 50  # variable_label
        worksheet.column_dimensions['C'].width = 15  # variable_type
        worksheet.column_dimensions['D'].width = 60  # value_labels
        worksheet.column_dimensions['E'].width = 70  # notes
        worksheet.column_dimensions['F'].width = 12  # source
        worksheet.column_dimensions['G'].width = 35  # section
        
    print("✓ Data dictionary saved as: rwanda_dhs_data_dictionary.xlsx")
except:
    print("  (Excel file could not be created - openpyxl may not be installed)")

# ==================== SUMMARY STATISTICS ====================
print("\n" + "="*80)
print("DATA DICTIONARY SUMMARY")
print("="*80)
print(f"\nTotal variables documented: {len(dict_df)}")
print(f"\nVariables by source:")
print(dict_df['source'].value_counts())
print(f"\nVariables by type:")
print(dict_df['variable_type'].value_counts())
print(f"\nVariables by section:")
print(dict_df.groupby('section').size())

print("\n" + "="*80)
print("QUICK REFERENCE GUIDE")
print("="*80)
print("\nKEY VARIABLES FOR ANALYSIS:")
print("\nOutcome Variables:")
print("  - v525 / sexual_debut_category: Sexual debut timing")
print("  - v531 / early_first_birth: Age at first birth")
print("  - early_sexual_debut: Binary indicator for early debut")
print("\nIndependent Variables (Main Predictors):")
print("  - v106 / education_category: Education level")
print("  - v190 / wealth_category: Wealth quintile")
print("  - v025 / residence: Urban/rural")
print("  - v501 / marital_status: Marital status")
print("  - v012 / age_group: Age of respondent")
print("\nImportant Note:")
print("  - ALWAYS use sample_weight in all analyses")
print("  - For v525: Only values 1-49 are valid ages; 0/NaN = never had sex")
print("  - For v531: Only values 1-49 are valid ages; 0/NaN = never gave birth")
print("\n✓ Data dictionary complete!")