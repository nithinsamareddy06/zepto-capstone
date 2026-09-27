# Analytics Module - EDA Report

## 1. Load & Profile

Saved raw load to `titanic.csv` (committed offline fallback; 02_modeling.py reads this same file and never reloads from seaborn).

- Shape: 891 rows x 15 columns

### df.info()
```
<class 'pandas.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 15 columns):
 #   Column       Non-Null Count  Dtype   
---  ------       --------------  -----   
 0   survived     891 non-null    int64   
 1   pclass       891 non-null    int64   
 2   sex          891 non-null    str     
 3   age          714 non-null    float64 
 4   sibsp        891 non-null    int64   
 5   parch        891 non-null    int64   
 6   fare         891 non-null    float64 
 7   embarked     889 non-null    str     
 8   class        891 non-null    category
 9   who          891 non-null    str     
 10  adult_male   891 non-null    bool    
 11  deck         203 non-null    category
 12  embark_town  889 non-null    str     
 13  alive        891 non-null    str     
 14  alone        891 non-null    bool    
dtypes: bool(2), category(2), float64(2), int64(4), str(5)
memory usage: 80.7 KB
```

### df.describe()
|       |   survived |     pclass |      age |      sibsp |      parch |     fare |
|:------|-----------:|-----------:|---------:|-----------:|-----------:|---------:|
| count | 891        | 891        | 714      | 891        | 891        | 891      |
| mean  |   0.383838 |   2.30864  |  29.6991 |   0.523008 |   0.381594 |  32.2042 |
| std   |   0.486592 |   0.836071 |  14.5265 |   1.10274  |   0.806057 |  49.6934 |
| min   |   0        |   1        |   0.42   |   0        |   0        |   0      |
| 25%   |   0        |   2        |  20.125  |   0        |   0        |   7.9104 |
| 50%   |   0        |   3        |  28      |   0        |   0        |  14.4542 |
| 75%   |   1        |   3        |  38      |   1        |   0        |  31      |
| max   |   1        |   3        |  80      |   8        |   6        | 512.329  |

### Missing values (%) — columns with any missing data
|             |     0 |
|:------------|------:|
| deck        | 77.22 |
| age         | 19.87 |
| embarked    |  0.22 |
| embark_town |  0.22 |

## 2. Missing-Value Handling (threshold rule)

- **deck** (77.22% missing, very high): imputation would be unreliable, so 'Missing' is encoded as its own explicit category rather than dropped or imputed.
- **age** (19.87% missing, 5-30%): imputed with median = 28.00.
- **embarked** (0.22% missing, <5%): dropped 2 rows.
- **embark_town** (0.22% missing, <5%): dropped 0 rows.

Shape after missing-value handling: (889, 15)

## 3. Univariate Analysis — age & fare

Saved `charts/univariate_age_fare.png`.
- IQR outliers in **age**: 65 (bounds: [2.50, 54.50])
- IQR outliers in **fare**: 114 (bounds: [-26.76, 65.66])

Fare — mean=32.10, median=14.45, mode=8.05
Since mean > median > mode, the fare distribution is **right-skewed**: a small number of high-fare (first-class) passengers pull the mean well above the typical (median) fare.

## 4. Bivariate Analysis

### Survival rate by sex
- male: 0.189
- female: 0.740

### Survival rate by pclass
- class 1: 0.626
- class 2: 0.473
- class 3: 0.242

### Survival rate by sex AND pclass
- sex=male, pclass=1: 0.369
- sex=male, pclass=2: 0.157
- sex=male, pclass=3: 0.135
- sex=female, pclass=1: 0.967
- sex=female, pclass=2: 0.921
- sex=female, pclass=3: 0.500

Saved `charts/correlation_heatmap.png`.

### Two strongest correlations
- **pclass vs fare**: r = -0.548
- **sibsp vs parch**: r = 0.415

Interpretation:
- **pclass vs fare** (r=-0.55): this is expected: pclass is coded 1 (best) to 3 (worst), so a negative correlation with fare simply means higher-class tickets cost more.
- **sibsp vs parch** (r=0.41): this reflects that passengers traveling with siblings/spouses also tend to be traveling with parents/children, i.e. it picks out passengers traveling as part of a larger family unit.

## 5. Multivariate Data Story (4+ charts)

**Chart 1 — Survival rate by class & sex (bar):** Women survived at far higher rates than men in every class, and within each sex, survival drops sharply from 1st to 3rd class. This suggests both the 'women and children first' evacuation norm and unequal access to lifeboats by cabin class were both at play, rather than either factor alone explaining the outcome.

**Chart 2 — Age by survival (box):** Median age is broadly similar between survivors and non-survivors, but survivors show a visibly wider lower tail, consistent with young children being prioritized for rescue even though age alone is a weak predictor overall.

**Chart 3 — Fare vs age, colored by survival (scatter):** Survivors are visibly concentrated at higher fares, while the cluster of low-fare passengers across all ages skews toward non-survival. Fare, as a proxy for cabin class and deck location, looks like a stronger survival signal than age by itself.

**Chart 4 — Survival rate by family size (bar):** Passengers traveling with a small family (1-3 relatives aboard) survived at higher rates than those traveling completely alone or in very large families, suggesting mutual help in small groups, while large families struggled to stay together and evacuate as a unit.

## 6. Exploratory Standardization Check (age & fare)

- **age** — before: mean=29.315, std=12.985 | after z-score: mean=0.000, std=1.000
- **fare** — before: mean=32.097, std=49.698 | after z-score: mean=0.000, std=1.000

Note: this standardization is an EDA-stage sanity check only. It does not feed into the modeling pipeline in `02_modeling.py`, which performs its own train-only scaling to avoid leakage.