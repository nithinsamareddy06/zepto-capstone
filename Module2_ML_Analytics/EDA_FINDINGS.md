# EDA Findings — Titanic Dataset

## 1. Profiling

Shape: **891 rows x 15 columns**

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

### `df.describe()`

|        |   survived |     pclass | sex   |      age |      sibsp |      parch |     fare | embarked   | class   | who   |   adult_male | deck   | embark_town   | alive   |   alone |
|:-------|-----------:|-----------:|:------|---------:|-----------:|-----------:|---------:|:-----------|:--------|:------|-------------:|:-------|:--------------|:--------|--------:|
| count  | 891        | 891        | 891   | 714      | 891        | 891        | 891      | 889        | 891     | 891   |          891 | 203    | 889           | 891     |     891 |
| unique | nan        | nan        | 2     | nan      | nan        | nan        | nan      | 3          | 3       | 3     |            2 | 7      | 3             | 2       |       2 |
| top    | nan        | nan        | male  | nan      | nan        | nan        | nan      | S          | Third   | man   |            1 | C      | Southampton   | no      |       1 |
| freq   | nan        | nan        | 577   | nan      | nan        | nan        | nan      | 644        | 491     | 537   |          537 | 59     | 644           | 549     |     537 |
| mean   |   0.383838 |   2.30864  | nan   |  29.6991 |   0.523008 |   0.381594 |  32.2042 | nan        | nan     | nan   |          nan | nan    | nan           | nan     |     nan |
| std    |   0.486592 |   0.836071 | nan   |  14.5265 |   1.10274  |   0.806057 |  49.6934 | nan        | nan     | nan   |          nan | nan    | nan           | nan     |     nan |
| min    |   0        |   1        | nan   |   0.42   |   0        |   0        |   0      | nan        | nan     | nan   |          nan | nan    | nan           | nan     |     nan |
| 25%    |   0        |   2        | nan   |  20.125  |   0        |   0        |   7.9104 | nan        | nan     | nan   |          nan | nan    | nan           | nan     |     nan |
| 50%    |   0        |   3        | nan   |  28      |   0        |   0        |  14.4542 | nan        | nan     | nan   |          nan | nan    | nan           | nan     |     nan |
| 75%    |   1        |   3        | nan   |  38      |   1        |   0        |  31      | nan        | nan     | nan   |          nan | nan    | nan           | nan     |     nan |
| max    |   1        |   3        | nan   |  80      |   8        |   6        | 512.329  | nan        | nan     | nan   |          nan | nan    | nan           | nan     |     nan |

### Missing values (%) — columns with any missing data

|             |     0 |
|:------------|------:|
| deck        | 77.22 |
| age         | 19.87 |
| embarked    |  0.22 |
| embark_town |  0.22 |

## 2. Missing-value handling (threshold rule: <5% -> drop rows, 5-30% -> impute, very-high -> drop column / encode as its own category)

- `deck`: 77.22% missing -> **>30%**; **dropped the column**. Imputing a deck letter for that many rows would be mostly fabricated information, and its signal (cabin location) is already largely captured by `pclass` and `fare`.

- `age`: 19.87% missing -> **5-30%**; imputed with median = 28.00.

- `embarked`: 0.22% missing -> **<5%**; dropped 2 row(s) with missing values.

- `embark_town`: 0.22% missing -> **<5%**; dropped 0 row(s) with missing values.


Shape after cleaning: (889, 14)

## 3. Univariate analysis — age & fare

- IQR outliers in `age`: **65** (bounds: [2.5, 54.5])

- IQR outliers in `fare`: **114** (bounds: [-26.8, 65.7])

- `fare`: mean=32.10, median=14.45, mode=8.05 -> mean > median > mode indicates the distribution is **right-skewed** (a long right tail of high-fare first-class passengers pulls the mean above the median, which in turn sits above the most common low fare).

## 4. Bivariate analysis

### Survival rate by sex
| sex    |   survived |
|:-------|-----------:|
| female |      0.74  |
| male   |      0.189 |

### Survival rate by pclass
|   pclass |   survived |
|---------:|-----------:|
|        1 |      0.626 |
|        2 |      0.473 |
|        3 |      0.242 |

### Survival rate by sex & pclass (boolean masking)
| sex    |   pclass |   survival_rate |
|:-------|---------:|----------------:|
| male   |        1 |           0.369 |
| male   |        2 |           0.157 |
| male   |        3 |           0.135 |
| female |        1 |           0.967 |
| female |        2 |           0.921 |
| female |        3 |           0.5   |

### Correlation matrix
|          |   survived |   pclass |   age |   sibsp |   parch |   fare |
|:---------|-----------:|---------:|------:|--------:|--------:|-------:|
| survived |       1    |    -0.34 | -0.07 |   -0.03 |    0.08 |   0.26 |
| pclass   |      -0.34 |     1    | -0.34 |    0.08 |    0.02 |  -0.55 |
| age      |      -0.07 |    -0.34 |  1    |   -0.23 |   -0.17 |   0.09 |
| sibsp    |      -0.03 |     0.08 | -0.23 |    1    |    0.41 |   0.16 |
| parch    |       0.08 |     0.02 | -0.17 |    0.41 |    1    |   0.22 |
| fare     |       0.26 |    -0.55 |  0.09 |    0.16 |    0.22 |   1    |

### Two strongest correlations (by absolute value)
| level_0   | level_1   |      corr |   abs_corr |
|:----------|:----------|----------:|-----------:|
| pclass    | fare      | -0.548193 |   0.548193 |
| sibsp     | parch     |  0.414542 |   0.414542 |

- The strongest pair is **pclass vs fare** at r=-0.55, and the second-strongest is **sibsp vs parch** at r=0.41. `pclass` and `fare` are strongly negatively correlated because class number counts *down* from luxury (1st) to economy (3rd) while ticket price counts up, so a lower class number pairs with a higher fare. `sibsp` and `parch` are moderately positively correlated because both simply count family members travelling together (siblings/spouses vs. parents/children) — passengers who brought one kind of relative were more likely to have brought the other too, i.e. they travelled as larger family units rather than alone. Note that `survived` itself is only moderately correlated with `pclass` (r=-0.34) and `fare` (r=0.26) individually — real, but weaker than either of the two pairs above.

## 5. Multivariate data story (>=4 charts)

**Chart 1 — Survival by class & sex:** Women survived at a far higher rate than men in every class, and within each sex, 1st class passengers fared better than 3rd class. This suggests both the 'women and children first' evacuation norm and cabin-class proximity to lifeboats both mattered independently.

**Chart 2 — Age vs survival:** Median age is similar between survivors and non-survivors, but survivors show a visibly denser cluster among young children, consistent with children being prioritised during evacuation even though the overall age effect is modest.

**Chart 3 — Fare vs age scatter:** Survivors (orange) skew toward higher fares at almost every age, reinforcing that ticket price — a proxy for class and deck location — was a stronger survival driver than age alone.

**Chart 4 — Family size vs survival:** Passengers travelling with a small family (1-3 companions) survived at a higher rate than those travelling completely alone or in very large families, suggesting a 'sweet spot' where companions could help each other reach a lifeboat without the group being too large to move together.

## 6. EDA-stage standardization check (age & fare)

This is a pure sanity check — it does **not** feed the modeling pipeline, which does its own train-only scaling in Part B.

**Before standardization:**
|      |    age |   fare |
|:-----|-------:|-------:|
| mean | 29.315 | 32.097 |
| std  | 12.985 | 49.698 |

**After standardization (z-scores):**
|      |   age_z |   fare_z |
|:-----|--------:|---------:|
| mean |       0 |        0 |
| std  |       1 |        1 |

Both z-scored columns land at ~mean 0 / std 1, confirming the manual z-score transform worked as expected.
