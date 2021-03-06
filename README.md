House prices
================

# Introduction

This notebook is for the prediction of property sale price. My initial
thoughts based on a glimpse at the data is to predict sale price using
the general features of quality, size, and location. This notebook will
be split into four major sections.

1.  [processing missing analysis](#1)
2.  [exploratory analysis, e.g.¬†correlation plots](#2)
3.  [feature engineering](#3)
4.  [modeling](#4)

# Importing

Package libraries and data importing.

``` r
devtools::load_all()
x <- list(
"magrittr",
"tidyverse",
"purrr",
"rcompanion",
"naniar",
"caret", # for dummy and one hot encode
"glmnet",
"plotmo",
"corrr",
"corrplot")
lapply(x, require, character.only = TRUE)
```

    ## [[1]]
    ## [1] TRUE
    ## 
    ## [[2]]
    ## [1] TRUE
    ## 
    ## [[3]]
    ## [1] TRUE
    ## 
    ## [[4]]
    ## [1] TRUE
    ## 
    ## [[5]]
    ## [1] TRUE
    ## 
    ## [[6]]
    ## [1] TRUE
    ## 
    ## [[7]]
    ## [1] TRUE
    ## 
    ## [[8]]
    ## [1] TRUE
    ## 
    ## [[9]]
    ## [1] TRUE
    ## 
    ## [[10]]
    ## [1] TRUE

``` r
# IMPORTING
# Import training and test data, which we will soon combine, as we want to process explanatory variables together
dftrain <- read.csv(here::here("data-raw", "train.csv")) %>% dplyr::mutate(set = "train")
dftest <- read.csv(here::here("data-raw", "test.csv")) %>% dplyr::mutate(set = "test") %>% dplyr::mutate(SalePrice = NA)
df <- dplyr::bind_rows(dftrain, dftest)
```

# <a name="1"></a>

# Processing and missing analysis

First we take a glimpse at missing values. Columns with extremely high
values of NAs will be removed. There is patternistic missing (similar
percentage in many columns) which is likely due to the missing being
meaningful. The data description gives us insight on some of these NA
values.

``` r
df %>%
  naniar::miss_var_summary() %>%
  dplyr::filter(pct_miss > 0) %>%
  print(n=Inf)
```

    ## # A tibble: 35 x 3
    ##    variable     n_miss pct_miss
    ##    <chr>         <int>    <dbl>
    ##  1 PoolQC         2909  99.7   
    ##  2 MiscFeature    2814  96.4   
    ##  3 Alley          2721  93.2   
    ##  4 Fence          2348  80.4   
    ##  5 SalePrice      1459  50.0   
    ##  6 FireplaceQu    1420  48.6   
    ##  7 LotFrontage     486  16.6   
    ##  8 GarageYrBlt     159   5.45  
    ##  9 GarageFinish    159   5.45  
    ## 10 GarageQual      159   5.45  
    ## 11 GarageCond      159   5.45  
    ## 12 GarageType      157   5.38  
    ## 13 BsmtCond         82   2.81  
    ## 14 BsmtExposure     82   2.81  
    ## 15 BsmtQual         81   2.77  
    ## 16 BsmtFinType2     80   2.74  
    ## 17 BsmtFinType1     79   2.71  
    ## 18 MasVnrType       24   0.822 
    ## 19 MasVnrArea       23   0.788 
    ## 20 MSZoning          4   0.137 
    ## 21 Utilities         2   0.0685
    ## 22 BsmtFullBath      2   0.0685
    ## 23 BsmtHalfBath      2   0.0685
    ## 24 Functional        2   0.0685
    ## 25 Exterior1st       1   0.0343
    ## 26 Exterior2nd       1   0.0343
    ## 27 BsmtFinSF1        1   0.0343
    ## 28 BsmtFinSF2        1   0.0343
    ## 29 BsmtUnfSF         1   0.0343
    ## 30 TotalBsmtSF       1   0.0343
    ## 31 Electrical        1   0.0343
    ## 32 KitchenQual       1   0.0343
    ## 33 GarageCars        1   0.0343
    ## 34 GarageArea        1   0.0343
    ## 35 SaleType          1   0.0343

First, lets remove variables which have such a high percentage of NA
that they are not of use, even with correctly imputed values.

We know from the data description the NAs are categorical for
not-applicable for most variables. For categorical variables described
we will create this level. For other categorical variables we will use
the mode. Numeric variables which represent categories such as year or
MSzone, I will convert them to strings and impute with the mode.

For numeric variables we will impute with zero. I believe most numeric
NAs in this dataset to be meaningful zeros.

``` r
# PoolQC and MiscFeatures are too heavily not-applicable and will not be useful
df <- df %>%
  dplyr::select(-c("PoolQC", "MiscFeature"))


mymode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}



df <- df %>%
  dplyr::mutate(Functional = ifelse(is.na(Functional), mymode(Functional), Functional)) %>%
  dplyr::mutate(Electrical = ifelse(is.na(Electrical), mymode(Electrical), Electrical)) %>%
  dplyr::mutate(KitchenQual = ifelse(is.na(KitchenQual), mymode(KitchenQual), KitchenQual)) %>%
  dplyr::mutate(Exterior1st = ifelse(is.na(Exterior1st), mymode(Exterior1st), Exterior1st)) %>%
  dplyr::mutate(Exterior2nd = ifelse(is.na(Exterior2nd), mymode(Exterior2nd), Exterior2nd)) %>%
  dplyr::mutate(SaleType = ifelse(is.na(SaleType), mymode(SaleType), SaleType))  %>%
  dplyr::mutate(MasVnrType = ifelse(is.na(MasVnrType), mymode(MasVnrType), MasVnrType)) %>%
  dplyr::mutate(MSZoning = ifelse(is.na(MSZoning), mymode(MSZoning), MSZoning)) %>%
  dplyr::mutate(Utilities = ifelse(is.na(Utilities), mymode(Utilities), Utilities)) %>%
  dplyr::mutate(Alley = ifelse(is.na(Alley), "None", Alley)) %>%
  dplyr::mutate(FireplaceQu = ifelse(is.na(FireplaceQu), "None", FireplaceQu))%>%
  dplyr::mutate(Fence = ifelse(is.na(Fence), "None", Fence))%>%
  dplyr::mutate_at(c("BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"), ~ifelse(is.na(.), "None", .)) %>%
  dplyr::mutate_at(c("GarageType", "GarageFinish", "GarageQual", "GarageCond"), ~ifelse(is.na(.), "None", .)) %>%
  dplyr::mutate(MSSubClass = as.character(MSSubClass)) %>%
  dplyr::mutate(MSSubClass = ifelse(is.na(MSSubClass), mymode(MSSubClass), MSSubClass)) 
  
vars_missing <- df %>%
  naniar::miss_var_summary() %>%
  dplyr::filter(pct_miss > 0) %>%
  dplyr::pull(variable)
num_vars <- names(df)[sapply(df, class) %in% c("numeric", "integer")]
num_vars_missing <- vars_missing[vars_missing %in% num_vars]
num_vars_missing <- num_vars_missing[!num_vars_missing %in% c("SalePrice")]

df <- df %>%
  dplyr::mutate_at(num_vars_missing, ~tidyr::replace_na(., 0))  

#check to make sure we imputed everything
df %>%
  naniar::miss_var_summary() %>%
  dplyr::filter(pct_miss > 0) %>%
  print(n=Inf)
```

    ## # A tibble: 1 x 3
    ##   variable  n_miss pct_miss
    ##   <chr>      <int>    <dbl>
    ## 1 SalePrice   1459     50.0

Let‚Äôs also remove a few more variables which don‚Äôt contain enough
information (too sparse even when imputed) and factor non-ordinal
categories

``` r
# removing some vars that are entirely one group
df <- df %>%
  dplyr::select(-c("Street", "Utilities", "RoofMatl", "Heating"))
```

# <a name="2"></a>

# Exploratory data analysis

## Glimpse at the response variable (sale price)

``` r
# RESPONSE VARIABLE (move this past processing to EDA)
# check dist of response var is statistically different from normal
hist(dftrain$SalePrice,probability=T, main="Histogram of sales price", 50)
norm <- rnorm(100000, mean(dftrain$SalePrice), sd(dftrain$SalePrice))
lines(density(dftrain$SalePrice), col=1)
lines(density(norm),col=2)
```

![](README_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

``` r
# pointy dist with heavy tail on the right
shapiro.test(dftrain$SalePrice)
```

    ## 
    ##  Shapiro-Wilk normality test
    ## 
    ## data:  dftrain$SalePrice
    ## W = 0.86967, p-value < 2.2e-16

Sales price has a heavy right tail and a pointy distribution (black
line). Based on the shapiro test it is significantly different from a
normal distribution. We log scale to obtain a more normal distribution.

``` r
df <- df %>%
  dplyr::mutate(SalePrice = log(SalePrice))
hist(df$SalePrice,probability=T, main="Histogram of sales price", 50)
```

![](README_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

## Correlation

Now that we have done some of our cleaning complete we can take a look
at correlation plots.

``` r
charvars <- names(df)[sapply(df, class) == 'character']
dftemp <- df %>%
  dplyr::mutate_at(charvars, as.factor) %>%
  dplyr::mutate(MSSubClass = as.factor(MSSubClass))
nom_vars <- names(dftemp)[sapply(dftemp, class) == 'factor']
num_ord_vars <- names(dftemp)[!names(dftemp) %in% nom_vars]

temp <- dftemp %>%
  dplyr::select(c("SalePrice", num_ord_vars, nom_vars)) %>% # including sale price with ordinal as well
  dplyr::select(-set) %>%
  mixed_assoc() %>%
  dplyr::select(x, y, assoc) %>%
  tidyr::spread(y, assoc) %>%
  tibble::column_to_rownames("x")
```

    ## Note: Using an external vector in selections is ambiguous.
    ## i Use `all_of(num_ord_vars)` instead of `num_ord_vars` to silence this message.
    ## i See <https://tidyselect.r-lib.org/reference/faq-external-vector.html>.
    ## This message is displayed once per session.
    ## Note: Using an external vector in selections is ambiguous.
    ## i Use `all_of(nom_vars)` instead of `nom_vars` to silence this message.
    ## i See <https://tidyselect.r-lib.org/reference/faq-external-vector.html>.
    ## This message is displayed once per session.

``` r
corrplot::corrplot(temp %>% as.matrix %>% .[dim(temp)[1]:1,dim(temp)[1]:1], method = 'square', order = 'FPC', type = 'lower', diag = TRUE, tl.col = "black", tl.cex=.9)
```

![](README_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

## Correlation plot

Given the distance (non-correlation) between location and other aspect
variables, and the limited number of location variables, it may be vital
to squeeze the most out of these location variables. The number of
variables describing the size and quality is very high. There are
several variables which are highly correlated with each other
e.g.¬†garage and year built, MSSubClass and year built, etc. Niche item
like features such as screened porches, pools, fences, etc, do not seem
to be very correlated with the outcome.

## General plots

Just getting a glimpse at data.

``` r
library(ggplot2)
numvars <- names(df)[sapply(df, class) %in% c('numeric',"integer")]
temp <- df %>%
  dplyr::select(numvars) 
```

    ## Note: Using an external vector in selections is ambiguous.
    ## i Use `all_of(numvars)` instead of `numvars` to silence this message.
    ## i See <https://tidyselect.r-lib.org/reference/faq-external-vector.html>.
    ## This message is displayed once per session.

``` r
temp %>%
  tidyr::gather(-SalePrice, key = "var", value = "value") %>% 
  ggplot(aes(x = value, y = SalePrice)) +
    geom_point() +
    facet_wrap(~ var, scales = "free") +
    theme_bw()
```

![](README_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

``` r
ggplot(df) +
  geom_point(aes(x = GrLivArea, y = SalePrice))
```

![](README_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

We see two clear outliers on the right hand side beyond 4500. These
outliers also exist in the test dataset where we cannot remove them.
Let‚Äôs overwrite the SF value with the mean instead of removing them.

``` r
df %>% 
  dplyr::filter(GrLivArea > 4500) %>%
  dplyr::select(SalePrice, GrLivArea, YearBuilt, OverallQual, SaleType, MSZoning, TotalBsmtSF, set)
```

    ##   SalePrice GrLivArea YearBuilt OverallQual SaleType MSZoning TotalBsmtSF   set
    ## 1  12.12676      4676      2007          10      New       RL        3138 train
    ## 2  11.98293      5642      2008          10      New       RL        6110 train
    ## 3        NA      5095      2008          10      New       RL        5095  test

``` r
df <- df %>%
  dplyr::mutate(GrLivArea = ifelse(GrLivArea > 4500, mean(GrLivArea), GrLivArea))
```

## Sparse modeling for collapsing categories

Gertheiss and Tutz 2010

This paper presents Lasso-like solution paths that show how levels of
two categorical variables get merged together when regularization
strength increases. I wont be using weights as suggested in the paper
but borrowing the general concept.

For some of our sparse categorical data we will collapse them into fewer
groupings. One hot encoding is used for LASSO.

Let‚Äôs take a glimpse at counts for categorical variables (not including
ordinal).

``` r
nomvars <- names(df)[sapply(df, class) == 'factor']
temp <- df %>% 
  dplyr::select(nomvars)
```

    ## Note: Using an external vector in selections is ambiguous.
    ## i Use `all_of(nomvars)` instead of `nomvars` to silence this message.
    ## i See <https://tidyselect.r-lib.org/reference/faq-external-vector.html>.
    ## This message is displayed once per session.

``` r
summary(temp)
```

    ## < table of extent 0 x 0 >

The following variables caught my eye in terms of usefulness from
correlation, and in terms of having small categories.

``` r
nomvars_of_interest <- c(
"Neighborhood",
"SaleCondition",
"BsmtExposure",
"KitchenQual",
"BsmtQual")


df2 <- df %>%
  dplyr::select(c(nomvars_of_interest, "set")) %>%
  dplyr::mutate_at(nomvars_of_interest, as.factor) %>%
  dplyr::filter(set == "train") %>%
  dplyr::select(-set)
```

    ## Note: Using an external vector in selections is ambiguous.
    ## i Use `all_of(nomvars_of_interest)` instead of `nomvars_of_interest` to silence this message.
    ## i See <https://tidyselect.r-lib.org/reference/faq-external-vector.html>.
    ## This message is displayed once per session.

``` r
library(caret)
dummy <- dummyVars(" ~ .", data=df2)
df3 <- data.frame(predict(dummy, newdata = df2))
y <- df %>%
  dplyr::filter(set == "train") %>%
  dplyr::pull(SalePrice)


library(glmnet)
library(plotmo)
# Lets do individual lasso instead
# Can we figure out weights from munic real estate paper?
for (name in names(df2)) {
dftemp <- df3 %>% dplyr::select(dplyr::starts_with(name))
print(df3 %>% dplyr::select(dplyr::starts_with(name)) %>% colnames)
fit <- glmnet(dftemp, y = y, alpha = 1) #alpha 1 is lasso, 0 is ridge
plot_glmnet(fit, label=TRUE)
}
```

    ##  [1] "Neighborhood.Blmngtn" "Neighborhood.Blueste" "Neighborhood.BrDale" 
    ##  [4] "Neighborhood.BrkSide" "Neighborhood.ClearCr" "Neighborhood.CollgCr"
    ##  [7] "Neighborhood.Crawfor" "Neighborhood.Edwards" "Neighborhood.Gilbert"
    ## [10] "Neighborhood.IDOTRR"  "Neighborhood.MeadowV" "Neighborhood.Mitchel"
    ## [13] "Neighborhood.NAmes"   "Neighborhood.NoRidge" "Neighborhood.NPkVill"
    ## [16] "Neighborhood.NridgHt" "Neighborhood.NWAmes"  "Neighborhood.OldTown"
    ## [19] "Neighborhood.Sawyer"  "Neighborhood.SawyerW" "Neighborhood.Somerst"
    ## [22] "Neighborhood.StoneBr" "Neighborhood.SWISU"   "Neighborhood.Timber" 
    ## [25] "Neighborhood.Veenker"

![](README_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

    ## [1] "SaleCondition.Abnorml" "SaleCondition.AdjLand" "SaleCondition.Alloca" 
    ## [4] "SaleCondition.Family"  "SaleCondition.Normal"  "SaleCondition.Partial"

![](README_files/figure-gfm/unnamed-chunk-12-2.png)<!-- -->

    ## [1] "BsmtExposure.Av"   "BsmtExposure.Gd"   "BsmtExposure.Mn"  
    ## [4] "BsmtExposure.No"   "BsmtExposure.None"

![](README_files/figure-gfm/unnamed-chunk-12-3.png)<!-- -->

    ## [1] "KitchenQual.Ex" "KitchenQual.Fa" "KitchenQual.Gd" "KitchenQual.TA"

![](README_files/figure-gfm/unnamed-chunk-12-4.png)<!-- -->

    ## [1] "BsmtQual.Ex"   "BsmtQual.Fa"   "BsmtQual.Gd"   "BsmtQual.None"
    ## [5] "BsmtQual.TA"

![](README_files/figure-gfm/unnamed-chunk-12-5.png)<!-- -->

# <a name="sparse2"></a>

## Sparse modeling for collapsing categories (continued)

1.  [BsmtQual_ind](#sparse2) Creating an indicator for feature
    engineering.
2.  [BsmtExposure](#sparse2) Collapsing into an indicator.
3.  [GarageType_ind](#sparse2) Collapsing into an indicator. (main for
    feature engineering)
4.  [KitchenQual](#sparse2) Collapsing some of the smaller categories.
5.  [SaleCondition](#sparse2) Collapsing to norm, abnorm, and partial,
    other categories coefs get zeroed out early / not enough data

``` r
df <- df %>%
  dplyr::mutate(Neighborhood2 = ifelse(Neighborhood %in% c("BrDale", "IDOTRR"), "MeadowV", Neighborhood)) %>% #small hack for later feature that requires more samples
  dplyr::mutate(BsmtQual_ind = ifelse(BsmtQual %in% c("TA", "None", "Fa"), 0,
                                  ifelse(BsmtQual %in% c("Gd"), 1,
                                         ifelse(BsmtQual %in% c("Ex"), 2, NA)))) %>%
  dplyr::mutate(GarageType_ind = ifelse(GarageType %in% c("BuiltIn", "Attchd"), 1, 0)) %>%
  dplyr::mutate(BsmtExposure = ifelse(BsmtExposure == "Gd", 1, 0)) %>%
  dplyr::mutate(KitchenQual = ifelse(KitchenQual == "Ex", "Ex",
                                     ifelse(KitchenQual == "Gd", "Gd", "NW"))) %>%
  dplyr::mutate(SaleCondition = ifelse(SaleCondition == "Normal", "Normal",
                                       ifelse(SaleCondition == "Partial", "Partial", "Abnorml")))
```

# <a name="3"></a>

# Feature engineering

## Age and year

Let‚Äôs obtain how recently home was built / remodeled as those provide
different information from the year. The year/ month can provide time
series information and categorized year could provide information on
home style (e.g.¬†mid century modern homes built 1950-1969) but we will
leave for that another time. Since we are limited on time, let‚Äôs also
obtain a simple season indicator which is more simple than time series.

``` r
df <- df %>%
  dplyr::mutate(Age = YrSold-YearBuilt) %>%
  dplyr::mutate(AgeOfRemodel = YrSold-YearRemodAdd) %>%
  dplyr::mutate(AgeOfGarage = YrSold-GarageYrBlt) 

ggplot(df) +
  geom_point(aes(x = Age, y = SalePrice))
```

![](README_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

``` r
ggplot(df) +
  geom_point(aes(x = AgeOfRemodel, y = SalePrice))
```

![](README_files/figure-gfm/unnamed-chunk-14-2.png)<!-- -->

## Liveable square feet (SF)

I want to improve upon the square footage variables of the home (similar
to the idea behind the given above ground SF variable). As a home buyer
myself, I find listed square footage essential but often misleading as
it can refer to basement square footage (SF), or poorly utilized SF.
Basement SF is the most misleading so it will be my focus here. Since we
have square footage breakdown we can sum them in a custom way to create
a heuristic total square footage we will call this `LiveableModSF`.

``` r
# This variables represents home visual size from exterior and the usability of the square footage (SF). Ideally I should have one variable for each of these features of size and usability. Future work would be to make two separate variables here.
# log of baths used to imply diminishing return of baths
# baths, qual, and exposure of basement impact the usability of its square footage (SF)
# attached garage makes house look bigger but less impact
# screen porch makes house look bigger but less impact 

df <- df %>%
  dplyr::mutate(LiveableModSF = (((log(BsmtFullBath+2)*.3 + log(BsmtHalfBath+2)*.15)/2)+(BsmtQual_ind/10)+.3*BsmtExposure)*(BsmtFinSF1 + BsmtFinSF2) + X1stFlrSF + X2ndFlrSF + .3*GarageType_ind*GarageArea+.3*X3SsnPorch)

#garage SF gets zeroed if not attached to the house. attached garage makes home look bigger, feel bigger, curb appeal, etc
ggplot(df) +
  geom_point(aes(x = LiveableModSF, y = SalePrice))
```

![](README_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

## Open floor plan

Going a step further with total SF, let‚Äôs see if we can proxy for homes
with open floor plans.

``` r
df <- df %>% 
  dplyr::mutate(open_floor_proxy = log(GrLivArea)/TotRmsAbvGrd)
ggplot(df) +
  geom_point(aes(x = open_floor_proxy, y = SalePrice))
```

![](README_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

## Total number of bathrooms (zillow search feature :))

As a home buyer I know the total number of bathrooms is important
especially for families.

``` r
df <- df %>% 
  dplyr::mutate(BathTotal= FullBath + BsmtFullBath + (1/2)*HalfBath + (1/2)*BsmtHalfBath) 
ggplot(df) +
  geom_point(aes(x = BathTotal, y = SalePrice))
```

![](README_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

## Quality materials that are in good condition!

As a home buyer I‚Äôm familiar with old homes (like one I am in the middle
of purchasing) with quality materials not selling very well due to the
condition. Wood floors that need to be refinished can dramatically
decrease sale value! Let‚Äôs express this with a new feature.

``` r
df <- df %>%
  dplyr::mutate(ConditionalQuality = OverallQual*OverallCond)

ggplot(df) +
  geom_point(aes(x = ConditionalQuality, y = SalePrice))
```

![](README_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

## Keeping up with the Jones!

The value of the size of the lot depends on its location. We can see
from plotting the lot size alone that there is not a direct relationship
between lot size and price. I will try to feature engineer such that we
are comparing the size of the lot relative to other nearby lots. We will
make a second attempt at area specific lot size after target encoding
the neighborhood variables.

``` r
quants <- df %>%
  dplyr::group_by(Neighborhood2) %>%
  dplyr::summarise(zone_quant = quantile(LotArea, c(0, .25, 0.5, .75, 1)))
```

    ## `summarise()` has grouped output by 'Neighborhood2'. You can override using the
    ## `.groups` argument.

``` r
nnames <- unique(df$Neighborhood2)
neighborhood_specific_quantile <- list()

for(name in nnames){
  neighborhood_specific_quantile[[name]] <-  quants %>% 
  dplyr::filter(Neighborhood2 == name) %>%
  dplyr::pull(zone_quant)
  neighborhood_specific_quantile[[name]][1] <- 0
}

df <- df %>%
  dplyr::rowwise() %>%
  dplyr::mutate(keeping_up_with_jones = 
                  as.factor(
                    cut(LotArea, 
                        breaks = neighborhood_specific_quantile[[as.character(Neighborhood2)]], 
                        labels = c(1,2,3,4), 
                        include.lowest=TRUE,
                        )
                    )
                )

ggplot(df) +
  geom_boxplot(aes(x = keeping_up_with_jones, y = SalePrice))
```

![](README_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

## Log scaling of explanatory variables

A few of the continuous features are log normal. We will not transform
some of the continuous variables with zeros e.g.¬†garage and basement as
it will create even more distance between the meaningful zeros and the
other values.

``` r
# transforming continuous feature
df <- df %>%
  dplyr::mutate(LiveableModSF = log(LiveableModSF),
                GrLivArea = log(GrLivArea),
                LotArea = log(LotArea))
```

# <a name="4"></a>

# Modeling fitting

We will use the h2o auto ML platform for modeling / tuning. We will only
work with a simple GLM here. Ideally, with more time, we would explore
the auto tuning of xgboost and catboost in python.

Installing h2o latest release (if you need to remove/update old
installation use commented out code) h2o requires java, download java
before installing h2o <https://java.com/en/download/>

``` r
# # The following two commands remove any previously installed H2O packages for R.
# if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
# if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }
# 
# # Next, we download packages that H2O depends on.
# pkgs <- c("RCurl","jsonlite")
# for (pkg in pkgs) {
# if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
# }
# 
# # Now we download, install and initialize the H2O package for R.
# install.packages("h2o", type="source", repos="https://h2o-release.s3.amazonaws.com/h2o/rel-zumbo/2/R")
```

## Target encoding

I will be using the target encoding (TE) feature of the h2o platform.
The goal is to make the columns with several categories more usable.
I‚Äôve also added the land contour column for the sake of feature
engineering with the TE version.

``` r
to_be_encoded_columns <- c("Neighborhood", "MSSubClass", "MSZoning", "LandContour")

dfmodel <- df %>%
  dplyr::mutate_at(to_be_encoded_columns, as.factor)
# for target encoding later

dfmodeltrain <- dfmodel %>%
  dplyr::filter(set == "train")
dfmodeltest<- dfmodel %>%
  dplyr::filter(set == "test")

library(h2o)
```

    ## 
    ## ----------------------------------------------------------------------
    ## 
    ## Your next step is to start H2O:
    ##     > h2o.init()
    ## 
    ## For H2O package documentation, ask for help:
    ##     > ??h2o
    ## 
    ## After starting H2O, you can use the Web UI at http://localhost:54321
    ## For more information visit https://docs.h2o.ai
    ## 
    ## ----------------------------------------------------------------------

    ## 
    ## Attaching package: 'h2o'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     cor, sd, var

    ## The following objects are masked from 'package:base':
    ## 
    ##     %*%, %in%, &&, ||, apply, as.factor, as.numeric, colnames,
    ##     colnames<-, ifelse, is.character, is.factor, is.numeric, log,
    ##     log10, log1p, log2, round, signif, trunc

``` r
h2o.init(nthreads= -1, max_mem_size = "8g")
```

    ##  Connection successful!
    ## 
    ## R is connected to the H2O cluster: 
    ##     H2O cluster uptime:         1 hours 19 minutes 
    ##     H2O cluster timezone:       America/New_York 
    ##     H2O data parsing timezone:  UTC 
    ##     H2O cluster version:        3.36.1.2 
    ##     H2O cluster version age:    24 days  
    ##     H2O cluster name:           H2O_started_from_R_desktop-g_zzi767 
    ##     H2O cluster total nodes:    1 
    ##     H2O cluster total memory:   6.63 GB 
    ##     H2O cluster total cores:    16 
    ##     H2O cluster allowed cores:  16 
    ##     H2O cluster healthy:        TRUE 
    ##     H2O Connection ip:          localhost 
    ##     H2O Connection port:        54321 
    ##     H2O Connection proxy:       NA 
    ##     H2O Internal Security:      FALSE 
    ##     R Version:                  R version 4.1.3 (2022-03-10)

``` r
dfmodeltrain <- as.h2o(dfmodeltrain)
```

    ##   |                                                                              |                                                                      |   0%  |                                                                              |======================================================================| 100%

``` r
dfmodeltest <- as.h2o(dfmodeltest)
```

    ##   |                                                                              |                                                                      |   0%  |                                                                              |======================================================================| 100%

``` r
seed <- 1234

# For k_fold strategy we need to provide fold column
dfmodeltrain$fold <- h2o.kfold_column(data = dfmodeltrain, nfolds = 10, seed = seed)


# Train a TE model
target_encoder <- h2o.targetencoder(training_frame = dfmodeltrain,
                                    x = to_be_encoded_columns,
                                    y = "SalePrice",
                                    fold_column = "fold",
                                    data_leakage_handling = "KFold",
                                    blending = TRUE,
                                    inflection_point = 3,
                                    smoothing = 10,
                                    noise = 0.15,     # In general, the less data you have the more regularisation you need
                                    seed = seed)
```

    ##   |                                                                              |                                                                      |   0%  |                                                                              |======================================================================| 100%

``` r
# New target encoded train and test sets
transformed_train <- h2o.transform(target_encoder, dfmodeltrain, as_training=TRUE)
transformed_test <- h2o.transform(target_encoder, dfmodeltest, noise=0)

train <- as.data.frame(transformed_train)
test <- as.data.frame(transformed_test)
h2o.shutdown(prompt = TRUE)
```

    ## Are you sure you want to shutdown the H2O instance running at http://localhost:54321/ (Y/N)?

## More feature engineering with target encoded features.

The type of house may have a different value depending on where it is
located. We multiply the two MS features to make this feature. The land

``` r
#minor feature engineering with new target encoded columns. I'm interested in the feature/interaction of type of home and the type of location.
dffinal <- rbind(train %>% dplyr::select(-fold), test)
dffinal <- dffinal %>%
  dplyr::mutate(
                ms = MSSubClass_te * MSZoning_te,
                lot_age_and_qual = LotArea  - LandContour_te - sqrt(Age)) %>%
  dplyr::select(-to_be_encoded_columns)
```

    ## Note: Using an external vector in selections is ambiguous.
    ## i Use `all_of(to_be_encoded_columns)` instead of `to_be_encoded_columns` to silence this message.
    ## i See <https://tidyselect.r-lib.org/reference/faq-external-vector.html>.
    ## This message is displayed once per session.

``` r
library(ggplot2)
ggplot(dffinal) +
  geom_point(aes(x = lot_age_and_qual, y = SalePrice))
```

![](README_files/figure-gfm/unnamed-chunk-23-1.png)<!-- -->

``` r
# save our processed data as a checkpoint (next step is model fitting.)
write.csv(dffinal, row.names = FALSE, "dffinal.csv")
```

## Modeling fitting with grid search

I will be using some of the automated ML from the h2o platform. I will
grid search to tune hyperparameters. Ideally I would move this to python
for h2o.xgboost (not available for h2o windows R platform) and catboost.
Before modeling I will remove variables that are redundant after feature
engineering and variables with too low variability.

``` r
library(magrittr)
df <- read.csv("dffinal.csv") 
# I will be removing a few categorical columns where variability seems to be too low.
too_low_var <-
  c(
"Id",
"Condition2",    
"Condition1",     
"Alley",      
"Fence",        
"BsmtCond",     
"Electrical",   
"LandSlope",    
"ExterCond",  
"Functional",
"MoSold",
#below were used for hacks or are redundant after feature engineering
"Neighborhood2",
"LotArea",
"BsmtQual_ind",
"GarageYrBlt",
"YearBuilt", 
"YearRemodAdd",
"GarageType_ind",
"BsmtFullBath",
"HalfBath",
"BsmtHalfBath",
"LandContour_te",
"MSSubClass_te",
"MSZoning_te",
"X3SsnPorch",
"BsmtFinSF1",
"BsmtFinSF2",
"BsmtFinType2")
# one hot encode
df <- df %>%
  dplyr::select(-too_low_var)
```

    ## Note: Using an external vector in selections is ambiguous.
    ## i Use `all_of(too_low_var)` instead of `too_low_var` to silence this message.
    ## i See <https://tidyselect.r-lib.org/reference/faq-external-vector.html>.
    ## This message is displayed once per session.

``` r
# one hot encode
nomvars <- names(df)[sapply(df, class) %in% c('factor',"character")]
df2 <- df %>%
  dplyr::mutate_at(nomvars, as.factor)%>%
  dplyr::select(nomvars) %>%
  dplyr::select(-set)
library(caret)
dummy <- caret::dummyVars(" ~ .", data=df2, fullRank = FALSE)
df2 <- data.frame(predict(dummy, newdata = df2))
dff <- cbind(df2, df %>% dplyr::select(-nomvars))
dff$set <- df$set



train <- dff %>%
  dplyr::filter(set == "train") %>%
  dplyr::select(-set)
 
test <- dff %>%
  dplyr::filter(set == "test") %>%
  dplyr::select(-set) 


library(h2o)
h2o.init(nthreads= -1, max_mem_size = "8g")
```

    ##  Connection successful!
    ## 
    ## R is connected to the H2O cluster: 
    ##     H2O cluster uptime:         1 hours 19 minutes 
    ##     H2O cluster timezone:       America/New_York 
    ##     H2O data parsing timezone:  UTC 
    ##     H2O cluster version:        3.36.1.2 
    ##     H2O cluster version age:    24 days  
    ##     H2O cluster name:           H2O_started_from_R_desktop-g_zzi767 
    ##     H2O cluster total nodes:    1 
    ##     H2O cluster total memory:   6.63 GB 
    ##     H2O cluster total cores:    16 
    ##     H2O cluster allowed cores:  16 
    ##     H2O cluster healthy:        TRUE 
    ##     H2O Connection ip:          localhost 
    ##     H2O Connection port:        54321 
    ##     H2O Connection proxy:       NA 
    ##     H2O Internal Security:      FALSE 
    ##     R Version:                  R version 4.1.3 (2022-03-10)

``` r
train <- as.h2o(train)
```

    ##   |                                                                              |                                                                      |   0%  |                                                                              |======================================================================| 100%

``` r
test <- as.h2o(test)
```

    ##   |                                                                              |                                                                      |   0%  |                                                                              |======================================================================| 100%

``` r
seed <- 12345
# For k_fold strategy we need to provide fold column
# train$fold <- h2o.kfold_column(data = train, nfolds = 10, seed = seed)

 
# hyper_params <- list( lambda = seq(.001, .005, 0.0002),
#                        alpha = seq(0,.5, .1))
# 
# y <- "SalePrice"
# x <- setdiff(names(train), c(y, "fold"))
# # Grid search for selecting the best model
# grid <- h2o.grid(x = x, y = y ,
#                  training_frame = train,
#                  fold_column = "fold",
#                  algorithm = "glm", grid_id = "id10", hyper_params = hyper_params,seed=1,
#                  search_criteria = list(strategy = "Cartesian"))
# grid
# sortedGrid <- h2o.getGrid("id10", sort_by = "rmse", decreasing = FALSE)
# sortedGrid


y <- "SalePrice"
x <- setdiff(names(train), c(y, "fold"))
glmfin <- h2o.glm(x = x, y = y,
        training_frame = train, seed=1,
        alpha = .25,
        lambda = .002) #increasing lambda beyond grid to reduce predictors
```

    ##   |                                                                              |                                                                      |   0%  |                                                                              |======================================================================| 100%

``` r
glmfin
```

    ## Model Details:
    ## ==============
    ## 
    ## H2ORegressionModel: glm
    ## Model ID:  GLM_model_R_1655677405487_7 
    ## GLM Model: summary
    ##     family     link                              regularization
    ## 1 gaussian identity Elastic Net (alpha = 0.25, lambda = 0.002 )
    ##   number_of_predictors_total number_of_active_predictors number_of_iterations
    ## 1                        175                         131                    1
    ##     training_frame
    ## 1 train_sid_a7d6_7
    ## 
    ## Coefficients: glm coefficients
    ##          names coefficients standardized_coefficients
    ## 1    Intercept    12.270623                 12.024051
    ## 2 LotShape.IR1    -0.002945                 -0.001387
    ## 3 LotShape.IR2     0.017199                  0.002842
    ## 4 LotShape.IR3    -0.027960                 -0.002307
    ## 5 LotShape.Reg     0.000000                  0.000000
    ## 
    ## ---
    ##                     names coefficients standardized_coefficients
    ## 171      open_floor_proxy     0.044565                  0.011711
    ## 172             BathTotal     0.035876                  0.028177
    ## 173    ConditionalQuality     0.002315                  0.021348
    ## 174 keeping_up_with_jones     0.013686                  0.015382
    ## 175                    ms     0.000000                  0.000000
    ## 176      lot_age_and_qual     0.018474                  0.054086
    ## 
    ## H2ORegressionMetrics: glm
    ## ** Reported on training data. **
    ## 
    ## MSE:  0.01208382
    ## RMSE:  0.1099264
    ## MAE:  0.07789377
    ## RMSLE:  0.008556601
    ## Mean Residual Deviance :  0.01208382
    ## R^2 :  0.9242168
    ## Null Deviance :232.8007
    ## Null D.o.F. :1459
    ## Residual Deviance :17.64238
    ## Residual D.o.F. :1328
    ## AIC :-2037.896

``` r
pred <- exp(h2o.predict(object=glmfin, newdata=test))
```

    ##   |                                                                              |                                                                      |   0%  |                                                                              |======================================================================| 100%

``` r
pred
```

    ##   exp(predict)
    ## 1     117257.5
    ## 2     153474.3
    ## 3     178546.2
    ## 4     197184.6
    ## 5     188510.3
    ## 6     173011.1
    ## 
    ## [1459 rows x 1 column]

``` r
dftest <- read.csv(here::here("data-raw", "test.csv"))
final <- data.frame(Id = dftest %>% dplyr::pull(Id), SalePrice = as.vector(pred))
write.csv(final, row.names = FALSE, "predictions.csv")
```

## Future work

auto ML for xgboost and catboost, time series analysis, mixed modeling
