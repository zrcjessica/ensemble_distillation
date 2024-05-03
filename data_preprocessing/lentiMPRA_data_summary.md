- [HepG2](#hepg2)
  - [Data](#data)
  - [Train/Test/Val splits](#traintestval-splits)
    - [Train](#train)
    - [Test](#test)
    - [Val](#val)
- [K562](#k562)
  - [Data](#data-1)
  - [Train/Test/Val splits](#traintestval-splits-1)
    - [Train](#train-1)
    - [Test](#test-1)
    - [Val](#val-1)

[Paper](https://www.biorxiv.org/content/10.1101/2023.03.05.531189v1)

# HepG2

## [Data](https://www.biorxiv.org/content/10.1101/2023.03.05.531189v1.full)
- `HepG2_seqs.csv`: contains sequence information from corresponding tab of [Supplementary Table 3](https://www.biorxiv.org/content/10.1101/2023.03.05.531189v1.supplementary-material)
  - 164307 unique seqs
- `HepG2_data.csv`: contains replicate activity measurements for each sequence + mean & std across replicates from summary level tab of [Supplementary Table 4](https://www.biorxiv.org/content/10.1101/2023.03.05.531189v1.supplementary-material)
  - 139885 unique seqs
    - 139877/139885 have corresponding DNA seqs
  - 67485 seqs are reverse orientation
    - 153/67485 seqs contain `_Reversed_0` string (instead of `_Reversed`)
    - 61668/67485 seqs have complementary forward seqs with activity data
  - **123336 seqs from complementary fwd/rev seq pairs**
  - **16541 seqs (other)**
- `HepG2_data.h5`: one-hot encoded DNA seqs and corresponding mean activity saved in train/test/val splits

## Train/Test/Val splits
80/10/10

Randomly selected corresponding percentage of seqs from each the fwd/rev pairs and the other seqs. 

| fold  | nseqs  |
|-------|--------|
| Train | 111901 |
| Test  | 13988  |
| Val   | 13988  |

### Train
- 111901 seqs
  - 98668 seqs in fwd/rev pair
  - 13233 seqs (other)

### Test
- 13988 seqs
  - 12334 seqs in fwd/rev pair
  - 1654 seqs (other)

### Val
- 13988 seqs
  - 12334 seqs in fwd/rev pair
  - 1654 seqs (other)

# K562

## [Data](https://www.biorxiv.org/content/10.1101/2023.03.05.531189v1.full)
- `K562_seqs.csv`: contains sequence information 
  - 243780 unique seqs from corresponding tab of [Supplementary Table 3](https://www.biorxiv.org/content/10.1101/2023.03.05.531189v1.supplementary-material)
- `K562_data.csv`: contains replicate activity measurements for each sequence + mean & std across replicates from summary level tab of [Supplementary Table 4](https://www.biorxiv.org/content/10.1101/2023.03.05.531189v1.supplementary-material)
  - 226254 unique seqs
    - all have corresponding sequence data
  - 112868 seqs are reverse orientation
    - 109598/112868 seqs have complementary forward seqs with activity data
  - **219196 seqs from complementary fwd/rev pairs**
  - **7058 seqs (other)**
- `K562_data.h5`: one-hot encoded DNA seqs and corresponding mean activity saved in train/test/val splits

## Train/Test/Val splits
80/10/10

Randomly selected corresponding percentage of seqs from each the fwd/rev pairs and the other seqs. 

| fold  | nseqs  |
|-------|--------|
| Train | 181002 |
| Test  | 21920  |
| Val   | 21920  |

### Train
- 181002 seqs
  - 175356 seqs in fwd/rev pair
  - 5646 seqs (other)

### Test
- 22626 seqs
  - 21920 seqs in fwd/rev pair
  - 706 seqs (other)

### Val
- 22626 seqs
  - 21920 seqs in fwd/rev pair
  - 706 seqs (other)