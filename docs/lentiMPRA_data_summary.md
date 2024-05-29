- [HepG2](#hepg2)
  - [Data](#data)
    - [Replicate mean only](#replicate-mean-only)
    - [Replicate mean and std.](#replicate-mean-and-std)
  - [Train/Test/Val splits](#traintestval-splits)
    - [Replicate mean only](#replicate-mean-only-1)
    - [Replicate mean and std.](#replicate-mean-and-std-1)
- [K562](#k562)
  - [Data](#data-1)
    - [Replicate mean only](#replicate-mean-only-2)
    - [Replicate mean and std.](#replicate-mean-and-std-2)
  - [Train/Test/Val splits](#traintestval-splits-1)
    - [Replicate mean only](#replicate-mean-only-3)
    - [Replicate mean and std.](#replicate-mean-and-std-3)

[Paper](https://www.biorxiv.org/content/10.1101/2023.03.05.531189v1)

# HepG2

## [Data](https://www.biorxiv.org/content/10.1101/2023.03.05.531189v1.full)
- `HepG2_seqs.csv`: contains sequence information from corresponding tab of [Supplementary Table 3](https://www.biorxiv.org/content/10.1101/2023.03.05.531189v1.supplementary-material)
  - 164307 unique seqs
- `HepG2_data.csv`: contains replicate activity measurements for each sequence + mean & std across replicates from summary level tab of [Supplementary Table 4](https://www.biorxiv.org/content/10.1101/2023.03.05.531189v1.supplementary-material)
  
### Replicate mean only
- `HepG2_data.h5`: one-hot encoded DNA seqs and corresponding mean activity saved in train/test/val splits
  - 139885 unique seqs
    - 139877/139885 have corresponding DNA seqs
  - 67485 seqs are reverse orientation
    - 153/67485 seqs contain `_Reversed_0` string (instead of `_Reversed`)
    - 61668/67485 seqs have complementary forward seqs with activity data
  - **123336 seqs from complementary fwd/rev seq pairs**
  - **16541 seqs (other)**

### Replicate mean and std.
- `HepG2_data_with_aleatoric.h5`: one-hot encoded DNA seqs and corresponding replicate level mean and standard deviation of activity saved in train/test/val splits
  - 139407/139885 unique seqs have replicate level std. values
    - 139399/139407 have corresponding DNA seqs
  - 67215 seqs w/ std. values are reverse orientation
    - 61253/67215 seqs have complementary forward seqs w/ mean & std. of activity data 
  - **122506 seqs from complementary fwd/rev seq pairs w/ mean & std. of activity data**
  - **16893 seqs (other) in dataset w/ mean & std. of activity data**
## Train/Test/Val splits
80/10/10

Randomly selected corresponding percentage of seqs from each the fwd/rev pairs and the other seqs. 

### Replicate mean only 
| fold  | nseqs  | fwd/rev | other |
|-------|--------|---------|-------|
| Train | 111901 | 98668   | 13233 |
| Test  | 13988  | 12334   | 1654  |
| Val   | 13988  | 12334   | 1654  |

### Replicate mean and std. 
| fold  | nseqs  | fwd/rev | other |
|-------|--------|---------|-------|
| Train | 111518 | 98004   | 13514 |
| Test  | 13939  | 12250   | 1689  |
| Val   | 13942  | 12252   | 1690  |

# K562

## [Data](https://www.biorxiv.org/content/10.1101/2023.03.05.531189v1.full)
- `K562_seqs.csv`: contains sequence information 
  - 243780 unique seqs from corresponding tab of [Supplementary Table 3](https://www.biorxiv.org/content/10.1101/2023.03.05.531189v1.supplementary-material)
- `K562_data.csv`: contains replicate activity measurements for each sequence + mean & std across replicates from summary level tab of [Supplementary Table 4](https://www.biorxiv.org/content/10.1101/2023.03.05.531189v1.supplementary-material)

### Replicate mean only
- `K562_data.h5`: one-hot encoded DNA seqs and corresponding mean activity saved in train/test/val splits
  - 226254 unique seqs
    - all have corresponding sequence data
  - 112868 seqs are reverse orientation
    - 109598/112868 seqs have complementary forward seqs with activity data
  - **219196 seqs from complementary fwd/rev pairs**
  - **7058 seqs (other)**
### Replicate mean and std.
- `K562_data_with_aleatoric.h5`: one-hot encoded DNA seqs and corresponding replicate level mean and standard deviation of activity saved in train/test/val splits
  - 225705/226254 unique seqs have replicate level std. values
    - all have corresponding sequence data
  - 112599 seqs are reverse orientation
    - 109156/112599 seqs have complementary forward seqs with replicate level mean & std. of activity 
  - **218312 seqs from complementary fwd/rev pairs** 
  - **7393 seqs (other)**
## Train/Test/Val splits
80/10/10

Randomly selected corresponding percentage of seqs from each the fwd/rev pairs and the other seqs. 

### Replicate mean only 
| fold  | nseqs  | fwd/rev | other |
|-------|--------|---------|-------|
| Train | 181002 | 175356   | 5646 |
| Test  | 22626  | 21920   | 706  |
| Val   | 22626  | 21920   | 706  |
### Replicate mean and std.
| fold  | nseqs  | fwd/rev | other |
|-------|--------|---------|-------|
| Train | 180564 | 174650   | 5914 |
| Test  | 22571  | 21832   | 739  |
| Val   | 22570  | 21830   | 740  |