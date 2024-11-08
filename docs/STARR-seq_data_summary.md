# STARR-seq data summary

## Data sources
- All: `wget https://data.starklab.org/almeida/DeepSTARR/Tutorial/Sequences_activity_all.txt`
- Subset: `wget https://data.starklab.org/almeida/DeepSTARR/Tutorial/Sequences_activity_subset.txt`

## Data summary
Input sequences are 249bp longs and are stored as one-hot encoded numpy arrays of size `(249,9)`.

Output arrays have shape `(N,2)` where `N` is the number of sequences in the fold of data and the first and second columns correspond to the Dev and Hk target values, respectively. When data is updated for training distilled models, two additional columns are added for an output array shape of `(N,4)`. The third and fourth columns correspond to epistemic uncertainty for the Dev and Hk activity output heads, respectively.

### All
| fold  | nseqs  |
|-------|--------|
| Train | 402296 |
| Test  | 41186  |
| Val   | 40570  |
### Subset
| fold  | nseqs |
|-------|-------|
| Train | 50000 |
| Test  | 41186 |
| Val   | 40570 |

## Code
`parse_DeepSTARR_data.ipynb`: Parses data for training teacher models and distilled (student) models.