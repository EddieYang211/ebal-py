# Entropy Balancing for Binary and Continuous Treatment

Python implementation of Entropy Balancing for binary and continuous treatment

> Author: Eddie Yang

ebal for binary treatment is based on Hainmueller ([2012](https://web.stanford.edu/~jhain/Paper/PA2012.pdf)) and ebal for continuous treatment is based on Tübbicke ([2020](https://arxiv.org/abs/2001.06281)) and Vegetabile et al. ([2021](https://arxiv.org/pdf/2003.02938.pdf)).

## Installation

```bash
# From PyPI
pip install ebal

# From GitHub
pip install git+https://github.com/EddieYang211/ebal-py.git

# For development
git clone https://github.com/EddieYang211/ebal-py.git
cd ebal
pip install -e .
```

## Examples

```python
# Binary
import pandas as pd
from ebal import ebal_bin

df = pd.read_csv('~/data.csv')

treatment="treat"
tmp_df = df[df[treatment].isin([0, 1])]
tmp_df = tmp_df.sort_values(by=[treatment])
tmp_df = tmp_df.reset_index(drop=True)

Y = tmp_df.Y.values
X = tmp_df.loc[:,0:10]                         #features
T = tmp_df[treatment].values                   #treatment
e = ebal_bin()
out = e.ebalance(T, X, Y)

# print ebal-weighted regression result
out['wls'].summary()

# check post-weighting balance between treatment and control groups
e.check_balance(X, T, out['w'])
```

```python
# Continuous
import pandas as pd
from ebal import ebal_con

df = pd.read_csv('~/data_cont.csv')

Y = df.Y.values
X = df.iloc[:,4:]                         #features
T = df.T.values                           #treatment

e = ebal_con()
out = e.ebalance(T, X)

# check post-weighting Pearson correlation
e.check_balance(X, T, out['w'])
```

## Documentation

For detailed documentation in Chinese:
- [Binary Treatment Documentation](docs/README_binary.md)
- [Continuous Treatment Documentation](docs/README_continuous.md)

## Reference
* Hainmueller, J. (2012). Entropy balancing for causal effects: A multivariate reweighting method to produce balanced samples in observational studies. Political analysis, 20(1), 25-46.
* Vegetabile, B. G., Griffin, B. A., Coffman, D. L., Cefalu, M., Robbins, M. W., & McCaffrey, D. F. (2021). Nonparametric estimation of population average dose-response curves using entropy balancing weights for continuous exposures. Health Services and Outcomes Research Methodology, 21(1), 69-110.
* Tübbicke, S. (2020). Entropy balancing for continuous treatments. arXiv preprint arXiv:2001.06281.
* R Package 'ebal': https://github.com/cran/ebal

## License

MIT License - see [LICENSE](LICENSE) for details.
