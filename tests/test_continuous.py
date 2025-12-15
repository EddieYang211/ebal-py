"""Tests for the continuous entropy balancing module."""

import numpy as np
import pandas as pd
import pytest
from ebal import ebal_con


class TestEbalConInstantiation:
    """Tests for ebal_con class instantiation."""

    def test_default_instantiation(self):
        """Test that ebal_con can be instantiated with default parameters."""
        e = ebal_con()
        assert e.max_iterations == 500
        assert e.constraint_tolerance == 0.0001
        assert e.print_level == 0
        assert e.lr == 1
        assert e.PCA == True
        assert e.max_moment_treat == 2
        assert e.max_moment_X == 1

    def test_custom_parameters(self):
        """Test that ebal_con can be instantiated with custom parameters."""
        e = ebal_con(
            max_iterations=100,
            constraint_tolerance=0.001,
            print_level=1,
            lr=0.5,
            PCA=False,
            max_moment_treat=3
        )
        assert e.max_iterations == 100
        assert e.constraint_tolerance == 0.001
        assert e.print_level == 1
        assert e.lr == 0.5
        assert e.PCA == False
        assert e.max_moment_treat == 3


class TestEbalConBalance:
    """Tests for ebal_con ebalance method."""

    @pytest.fixture
    def simple_continuous_data(self):
        """Generate simple synthetic data with continuous treatment."""
        np.random.seed(42)
        n = 200

        # Generate covariates
        X1 = np.random.normal(0, 1, n)
        X2 = np.random.normal(0, 1, n)
        X = np.column_stack([X1, X2])

        # Generate continuous treatment (correlated with covariates)
        Treatment = 0.5 * X1 + 0.3 * X2 + np.random.normal(0, 1, n)

        return Treatment, X

    def test_ebalance_returns_dict(self, simple_continuous_data):
        """Test that ebalance returns a dictionary with expected keys."""
        Treatment, X = simple_continuous_data
        e = ebal_con(print_level=-1)
        result = e.ebalance(Treatment, X)

        assert isinstance(result, dict)
        assert 'converged' in result
        assert 'maxdiff' in result
        assert 'w' in result

    def test_ebalance_convergence(self, simple_continuous_data):
        """Test that ebalance converges on simple data."""
        Treatment, X = simple_continuous_data
        e = ebal_con(print_level=-1)
        result = e.ebalance(Treatment, X)

        assert result['converged'] == True
        assert result['maxdiff'] < e.constraint_tolerance

    def test_weights_shape(self, simple_continuous_data):
        """Test that weights have correct shape."""
        Treatment, X = simple_continuous_data
        e = ebal_con(print_level=-1)
        result = e.ebalance(Treatment, X)

        # Weights are returned as (n,) array (consistent with binary)
        assert result['w'].shape == (len(Treatment),)

    def test_weights_positive(self, simple_continuous_data):
        """Test that weights are positive."""
        Treatment, X = simple_continuous_data
        e = ebal_con(print_level=-1)
        result = e.ebalance(Treatment, X)

        assert np.all(result['w'] > 0)


class TestEbalConMoments:
    """Tests for different moment specifications."""

    @pytest.fixture
    def larger_data(self):
        """Generate larger synthetic data for moment tests."""
        np.random.seed(456)
        n = 500

        X1 = np.random.normal(0, 1, n)
        X2 = np.random.normal(0, 1, n)
        X3 = np.random.normal(0, 1, n)
        X = np.column_stack([X1, X2, X3])

        Treatment = 0.3 * X1 + 0.2 * X2 + 0.1 * X3 + np.random.normal(0, 1, n)

        return Treatment, X

    def test_moment_2(self, larger_data):
        """Test with max_moment_treat=2."""
        Treatment, X = larger_data
        e = ebal_con(max_moment_treat=2, print_level=-1)
        result = e.ebalance(Treatment, X)

        assert result['converged'] == True

    def test_moment_3(self, larger_data):
        """Test with max_moment_treat=3."""
        Treatment, X = larger_data
        e = ebal_con(max_moment_treat=3, print_level=-1, max_iterations=1000)
        result = e.ebalance(Treatment, X)

        # Higher moment may need more iterations or may not converge as easily
        # Just check it runs without error
        assert 'w' in result


class TestEbalConPCA:
    """Tests for PCA functionality."""

    @pytest.fixture
    def correlated_data(self):
        """Generate data with correlated features."""
        np.random.seed(789)
        n = 200

        # Create correlated features
        X1 = np.random.normal(0, 1, n)
        X2 = 0.8 * X1 + np.random.normal(0, 0.5, n)  # Correlated with X1
        X3 = np.random.normal(0, 1, n)
        X = np.column_stack([X1, X2, X3])

        Treatment = 0.5 * X1 + 0.3 * X3 + np.random.normal(0, 1, n)

        return Treatment, X

    def test_with_pca(self, correlated_data):
        """Test with PCA enabled."""
        Treatment, X = correlated_data
        e = ebal_con(PCA=True, print_level=-1)
        result = e.ebalance(Treatment, X)

        assert result['converged'] == True

    def test_without_pca(self, correlated_data):
        """Test with PCA disabled."""
        Treatment, X = correlated_data
        e = ebal_con(PCA=False, print_level=-1)
        result = e.ebalance(Treatment, X)

        # Should still produce results (may or may not converge)
        assert 'w' in result


class TestEbalConExceptions:
    """Tests for proper exception handling."""

    def test_missing_treatment_data_raises_valueerror(self):
        """Test that missing treatment data raises ValueError."""
        np.random.seed(42)
        Treatment = np.array([0.5, 1.2, np.nan, 2.1, 0.8])
        X = np.random.normal(0, 1, (5, 2))

        e = ebal_con(print_level=-1)
        with pytest.raises(ValueError, match="Treatment contains missing data"):
            e.ebalance(Treatment, X)

    def test_missing_x_data_raises_valueerror(self):
        """Test that missing X data raises ValueError."""
        np.random.seed(42)
        Treatment = np.array([0.5, 1.2, 0.3, 2.1, 0.8])
        X = np.array([[1, 2], [3, np.nan], [5, 6], [7, 8], [9, 10]])

        e = ebal_con(print_level=-1)
        with pytest.raises(ValueError, match="X contains missing data"):
            e.ebalance(Treatment, X)

    def test_shape_mismatch_raises_valueerror(self):
        """Test that shape mismatch raises ValueError."""
        np.random.seed(42)
        Treatment = np.array([0.5, 1.2, 0.3, 2.1, 0.8])
        X = np.random.normal(0, 1, (10, 2))  # Wrong shape

        e = ebal_con(print_level=-1)
        with pytest.raises(ValueError, match="length\\(Treatment\\) != nrow\\(X\\)"):
            e.ebalance(Treatment, X)

    def test_invalid_max_iterations_raises_typeerror(self):
        """Test that non-integer max_iterations raises TypeError."""
        np.random.seed(42)
        Treatment = np.random.normal(0, 1, 100)
        X = np.random.normal(0, 1, (100, 2))

        e = ebal_con(max_iterations=100.5, print_level=-1)
        with pytest.raises(TypeError, match="max_iterations must be an integer"):
            e.ebalance(Treatment, X)

    def test_constant_treatment_raises_valueerror(self):
        """Test that constant treatment raises ValueError."""
        np.random.seed(42)
        Treatment = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # Constant
        X = np.random.normal(0, 1, (5, 2))

        e = ebal_con(print_level=-1)
        with pytest.raises(ValueError, match="Treatment indicator must not be a constant"):
            e.ebalance(Treatment, X)

    def test_binary_treatment_raises_valueerror(self):
        """Test that binary treatment raises ValueError (should use ebal_bin)."""
        np.random.seed(42)
        Treatment = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        X = np.random.normal(0, 1, (10, 2))

        e = ebal_con(print_level=-1)
        with pytest.raises(ValueError, match="Treatment has 2 unique values"):
            e.ebalance(Treatment, X)


class TestEbalConCheckBalance:
    """Tests for ebal_con check_balance method."""

    @pytest.fixture
    def balance_test_data(self):
        """Generate data for balance checking tests."""
        np.random.seed(42)
        n = 200

        # Generate covariates with known properties
        X1 = np.random.normal(0, 1, n)  # Continuous
        X2 = (np.random.random(n) > 0.5).astype(float)  # Binary
        X3 = np.random.normal(0, 1, n)  # Continuous

        X = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3})

        # Generate continuous treatment (correlated with X1)
        Treatment = 0.5 * X1 + 0.3 * X2 + np.random.normal(0, 1, n)

        return Treatment, X

    def test_check_balance_returns_dataframe(self, balance_test_data):
        """Test that check_balance returns a DataFrame."""
        Treatment, X = balance_test_data
        e = ebal_con(print_level=-1)
        result = e.ebalance(Treatment, X)

        balance_df = e.check_balance(X, Treatment, result['w'])

        assert isinstance(balance_df, pd.DataFrame)

    def test_check_balance_columns(self, balance_test_data):
        """Test that check_balance returns DataFrame with correct columns."""
        Treatment, X = balance_test_data
        e = ebal_con(print_level=-1)
        result = e.ebalance(Treatment, X)

        balance_df = e.check_balance(X, Treatment, result['w'])

        expected_columns = ['Types', 'Before_weighting_corr',
                           'After_weighting_corr', 'After_weighting_pvalue']
        assert list(balance_df.columns) == expected_columns

    def test_check_balance_index_matches_columns(self, balance_test_data):
        """Test that DataFrame index matches input covariate names."""
        Treatment, X = balance_test_data
        e = ebal_con(print_level=-1)
        result = e.ebalance(Treatment, X)

        balance_df = e.check_balance(X, Treatment, result['w'])

        assert list(balance_df.index) == list(X.columns)

    def test_check_balance_types_correct(self, balance_test_data):
        """Test that Types column correctly identifies binary vs continuous."""
        Treatment, X = balance_test_data
        e = ebal_con(print_level=-1)
        result = e.ebalance(Treatment, X)

        balance_df = e.check_balance(X, Treatment, result['w'])

        # X1 and X3 are continuous, X2 is binary
        assert balance_df.loc['X1', 'Types'] == 'cont'
        assert balance_df.loc['X2', 'Types'] == 'binary'
        assert balance_df.loc['X3', 'Types'] == 'cont'

    def test_check_balance_correlations_in_valid_range(self, balance_test_data):
        """Test that correlations are in valid range [-1, 1]."""
        Treatment, X = balance_test_data
        e = ebal_con(print_level=-1)
        result = e.ebalance(Treatment, X)

        balance_df = e.check_balance(X, Treatment, result['w'])

        before_corr = balance_df['Before_weighting_corr']
        after_corr = balance_df['After_weighting_corr']
        assert (before_corr >= -1).all() and (before_corr <= 1).all()
        assert (after_corr >= -1).all() and (after_corr <= 1).all()

    def test_check_balance_pvalues_valid(self, balance_test_data):
        """Test that p-values are in valid range [0, 1]."""
        Treatment, X = balance_test_data
        e = ebal_con(print_level=-1)
        result = e.ebalance(Treatment, X)

        balance_df = e.check_balance(X, Treatment, result['w'])

        pvalues = balance_df['After_weighting_pvalue']
        assert (pvalues >= 0).all()
        assert (pvalues <= 1).all()

    def test_check_balance_zero_variance_dropped(self):
        """Test that zero-variance columns are dropped from output."""
        np.random.seed(42)
        n = 100

        X1 = np.random.normal(0, 1, n)
        X = pd.DataFrame({'X1': X1})

        Treatment = np.random.normal(0, 1, n)

        e = ebal_con(print_level=-1, PCA=False)
        result = e.ebalance(Treatment, X)

        # Now test check_balance with a DataFrame that includes a zero-variance column
        X_with_const = pd.DataFrame({'X1': X1, 'X2_constant': np.ones(n)})
        balance_df = e.check_balance(X_with_const, Treatment, result['w'])

        assert 'X2_constant' not in balance_df.index
        assert 'X1' in balance_df.index

    def test_check_balance_printing_controlled_by_print_level(self, balance_test_data, capsys):
        """Test that printing is controlled by print_level."""
        Treatment, X = balance_test_data

        # With print_level=-1, should not print
        e = ebal_con(print_level=-1)
        result = e.ebalance(Treatment, X)
        e.check_balance(X, Treatment, result['w'])
        captured = capsys.readouterr()
        assert captured.out == ""

        # With print_level=0, should print
        e2 = ebal_con(print_level=-1)
        result2 = e2.ebalance(Treatment, X)
        e2.print_level = 0
        e2.check_balance(X, Treatment, result2['w'])
        captured2 = capsys.readouterr()
        assert len(captured2.out) > 0

    def test_check_balance_accepts_numpy_array(self):
        """Test that check_balance accepts numpy arrays (not just DataFrames)."""
        np.random.seed(42)
        n = 200

        X1 = np.random.normal(0, 1, n)
        X2 = np.random.normal(0, 1, n)
        X = np.column_stack([X1, X2])  # numpy array

        Treatment = 0.5 * X1 + 0.3 * X2 + np.random.normal(0, 1, n)

        e = ebal_con(print_level=-1)
        result = e.ebalance(Treatment, X)

        # Should not raise an error with numpy array
        balance_df = e.check_balance(X, Treatment, result['w'])

        assert isinstance(balance_df, pd.DataFrame)
        expected_columns = ['Types', 'Before_weighting_corr',
                           'After_weighting_corr', 'After_weighting_pvalue']
        assert list(balance_df.columns) == expected_columns
        # Default column names should be X0, X1
        assert list(balance_df.index) == ['X0', 'X1']
