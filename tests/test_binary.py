"""Tests for the binary entropy balancing module."""

import numpy as np
import pandas as pd
import pytest
from ebal import ebal_bin


class TestEbalBinInstantiation:
    """Tests for ebal_bin class instantiation."""

    def test_default_instantiation(self):
        """Test that ebal_bin can be instantiated with default parameters."""
        e = ebal_bin()
        assert e.max_iterations == 500
        assert e.constraint_tolerance == 0.0001
        assert e.print_level == 0
        assert e.lr == 1
        assert e.PCA == True
        assert e.effect == "ATT"

    def test_custom_parameters(self):
        """Test that ebal_bin can be instantiated with custom parameters."""
        e = ebal_bin(
            max_iterations=100,
            constraint_tolerance=0.001,
            print_level=1,
            lr=0.5,
            PCA=False,
            effect="ATC"
        )
        assert e.max_iterations == 100
        assert e.constraint_tolerance == 0.001
        assert e.print_level == 1
        assert e.lr == 0.5
        assert e.PCA == False
        assert e.effect == "ATC"


class TestEbalBinBalance:
    """Tests for ebal_bin ebalance method."""

    @pytest.fixture
    def simple_data(self):
        """Generate simple synthetic data for testing."""
        np.random.seed(42)
        n = 200

        # Generate covariates
        X1 = np.random.normal(0, 1, n)
        X2 = np.random.normal(0, 1, n)
        X = np.column_stack([X1, X2])

        # Generate treatment (biased toward higher X1)
        propensity = 1 / (1 + np.exp(-(X1 + 0.5 * X2)))
        Treatment = (np.random.random(n) < propensity).astype(int)

        # Generate outcome
        Y = 2 * Treatment + X1 + 0.5 * X2 + np.random.normal(0, 0.5, n)

        return Treatment, X, Y

    def test_ebalance_returns_dict(self, simple_data):
        """Test that ebalance returns a dictionary with expected keys."""
        Treatment, X, Y = simple_data
        e = ebal_bin(print_level=-1)
        result = e.ebalance(Treatment, X, Y)

        assert isinstance(result, dict)
        assert 'converged' in result
        assert 'maxdiff' in result
        assert 'w' in result
        assert 'wls' in result

    def test_ebalance_convergence(self, simple_data):
        """Test that ebalance converges on simple data."""
        Treatment, X, Y = simple_data
        e = ebal_bin(print_level=-1)
        result = e.ebalance(Treatment, X, Y)

        assert result['converged'] == True
        assert result['maxdiff'] < e.constraint_tolerance

    def test_weights_shape(self, simple_data):
        """Test that weights have correct shape."""
        Treatment, X, Y = simple_data
        e = ebal_bin(print_level=-1)
        result = e.ebalance(Treatment, X, Y)

        assert len(result['w']) == len(Treatment)

    def test_weights_positive(self, simple_data):
        """Test that weights are positive."""
        Treatment, X, Y = simple_data
        e = ebal_bin(print_level=-1)
        result = e.ebalance(Treatment, X, Y)

        assert np.all(result['w'] > 0)


class TestEbalBinEffects:
    """Tests for different effect types (ATT, ATC, ATE)."""

    @pytest.fixture
    def balanced_data(self):
        """Generate balanced synthetic data."""
        np.random.seed(123)
        n = 300

        X1 = np.random.normal(0, 1, n)
        X2 = np.random.normal(0, 1, n)
        X = np.column_stack([X1, X2])

        propensity = 1 / (1 + np.exp(-(0.5 * X1 + 0.3 * X2)))
        Treatment = (np.random.random(n) < propensity).astype(int)
        Y = 1.5 * Treatment + X1 + 0.5 * X2 + np.random.normal(0, 0.5, n)

        return Treatment, X, Y

    def test_att_estimation(self, balanced_data):
        """Test ATT estimation."""
        Treatment, X, Y = balanced_data
        e = ebal_bin(effect="ATT", print_level=-1)
        result = e.ebalance(Treatment, X, Y)

        assert result['converged'] == True

    def test_atc_estimation(self, balanced_data):
        """Test ATC estimation."""
        Treatment, X, Y = balanced_data
        e = ebal_bin(effect="ATC", print_level=-1)
        result = e.ebalance(Treatment, X, Y)

        assert result['converged'] == True

    def test_ate_estimation(self, balanced_data):
        """Test ATE estimation."""
        Treatment, X, Y = balanced_data
        e = ebal_bin(effect="ATE", print_level=-1)
        result = e.ebalance(Treatment, X, Y)

        assert result['converged'] == True


class TestEbalBinExceptions:
    """Tests for proper exception handling."""

    def test_invalid_effect_raises_valueerror(self):
        """Test that invalid effect raises ValueError."""
        with pytest.raises(ValueError, match="Effect must be one of ATT, ATC, or ATE"):
            ebal_bin(effect="INVALID")

    def test_missing_treatment_data_raises_valueerror(self):
        """Test that missing treatment data raises ValueError."""
        np.random.seed(42)
        Treatment = np.array([0, 1, np.nan, 1, 0])
        X = np.random.normal(0, 1, (5, 2))
        Y = np.random.normal(0, 1, 5)

        e = ebal_bin(print_level=-1)
        with pytest.raises(ValueError, match="Treatment contains missing data"):
            e.ebalance(Treatment, X, Y)

    def test_missing_x_data_raises_valueerror(self):
        """Test that missing X data raises ValueError."""
        np.random.seed(42)
        Treatment = np.array([0, 1, 0, 1, 0])
        X = np.array([[1, 2], [3, np.nan], [5, 6], [7, 8], [9, 10]])
        Y = np.random.normal(0, 1, 5)

        e = ebal_bin(print_level=-1)
        with pytest.raises(ValueError, match="X contains missing data"):
            e.ebalance(Treatment, X, Y)

    def test_shape_mismatch_raises_valueerror(self):
        """Test that shape mismatch raises ValueError."""
        np.random.seed(42)
        Treatment = np.array([0, 1, 0, 1, 0])
        X = np.random.normal(0, 1, (10, 2))  # Wrong shape
        Y = np.random.normal(0, 1, 5)

        e = ebal_bin(print_level=-1)
        with pytest.raises(ValueError, match="length\\(Treatment\\) != nrow\\(X\\)"):
            e.ebalance(Treatment, X, Y)

    def test_invalid_max_iterations_raises_typeerror(self):
        """Test that non-integer max_iterations raises TypeError."""
        np.random.seed(42)
        Treatment = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        X = np.random.normal(0, 1, (10, 2))
        Y = np.random.normal(0, 1, 10)

        e = ebal_bin(max_iterations=100.5, print_level=-1)
        with pytest.raises(TypeError, match="max_iterations must be an integer"):
            e.ebalance(Treatment, X, Y)

    def test_constant_treatment_raises_valueerror(self):
        """Test that constant treatment raises ValueError."""
        np.random.seed(42)
        Treatment = np.array([1, 1, 1, 1, 1])  # All treated
        X = np.random.normal(0, 1, (5, 2))
        Y = np.random.normal(0, 1, 5)

        e = ebal_bin(print_level=-1)
        with pytest.raises(ValueError, match="must contain both treatment and control"):
            e.ebalance(Treatment, X, Y)


class TestEbalBinCheckBalance:
    """Tests for ebal_bin check_balance method."""

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

        # Generate treatment (biased toward higher X1)
        propensity = 1 / (1 + np.exp(-(X1 + 0.5 * X2)))
        Treatment = (np.random.random(n) < propensity).astype(int)

        # Generate outcome
        Y = 2 * Treatment + X1 + 0.5 * X2 + np.random.normal(0, 0.5, n)

        return Treatment, X, Y

    def test_check_balance_returns_dataframe(self, balance_test_data):
        """Test that check_balance returns a DataFrame."""
        Treatment, X, Y = balance_test_data
        e = ebal_bin(print_level=-1)
        result = e.ebalance(Treatment, X, Y)

        balance_df = e.check_balance(X, Treatment, result['w'])

        assert isinstance(balance_df, pd.DataFrame)

    def test_check_balance_columns(self, balance_test_data):
        """Test that check_balance returns DataFrame with correct columns."""
        Treatment, X, Y = balance_test_data
        e = ebal_bin(print_level=-1)
        result = e.ebalance(Treatment, X, Y)

        balance_df = e.check_balance(X, Treatment, result['w'])

        expected_columns = ['Types', 'Before_weighting', 'After_weighting']
        assert list(balance_df.columns) == expected_columns

    def test_check_balance_index_matches_columns(self, balance_test_data):
        """Test that DataFrame index matches input covariate names."""
        Treatment, X, Y = balance_test_data
        e = ebal_bin(print_level=-1)
        result = e.ebalance(Treatment, X, Y)

        balance_df = e.check_balance(X, Treatment, result['w'])

        assert list(balance_df.index) == list(X.columns)

    def test_check_balance_types_correct(self, balance_test_data):
        """Test that Types column correctly identifies binary vs continuous."""
        Treatment, X, Y = balance_test_data
        e = ebal_bin(print_level=-1)
        result = e.ebalance(Treatment, X, Y)

        balance_df = e.check_balance(X, Treatment, result['w'])

        # X1 and X3 are continuous, X2 is binary
        assert balance_df.loc['X1', 'Types'] == 'cont'
        assert balance_df.loc['X2', 'Types'] == 'binary'
        assert balance_df.loc['X3', 'Types'] == 'cont'

    def test_check_balance_after_weighting_smaller(self, balance_test_data):
        """Test that after weighting differences are generally smaller."""
        Treatment, X, Y = balance_test_data
        e = ebal_bin(print_level=-1)
        result = e.ebalance(Treatment, X, Y)

        balance_df = e.check_balance(X, Treatment, result['w'])

        # After weighting, absolute differences should be small (close to 0)
        after_abs = balance_df['After_weighting'].abs()
        assert (after_abs < 0.1).all(), "After weighting differences should be small"

    def test_check_balance_zero_variance_dropped(self):
        """Test that zero-variance columns are dropped from output."""
        np.random.seed(42)
        n = 100

        X1 = np.random.normal(0, 1, n)
        X = pd.DataFrame({'X1': X1})

        Treatment = (np.random.random(n) > 0.5).astype(int)
        Y = np.random.normal(0, 1, n)

        e = ebal_bin(print_level=-1, PCA=False)
        result = e.ebalance(Treatment, X, Y)

        # Now test check_balance with a DataFrame that includes a zero-variance column
        X_with_const = pd.DataFrame({'X1': X1, 'X2_constant': np.ones(n)})
        balance_df = e.check_balance(X_with_const, Treatment, result['w'])

        assert 'X2_constant' not in balance_df.index
        assert 'X1' in balance_df.index

    def test_check_balance_atc_effect(self):
        """Test check_balance works correctly with ATC effect."""
        np.random.seed(42)
        n = 200

        X1 = np.random.normal(0, 1, n)
        X = pd.DataFrame({'X1': X1})

        propensity = 1 / (1 + np.exp(-X1))
        Treatment = (np.random.random(n) < propensity).astype(int)
        Y = 2 * Treatment + X1 + np.random.normal(0, 0.5, n)

        e = ebal_bin(effect="ATC", print_level=-1)
        result = e.ebalance(Treatment, X, Y)

        balance_df = e.check_balance(X, Treatment, result['w'])

        assert isinstance(balance_df, pd.DataFrame)
        assert 'After_weighting' in balance_df.columns

    def test_check_balance_ate_effect(self):
        """Test check_balance works correctly with ATE effect."""
        np.random.seed(42)
        n = 200

        X1 = np.random.normal(0, 1, n)
        X = pd.DataFrame({'X1': X1})

        propensity = 1 / (1 + np.exp(-X1))
        Treatment = (np.random.random(n) < propensity).astype(int)
        Y = 2 * Treatment + X1 + np.random.normal(0, 0.5, n)

        e = ebal_bin(effect="ATE", print_level=-1)
        result = e.ebalance(Treatment, X, Y)

        balance_df = e.check_balance(X, Treatment, result['w'])

        assert isinstance(balance_df, pd.DataFrame)
        assert 'After_weighting' in balance_df.columns

    def test_check_balance_printing_controlled_by_print_level(self, balance_test_data, capsys):
        """Test that printing is controlled by print_level."""
        Treatment, X, Y = balance_test_data

        # With print_level=-1, should not print
        e = ebal_bin(print_level=-1)
        result = e.ebalance(Treatment, X, Y)
        e.check_balance(X, Treatment, result['w'])
        captured = capsys.readouterr()
        assert captured.out == ""

        # With print_level=0, should print
        e2 = ebal_bin(print_level=-1)
        result2 = e2.ebalance(Treatment, X, Y)
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

        propensity = 1 / (1 + np.exp(-(X1 + 0.5 * X2)))
        Treatment = (np.random.random(n) < propensity).astype(int)
        Y = 2 * Treatment + X1 + 0.5 * X2 + np.random.normal(0, 0.5, n)

        e = ebal_bin(print_level=-1)
        result = e.ebalance(Treatment, X, Y)

        # Should not raise an error with numpy array
        balance_df = e.check_balance(X, Treatment, result['w'])

        assert isinstance(balance_df, pd.DataFrame)
        assert list(balance_df.columns) == ['Types', 'Before_weighting', 'After_weighting']
        # Default column names should be X0, X1
        assert list(balance_df.index) == ['X0', 'X1']
