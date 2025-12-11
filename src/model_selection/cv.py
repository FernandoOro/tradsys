import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold

class PurgedKFold(_BaseKFold):
    """
    K-Fold Cross Validation with Purging and Embargo.
    Prevents leakage in financial time series with overlapping labels.
    Reference: Marcos Lopez de Prado.
    """
    def __init__(self, n_splits=5, t1=None, pct_embargo=0.01):
        """
        Args:
            n_splits: Number of folds.
            t1: Series with index=event_start_time, value=event_end_time. 
                Used to determine which events overlap with the test set.
            pct_embargo: Percent of data to drop after the test split.
        """
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        indices = np.arange(X.shape[0])
        n_samples = X.shape[0]
        
        # Standard KFold split indices
        # We use sklearn's logic to define the blocks, then we purge
        embargo = int(n_samples * self.pct_embargo)
        
        # Calculate bounds for the test sets
        # We manually split to control purging easily
        fold_size = n_samples // self.n_splits
        # test_ranges = [(i*fold_size, (i+1)*fold_size) for i in range(self.n_splits)]
        # Handle remainder
        
        test_starts = [(i * n_samples) // self.n_splits for i in range(self.n_splits)]
        test_ends = [((i + 1) * n_samples) // self.n_splits for i in range(self.n_splits)]
        
        for i in range(self.n_splits):
            test_start, test_end = test_starts[i], test_ends[i]
            
            # Define Test Indices
            test_indices = indices[test_start:test_end]
            
            # Define Train Indices (initial, before purging)
            # Train = All excluding [test_start, test_end + embargo]
            
            # Identify indices to purge
            # We need the times of the test set
            # Assuming X index is time or we have t1 aligned with X
            
            # Ideally, X should be a DataFrame with an index.
            # If t1 is None, we assume standard Gap (Pure Time Series Split without overlapping labels logic)
            # But the purpose of PurgedKFold IS overlapping labels.
            
            if self.t1 is None:
                # Fallback to simple Embargo
                train_indices = np.concatenate([
                    indices[:test_start],
                    indices[test_end + embargo:]
                ])
            else:
                # Purging Logic
                # t0 = start time of event of sample i
                # t1 = end time of event of sample i
                # Test Interval: [test_start_time, test_end_time]
                # Drop from Train any sample j where (t0_j, t1_j) overlaps with Test Interval
                
                # For efficiency in this implementation, we assume t1 is aligned with X and X is sorted.
                # Test Start Index -> Time
                t_test_start = self.t1.index[test_start]
                t_test_end = self.t1.index[test_end-1] # inclusive?
                
                # Find max t1 in the test set (when the last label in test finishes)
                test_max_t1 = self.t1.iloc[test_start:test_end].max()
                
                # Train indices must not start before test_max_t1 + embargo
                # And also train labels from BEFORE test must not end AFTER test starts.
                
                # Part 1: Train BEFORE Test
                # We need to find where t1 < t_test_start
                # Vectorized search or mask
                
                # Mask implementation for robustness
                train_mask = np.ones(n_samples, dtype=bool)
                train_mask[test_start:test_end] = False # Mask Test
                
                # Purge Overlap coming from Left (Train indices before Test)
                # Drop if t1_j > t_test_start
                # We only check indices < test_start
                mask_left = (self.t1 < t_test_start)  # Event ends before test starts
                # But wait, t1 contains end times.
                # So for j < test_start, keep if t1_j < t_test_start
                
                # This check ensures no label in left train set spills into test set
                train_mask[:test_start] = train_mask[:test_start] & mask_left[:test_start].values
                
                # Purge Overlap coming from Right (Train indices after Test)
                # Embargo: Don't start until test_max_t1 + embargo
                # But typically embargo is a fixed time or count?
                # De Prado uses time based. Here we use count based 'pct_embargo'
                
                embargo_idx = test_end + embargo
                train_mask[test_end:embargo_idx] = False
                
                train_indices = indices[train_mask]
                
            yield train_indices, test_indices
