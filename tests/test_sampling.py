"""We prevent duplicate dates or unmatched products from changing samples."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from run_gridded_extended_collocation import _observation_indices, _storm_sufficient_statistics

def duplicate_dataset():
    values=np.array([1.,2.,2.]).reshape(3,1,1)
    variables={p:(('observation','lat','lon'),values.copy()) for p in ('conus404','conus404ba','mrms','persiann_cdr')}
    variables.update(date=('observation',pd.to_datetime(['2020-01-01','2020-01-02','2020-01-02'])),is_core_day=('observation',[1,1,1]))
    return xr.Dataset(variables)

def test_identical_calendar_duplicates_count_once():
    indices,discarded=_observation_indices(duplicate_dataset(),'all_windows','core')
    np.testing.assert_array_equal(indices,[0,1])
    assert discarded==1

def test_conflicting_duplicate_values_raise():
    ds=duplicate_dataset()
    ds['mrms'][2,0,0]=3.
    with pytest.raises(ValueError,match='not identical'):
        _observation_indices(ds,'all_windows','expanded')

def test_short_storm_blocks_are_excluded_before_covariance():
    values=np.ones((7,4,1))
    matched=np.ones((7,1),dtype=bool)
    counts,_,_,qualifying,final=_storm_sufficient_statistics(values,matched,np.array([0,0,1,1,1,1,1]),2,3)
    np.testing.assert_array_equal(counts[:,0],[0,5])
    assert not final[:2].any()
