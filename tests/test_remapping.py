"""We verify conservative depth averaging and the missing-area threshold."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
import shapely
import xarray as xr
import prepare_gridded_persiann_etc as preparation

def fixture(values):
    # Coordinates and areas in an equal-area Cartesian grid. Each fine cell is
    # 0.25 m2; each target cell is 1 m2. Areas come from actual intersections.
    x=np.array([.25,.75,1.25,1.75])
    y=np.array([.25,.75])
    source=[shapely.box(xc-.25,yc-.25,xc+.25,yc+.25) for yc in y for xc in x]
    targets=[shapely.box(0,0,1,1),shapely.box(1,0,2,1)]
    weights=np.array([[t.intersection(s).area for s in source] for t in targets])
    da=xr.DataArray(np.asarray(values,dtype=float).reshape(1,2,4),dims=('time','y','x'),coords={'time':[0],'x':x,'y':y}).rio.write_crs('EPSG:5070')
    remap=preparation.SparseRemap(product='test',source_grid_hash=preparation.grid_hash(da),target_grid_hash='test',row_start=0,row_stop=2,col_start=0,col_stop=4,matrix=sparse.csr_matrix(weights),target_area_m2=np.ones(2),geometric_coverage_fraction=np.ones(2),active_flat_indices=np.array([0,1]))
    target=SimpleNamespace(shape=(1,2),study_mask=np.ones((1,2),dtype=bool))
    return da,remap,target

def test_remapping_preserves_constant_depth_and_volume():
    da,remap,target=fixture(np.full(8,12.))
    mapped,coverage=preparation.apply_sparse_remap(da,remap,target)
    np.testing.assert_allclose(mapped,12.)
    np.testing.assert_allclose(coverage,1.)
    np.testing.assert_allclose(np.sum(da.values*.25),np.sum(mapped))

def test_remapping_heterogeneous_depth_preserves_volume():
    da,remap,target=fixture([2,4,6,8,2,4,6,8])
    mapped,_=preparation.apply_sparse_remap(da,remap,target)
    np.testing.assert_allclose(mapped,[[[3,7]]])
    np.testing.assert_allclose(np.sum(da.values*.25),np.sum(mapped))

def test_missing_area_is_masked_rather_than_zero_rainfall():
    da,remap,target=fixture([np.nan,12,12,12,12,12,12,12])
    mapped,coverage=preparation.apply_sparse_remap(da,remap,target)
    assert coverage[0,0,0]==pytest.approx(.75)
    assert np.isnan(mapped[0,0,0])
    assert mapped[0,0,1]==pytest.approx(12)
