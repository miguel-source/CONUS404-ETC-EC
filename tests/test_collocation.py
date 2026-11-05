"""We check estimators against covariance matrices with known error moments."""
import numpy as np
from tc_precip_collocation import four_product_ec, triple_collocation

def known_covariance():
    slopes=np.array([1.,1.1,.9,1.2])
    errors=np.diag([.74,1.,.81,1.21])
    errors[0,1]=errors[1,0]=.3
    return 16*np.outer(slopes,slopes)+errors,errors

def test_recovers_known_error_covariance_and_variances():
    covariance,errors=known_covariance()
    metrics=four_product_ec(covariance)
    np.testing.assert_allclose(metrics['error_variance'],np.diag(errors),atol=1e-12)
    np.testing.assert_allclose(metrics['error_covariance_model_pair'],.3,atol=1e-12)
    assert metrics['valid_primary_ec']

def test_ec_model_variances_equal_separate_triplets():
    covariance,_=known_covariance()
    ec=four_product_ec(covariance)
    for model in (0,1):
        indices=[model,2,3]
        tc=triple_collocation(covariance[np.ix_(indices,indices)])
        np.testing.assert_allclose(tc['error_variance_unmasked'][0],ec['error_variance'][model],atol=1e-12)

def test_negative_variance_is_preserved_not_clipped():
    covariance=np.array([[1,.8,.8],[.8,1,.3],[.8,.3,1.]])
    assert np.linalg.eigvalsh(covariance).min()>0
    tc=triple_collocation(covariance)
    assert tc['error_variance_unmasked'][0]<0
    assert np.isnan(tc['error_sd_unmasked'][0])
    assert not tc['valid_triplet']

def test_zero_cross_covariance_is_rejected():
    assert not triple_collocation(np.eye(3))['valid_triplet']
