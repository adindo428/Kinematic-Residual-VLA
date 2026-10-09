import pytest
from core.dom_stopping import record_validation, should_stop, validate_rule

RULE={'enabled':True,'metric':'h20/action_mse','patience':3,'min_epochs':5,'relative_min_delta':.005}


def test_plateau_minimum_epochs_and_duplicate_validation():
    progress={'epoch':0}
    for epoch in range(1,6):
        progress['epoch']=epoch
        assert record_validation(progress,{'h20/action_mse':1.},RULE)==(epoch==5)
        count=progress['early_stopping']['bad_validations']
        record_validation(progress,{'h20/action_mse':1.},RULE)
        assert progress['early_stopping']['bad_validations']==count
    assert should_stop(progress,RULE)


def test_threshold_accumulates_small_improvements_and_resets_patience():
    progress={'epoch':1}
    record_validation(progress,{'h20/action_mse':1.},RULE)
    progress['epoch']=2
    record_validation(progress,{'h20/action_mse':.997},RULE)
    assert progress['early_stopping']['bad_validations']==1
    progress['epoch']=3
    record_validation(progress,{'h20/action_mse':.994},RULE)
    assert progress['early_stopping']['bad_validations']==0
    assert progress['early_stopping']['best_epoch']==3


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1.])
def test_nonfinite_or_negative_validation_is_failure_not_completion(value):
    with pytest.raises(ValueError):record_validation({'epoch':1},{'h20/action_mse':value},RULE)


def test_old_recipes_preserve_full_epoch_behavior():
    assert not record_validation({'epoch':100},{},None)
    with pytest.raises(ValueError):validate_rule({**RULE,'patience':0})
