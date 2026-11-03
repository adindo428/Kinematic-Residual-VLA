import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pytest
from report import verify_records

def rows():
    return [{'env_name':name,'trial':trial,'status':status,'success':status==0,
             'n_steps':steps,'step_dt_s':.04,'task_time_s':steps*.04}
            for name in ('CR_a','CR_b') for trial,status,steps in ((1,0,100),(2,1,300))]

def test_success_is_binary_manipulation_and_time_includes_failures():
    result=verify_records(rows(),['CR_a','CR_b'],2,{'CR_a':12,'CR_b':12})
    assert result['success_percent']==50
    assert result['mean_task_time_s']==8

@pytest.mark.parametrize('issue',['missing','extra','duplicate','no_action','inconsistent_success','bad_time','excess_horizon'])
def test_invalid_benchmark_cannot_publish(issue):
    data=rows()
    if issue=='missing':data.pop()
    elif issue=='extra':data.append(dict(data[0]))
    elif issue=='duplicate':data[1]['trial']=1
    elif issue=='no_action':data[1]['status']=2
    elif issue=='inconsistent_success':data[1]['success']=True
    elif issue=='bad_time':data[1]['task_time_s']=4
    elif issue=='excess_horizon':data[1].update(n_steps=400,task_time_s=16)
    with pytest.raises(AssertionError):verify_records(data,['CR_a','CR_b'],2,{'CR_a':12,'CR_b':12})
