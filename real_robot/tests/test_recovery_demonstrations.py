import json
import pytest
from real_robot.recovery_demonstrations import register,validate_context


def context():
    return dict(schema='krvla_expert_recovery_v1',operator='fixture',parent_checkpoint='model/epoch-7',
        scenario='recover from missed object',deviation_source='actual_policy_deviation',
        expert_corrective_commands=True,synthetic_labels=False)


def test_register_preserves_recorded_commands_and_records_deviation_source(tmp_path):
    (tmp_path/'episodes').mkdir();episode=tmp_path/'episodes/episode_000000.json'
    episode.write_text('{"command": [1,2,3]}');original=episode.read_bytes()
    (tmp_path/'manifest.json').write_text(json.dumps(dict(collector='human_leader',mode='execute',split='train')))
    result=register(tmp_path,context())
    assert result['expert_recovery']['deviation_source']=='actual_policy_deviation'
    assert result['split']=='train' and episode.read_bytes()==original
    with pytest.raises(ValueError):register(tmp_path,context())


def test_no_synthetic_actions_or_unrecorded_recovery_context():
    for patch in [dict(expert_corrective_commands=False),dict(synthetic_labels=True),dict(deviation_source='synthetic_noise')]:
        with pytest.raises(ValueError):validate_context(dict(context(),**patch))


def test_recovery_provenance_survives_real_dataset_export(tmp_path):
    from real_robot.tests.test_dom_collection import make_session
    from real_robot.dom_dataset import export_sessions
    raw=tmp_path/'raw';make_session(raw)
    register(raw,context())
    export_sessions([raw],tmp_path/'export')
    rows=[json.loads(x) for x in (tmp_path/'export/meta/sources.jsonl').read_text().splitlines()]
    assert rows and all(row['expert_recovery']==context() for row in rows)
