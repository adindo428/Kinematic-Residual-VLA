"""Shared occlusion-estimation profile for live learned-policy evaluations."""
import json
from pathlib import Path

DEFAULT_MAX_GAP_S=2.
DEFAULT_MAX_DISPLACEMENT_M=.05
MODELS=('auto','constant_velocity','contact_stop_assumption')
OBJECTS={'orange':'rolling orange','marker':'moving green marker','striped_ball':'moving orange and white striped ball'}


def add_object_estimation_arguments(parser):
    parser.add_argument('--object-motion',choices=['auto','static','moving'],default='auto',
        help='auto uses checkpoint task data; static uses the prior contact-stop hypothesis, moving preserves velocity')
    parser.add_argument('--object-occlusion-model',choices=MODELS,default='auto',help='Explicit motion-hypothesis override')
    parser.add_argument('--object-prediction-max-gap-s',type=float,default=DEFAULT_MAX_GAP_S,
        help='Maximum prediction age; uncertainty/travel can expire it earlier; 0 disables')
    parser.add_argument('--object-prediction-max-displacement-m',type=float,default=DEFAULT_MAX_DISPLACEMENT_M)


def resolve_model(model='auto',motion='auto',dataset=None,task=None):
    if model not in MODELS or motion not in ('auto','static','moving'):raise ValueError('Unknown object-estimation profile')
    if model!='auto':return model
    if motion=='auto':
        motion='moving'  # With no dataset evidence, do not assume that motion stopped.
        if dataset is not None and task in OBJECTS:
            root=Path(dataset)
            episodes={row['episode_index']:row for row in map(json.loads,(root/'meta/episodes.jsonl').read_text().splitlines())}
            sources=list(map(json.loads,(root/'meta/sources.jsonl').read_text().splitlines()))
            matching=[]
            for row in sources:
                tasks=episodes[row['episode_index']]['tasks']
                descriptions=[json.loads(value) if isinstance(value,str) else value for value in tasks]
                if any(value.get('objects')==OBJECTS[task] for value in descriptions):matching.append(row)
            if matching and all(row.get('static_object_collection') for row in matching):motion='static'
    return 'contact_stop_assumption' if motion=='static' else 'constant_velocity'
