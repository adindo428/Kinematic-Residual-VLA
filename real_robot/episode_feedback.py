"""Persistent visual acknowledgments for collector episode boundaries."""
import cv2
import numpy as np


class EpisodeFeedback:
    def __init__(self, default_success=True):
        self.default_success = default_success
        self.last_saved = 'No episodes saved yet'
        self.last_outcome = None

    def saved(self, episode_id, outcome):
        self.last_saved = f'EPISODE {episode_id+1} SAVED - {outcome.upper()}'
        self.last_outcome = outcome

    def snapshot(self, current, following, successes, goal, state=None):
        return dict(controls=('Terminal focused: SPACE start | ENTER stop/save | X failed | Q finish | T pause' if self.default_success
                              else 'Terminal focused: N start | V success | X failed | Q finish | T pause'),
                    state=state or ('RECORDING' if current is not None else 'READY' if following else 'PAUSED'),
                    episode=None if current is None else current['episode_id']+1,
                    successes=successes, goal=goal, last_saved=self.last_saved,
                    last_outcome=self.last_outcome)


def render_banner(status):
    """Large fixed-size banner; previous save remains visible during the next run."""
    image=np.full((160,1280,3),24,np.uint8)
    color={'success':(70,230,100),'failed':(255,95,75),'incomplete':(255,190,60)}.get(status.get('last_outcome'),(220,220,220))
    # RGB array: this banner goes directly to Rerun, without BGR conversion.
    cv2.rectangle(image,(0,0),(1279,159),color,5)
    cv2.putText(image,status['last_saved'],(24,48),cv2.FONT_HERSHEY_SIMPLEX,1.05,color,2,cv2.LINE_AA)
    current=('READY - SPACE TO START' if status['state']=='READY' else f"{status['state']}") + (f" - episode {status['episode']}" if status.get('episode') else '')
    progress=f"Successes: {status['successes']}/{status.get('goal') or 'unlimited'}"
    cv2.putText(image,current,(24,97),cv2.FONT_HERSHEY_SIMPLEX,.85,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(image,progress,(790,97),cv2.FONT_HERSHEY_SIMPLEX,.85,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(image,status.get('controls','SPACE start | ENTER stop/save | X failed | Q finish | T pause'),(24,138),cv2.FONT_HERSHEY_SIMPLEX,.65,(210,210,210),1,cv2.LINE_AA)
    return image
