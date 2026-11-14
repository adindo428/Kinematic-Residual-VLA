"""Add current simulator time and the previously applied command to observations."""
import sampled_video_table1_server as wrapper
from table1_video_sampling import patch_source as video_patch


def patch_source(text):
    needle='                "index": len(sim_results["cam_views"]) - 1,'
    assert text.count(needle)==1
    text=text.replace(needle,needle+'\n                "timestamp": (len(sim_results["cam_views"])-1)*float(env.env.step_dt),\n                "previous_gripper_command": None if last_action is None else last_action[:, -1].cpu().numpy(),')
    return video_patch(text)


if __name__=='__main__':
    wrapper.patch_source=patch_source
    wrapper.main()
