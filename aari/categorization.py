from copy import deepcopy
from .ram_annotations import atari_dict

list_of_keys = [list(v.keys()) for v in atari_dict.values()]
all_keys = []
for a in list_of_keys:
    all_keys.extend(a)

small_object_names = ["ball", "missile"]
agent_names = ["agent", "player"]
localization_keys = [k for k in all_keys if any(coord in k for coord in ["_x", "_y", "_z", "_column"])]
agent_localization_keys = [k for k in localization_keys if
                           any(agent_name in k for agent_name in agent_names) and not any(
                               small_object_name in k for small_object_name in small_object_names)]
small_object_localization_keys = [k for k in localization_keys if
                                  any(small_object_name in k for small_object_name in small_object_names)]
other_localization_keys = [k for k in localization_keys if
                           k not in agent_localization_keys + small_object_localization_keys]

score_keys = [k for k in all_keys if "score" in k]
clock_keys = [k for k in all_keys if "clock" in k]
lives_keys = [k for k in all_keys if "lives" in k or "lifes" in k]
count_keys = [k for k in all_keys if "count" in k]
meter_keys = [k for k in all_keys if "meter" in k in k]
display_keys = [k for k in all_keys if "display" in k]
existence_keys = [k for k in all_keys if "bit_map" in k or "existence" in k]
score_clock_lives_display_keys = score_keys + clock_keys + lives_keys + meter_keys + display_keys
direction_keys = [k for k in all_keys if "direction" in k]
level_room_keys = [k for k in all_keys if "level" in k or "room" in k or "game_state" in k]
misc_keys = count_keys + existence_keys + level_room_keys + direction_keys

unused_keys = deepcopy(all_keys)

summary_key_dict = dict(small_object_localization=small_object_localization_keys,
                        agent_localization=agent_localization_keys,
                        other_localization=other_localization_keys,
                        score_clock_lives_display=score_clock_lives_display_keys,
                        misc_keys=misc_keys
                        )

detailed_key_dict = dict(
    agent_localization=agent_localization_keys,
    small_object_localization=small_object_localization_keys,
    other_localization=other_localization_keys,
    score=score_keys,
    clock=clock_keys,
    lives=lives_keys,
    count=count_keys,
    meter=meter_keys,
    display=display_keys,
    existence=existence_keys,
    direction=direction_keys,
    level_room=level_room_keys)