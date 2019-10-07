atari_dict = {

    "battlezone": dict(  # red_enemy_x=75,
        blue_tank_facing_direction=46,  # 17 left 21 forward 29 right
        blue_tank_size_y=47,  # tank gets larger as it gets closer
        blue_tank_x=48,
        blue_tank2_facing_direction=52,
        blue_tank2_size_y=53,
        blue_tank2_x=54,
        num_lives=58,
        missile_y=105,
        compass_needles_angle=84,
        angle_of_tank=4,  # as shown by what the mountains look like
        left_tread_position=59,  # got to mod this number by 8 to get unique values
        right_tread_position=60,  # got to mod this number by 8 to get unique values
        crosshairs_color=108,  # 0 if black 46 if yellow
        score=29),
    "skiing": dict(player_x=25,
                   clock_m=104,
                   clock_s=105,
                   clock_ms=106,
                   score=107,
                   object_y=range(87, 94)),  # object_y_1 is y position of whatever topmost object on the screen is

    "pitfall": dict(clock_sec=89, clock_min=88),

}
