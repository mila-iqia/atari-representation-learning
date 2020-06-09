"""In gym, the RAM is represented as an 128-element array, where each element in the array can range from 0 to 255

The atari_dict below is organized as so:
    key: the name of the game
    value: the game dictionary

Game dictionary is organized as:
    key: state variable name
    value: the element in the RAM array where the value of that state variable is stored
            e.g. the value of the x coordinate of the player in asteroids is stored in the 73rd (counting up from 0)
            element of the RAM array (when the player in asteroids moves horizontally, ram_array[73] should change
            in value correspondingly)
"""
""" MZR player_direction values:
         72:  facing left,
         40:  facing left, climbing down ladder/rope
         24:  facing left, climbing up ladder/rope
         128: facing right
         32:  facing right, climbing down ladder/rope
         16:  facing right climbing up ladder/rope """

atari_dict = {
    "asteroids": dict(enemy_asteroids_y=[3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19],
                      enemy_asteroids_x=[21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37],
                      player_x=73,
                      player_y=74,
                      num_lives_direction=60,
                      player_score_high=61,
                      player_score_low=62,
                      player_missile_x1=83,
                      player_missile_x2=84,
                      player_missile_y1=86,
                      player_missile_y2=87,
                      player_missile1_direction=89,
                      player_missile2_direction=90),

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

    "berzerk": dict(player_x=19,
                    player_y=11,
                    player_direction=14,
                    player_missile_x=22,
                    player_missile_y=23,
                    player_missile_direction=21,
                    robot_missile_direction=26,
                    robot_missile_x=29,
                    robot_missile_y=30,
                    num_lives=90,
                    robots_killed_count=91,
                    game_level=92,
                    enemy_evilOtto_x=46,
                    enemy_evilOtto_y=89,
                    enemy_robots_x=range(65, 73),
                    enemy_robots_y=range(56, 65),
                    player_score=range(93, 96)),

    "bowling": dict(ball_x=30,
                    ball_y=41,
                    player_x=29,
                    player_y=40,
                    frame_number_display=36,
                    pin_existence=range(57, 67),
                    score=33),

    "boxing": dict(player_x=32,
                   player_y=34,
                   enemy_x=33,
                   enemy_y=35,
                   enemy_score=19,
                   clock=17,
                   player_score=18),

    "breakout": dict(ball_x=99,
                     ball_y=101,
                     player_x=72,
                     blocks_hit_count=77,
                     block_bit_map=range(30),  # see breakout bitmaps tab
                     score=84),  # 5 for each hit

    "demonattack": dict(level=62,
                        player_x=22,
                        enemy_x1=17,
                        enemy_x2=18,
                        enemy_x3=19,
                        missile_y=21,
                        enemy_y1=69,
                        enemy_y2=70,
                        enemy_y3=71,
                        num_lives=114),

    "freeway": dict(player_y=14,
                    score=103,
                    enemy_car_x=range(108, 118)),  # which lane the car collided with player

    "frostbite": dict(
        top_row_iceflow_x=34,
        second_row_iceflow_x=33,
        third_row_iceflow_x=32,
        fourth_row_iceflow_x=31,
        enemy_bear_x=104,
        num_lives=76,
        igloo_blocks_count=77,  # 255 is none and 15 is all "
        enemy_x=range(84, 88),  # 84  bottom row -   87  top row
        player_x=102,
        player_y=100,
        player_direction=4,
        score=[72, 73, 74]),

    "hero": dict(player_x=27,
                 player_y=31,
                 power_meter=43,
                 room_number=28,
                 level_number=117,
                 dynamite_count=50,
                 score=[56, 57]),



    "montezumarevenge": dict(room_number=3,
                             player_x=42,
                             player_y=43,
                             player_direction=52, # 72:  facing left, 40:  facing left, climbing down ladder/rope 24:  facing left, climbing up ladder/rope 128: facing right 32:  facing right, climbing down ladder/rope, 16: facing right climbing up ladder/rope
                             enemy_skull_x=47,
                             enemy_skull_y=46,
                             key_monster_x=44,
                             key_monster_y=45,
                             level=57,
                             num_lives=58,
                             items_in_inventory_count=61,
                             room_state=62,
                             score_0=19,
                             score_1=20,
                             score_2=21),

    "mspacman": dict(enemy_sue_x=6,
                     enemy_inky_x=7,
                     enemy_pinky_x=8,
                     enemy_blinky_x=9,
                     enemy_sue_y=12,
                     enemy_inky_y=13,
                     enemy_pinky_y=14,
                     enemy_blinky_y=15,
                     player_x=10,
                     player_y=16,
                     fruit_x=11,
                     fruit_y=17,
                     ghosts_count=19,
                     player_direction=56,
                     dots_eaten_count=119,
                     player_score=120,
                     num_lives=123),

    "pitfall": dict(player_x=97,  # 8-148
                    player_y=105,  # 21-86 except for when respawning then 0-255 with confusing wraparound
                    enemy_logs_x=98,  # 0-160
                    enemy_scorpion_x=99,
                    # player_y_on_ladder= 108, # 0-20
                    # player_collided_with_rope= 5, #yes if bit 6 is 1
                    bottom_of_rope_y=18,  # 0-20 varies even when you can't see rope
                    clock_sec=89,
                    clock_min=88
                    ),

    "pong": dict(player_y=51,
                 player_x=46,
                 enemy_y=50,
                 enemy_x=45,
                 ball_x=49,
                 ball_y=54,
                 enemy_score=13,
                 player_score=14),

    "privateeye": dict(player_x=63,
                       player_y=86,
                       room_number=92,
                       clock=[67, 69],
                       player_direction=58,
                       score=[73, 74],
                       dove_x=48,
                       dove_y=39),

    "qbert": dict(player_x=43,
                  player_y=67,
                  player_column=35,
                  red_enemy_column=69,
                  green_enemy_column=105,
                  score=[89, 90, 91], # binary coded decimal score
                  tile_color=[         21,                # row of 1
                                     52,  54,             # row of 2
                                   83,  85,  87,          # row of 3
                                 98, 100, 102, 104,       # row of 4
                                1,  3,   5,   7,  9,      # row of 5
                              32, 34, 36,  38,  40, 42]), # row of 6

    "riverraid": dict(player_x=51,
                      missile_x=117,
                      missile_y=50,
                      fuel_meter_high=55,  # high value displayed
                      fuel_meter_low=56  # low value
                      ),

    "seaquest": dict(enemy_obstacle_x=range(30, 34),
                     player_x=70,
                     player_y=97,
                     diver_or_enemy_missile_x=range(71, 75),
                     player_direction=86,
                     player_missile_direction=87,
                     oxygen_meter_value=102,
                     player_missile_x=103,
                     score=[57, 58],
                     num_lives=59,
                     divers_collected_count=62),

    "skiing": dict(player_x=25,
                   clock_m=104,
                   clock_s=105,
                   clock_ms=106,
                   score=107,
                   object_y=range(87, 94)),  # object_y_1 is y position of whatever topmost object on the screen is

    "spaceinvaders": dict(invaders_left_count=17,
                          player_score=104,
                          num_lives=73,
                          player_x=28,
                          enemies_x=26,
                          missiles_y=9,
                          enemies_y=24),

    "tennis": dict(enemy_x=27,
                   enemy_y=25,
                   enemy_score=70,
                   ball_x=16,
                   ball_y=17,
                   player_x=26,
                   player_y=24,
                   player_score=69),

    "venture": dict(sprite0_y=20,
                    sprite1_y=21,
                    sprite2_y=22,
                    sprite3_y=23,
                    sprite4_y=24,
                    sprite5_y=25,
                    sprite0_x=79,
                    sprite1_x=80,
                    sprite2_x=81,
                    sprite3_x=82,
                    sprite4_x=83,
                    sprite5_x=84,
                    player_x=85,
                    player_y=26,
                    current_room=90,  # The number of the room the player is currently in 0 to 9_
                    num_lives=70,
                    score_1_2=71,
                    score_3_4=72),

    "videopinball": dict(ball_x=67,
                         ball_y=68,
                         player_left_paddle_y=98,
                         player_right_paddle_y=102,
                         score_1=48,
                         score_2=50),

    "yarsrevenge": dict(player_x=32,
                        player_y=31,
                        player_missile_x=38,
                        player_missile_y=37,
                        enemy_x=43,
                        enemy_y=42,
                        enemy_missile_x=47,
                        enemy_missile_y=46)
}

# break up any lists (e.g. dict(clock=[67, 69]) -> dict(clock_0=67, clock_1=69) )
update_dict = {k: {} for k in atari_dict.keys()}

remove_dict = {k: [] for k in atari_dict.keys()}

for game, d in atari_dict.items():
    for k, v in d.items():
        if isinstance(v, range) or isinstance(v, list):
            for i, vi in enumerate(v):
                update_dict[game]["%s_%i" % (k, i)] = vi
            remove_dict[game].append(k)

for k in atari_dict.keys():
    atari_dict[k].update(update_dict[k])
    for rk in remove_dict[k]:
        atari_dict[k].pop(rk)
