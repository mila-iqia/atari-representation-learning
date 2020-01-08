atari_dict = {

    "montezuma_revenge":
        {
            "common": dict(room_number=3,
                           player_x=42,  # 2-151
                           player_y=43,  # 134-252
                           player_direction=52,  # 72 if facing left, 128 if facing right
                           level=57,
                           num_lives=58,
                           items_in_inventory_state=65,
                           score_0=19,
                           score_1=20,
                           score_2=21),
            "room_0": dict(
                jewel_x=44,
                jewel_y=45),
            "room_1": dict(
                skull_x=47,
                skull_y=46,
                key_x=44,
                key_y=45),

            "room_2": dict(
                big_skulls_x=44,
                big_skulls_y=45),

            "room_3": dict(
                big_skulls_x=44,
                big_skulls_y=45),

            "room_4": dict(
                spider_x=44,
                spider_y=45),

            "room_5": dict(
                skull_x=47,
                skull_y=46,
                torch_x=44,
                torch_y=45),

            "room_6": dict(
                sword_x=44,
                sword_y=45),

            "room_7": dict(
                key_x=44,
                key_y=45),

            "room_8": dict(
                key_x=44,
                key_y=45),
            "room_9": dict(
                snake_x=44,
                snake_y=45),

            "room_10": dict(
                jewel_x=44,
                jewel_y=45),

            "room_11": dict(
                snake_x=44,
                snake_y=45),

            "room_12": {},

            "room_13": dict(
                spider_x=44,
                spider_y=45),

            "room_14": dict(
                key_x=44,
                key_y=45),

            "room_15": dict(
                jewel_x=44,
                jewel_y=45),

            "room_16": {},

            "room_17": {},

            "room_18": dict(
                skull_x=47,
                skull_y=46),

            "room_19": dict(
                amulet_x=44,
                amulet_y=45),

            "room_20": dict(
                jewel_x=44,
                jewel_y=45),

            "room_21": dict(
                spider_x=44,
                spider_y=45),

            "room_22": dict(
                snake_x=44,
                snake_y=45),

            "room_23": dict(
                jewel_x=44,
                jewel_y=45)

        },
    "venture": {

        "common": dict(
            current_room=90,  # The number of the room the player is currently in 0 to 9_
            num_lives=70,
            score_1_2=71,
            score_3_4=72,
            level=78),

        "room_8": dict(player_ball_x=85,
                       player_ball_y=26,
                       hallmonster1_y=20,
                       hallmonster2_y=21,
                       hallmonster3_y=22,
                       hallmonster4_y=23,
                       hallmonster5_y=24,
                       hallmonster6_y=25,
                       hallmonster1_x=79,
                       hallmonster2_x=80,
                       hallmonster3_x=81,
                       hallmonster4_x=82,
                       hallmonster5_x=83,
                       hallmonster6_x=84),

        "room_0": dict(ball_x=85,
                       ball_y=26,
                       player_y=20,
                       player_x=79,
                       room0monster1_y=21,
                       room0monster1_x=80,
                       treasure_chest_y=22,
                       treasure_chest_x=81,
                       room0monster2_y=23,
                       room0monster2_x=82,
                       room0monster3_y=24,
                       room0monster3_x=83,
                       big_hallmonster_y=25,  # if 89 is set to 128
                       big_hallmonster5_x=84),

        "room_1": dict(ball_x=85,
                       ball_y=26,
                       player_y=20,
                       player_x=79,
                       room1monster1_y=21,
                       room1monster1_x=80,
                       room1treasure_y=22,
                       room1treasure_x=81,
                       room1monster2_y=23,
                       room1monster2_x=82,
                       room1monster3_y=24,
                       room1monster3_x=83,
                       big_hallmonster_y=25,  # if 89 is set to 128
                       big_hallmonster5_x=84),

        "room_2": dict(ball_x=85,
                       ball_y=26,
                       player_y=20,
                       player_x=79,
                       snake1_y=21,
                       snake1_x=80,
                       apple_treasure_y=22,
                       apple_treasure_x=81,
                       snake2_y=23,
                       snake2_x=82,
                       snake3_y=24,
                       snake3_x=83,
                       big_hallmonster_y=25,  # if 89 is set to 128
                       big_hallmonster5_x=84),

        "room_3": dict(ball_x=85,
                       ball_y=26,
                       player_y=20,
                       player_x=79,
                       top_wall_y=21,
                       top_wall_x=80,
                       bottom_wall_y=25,
                       bottom_wall_x=84,
                       left_wall_x=81,
                       left_wall_y=22,
                       right_wall_x=83,
                       right_wall_y=24,
                       treasure_diamond_y=23,
                       treasure_diamond_x=82,
                       big_hallmonster_y=27,  # if 89 is set to 128
                       big_hallmonster5_x=86)
    }
}
