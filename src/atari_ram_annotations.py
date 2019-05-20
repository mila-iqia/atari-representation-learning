from copy import deepcopy

atari_dict = {"asteroids": dict(enemy_asteroids_y = [3,4,5,6,7,8,9,12,13,14,15,16,17,18,19],
                                enemy_asteroids_x = [21,22,23,24,25,26,27,30,31,32,33,34,35,36,37],
                                player_x = 73,
                                player_y = 74,	
                                num_lives_direction= 60,
                                player_score_high = 61,
                                player_score_low = 62,                
                                player_missile_x1 = 83,
                                player_missile_x2 = 84,
                                player_missile_y1 = 86,
                                player_missile_y2 = 87,
                                player_missile1_direction = 89,
                                player_missile2_direction = 90),

                "berzerk" : dict(player_x = 19, 
                                player_y = 11,
                                player_direction = 14,
                                player_missile_x = 22,
                                player_missile_y = 23,
                                player_missile_direction = 21,
                                robot_missile_direction = 26,
                                robot_missile_x = 29,
                                robot_missile_y = 30,
                                num_lives = 90,
                                robots_killed_count = 91,
                                game_level = 92,
                                enemy_evilOtto_x = 46,
                                enemy_evilOtto_y = 89,
                                enemy_robots_x = range(65,73),
                                enemy_robots_y = range(56,65), 
                                player_score = range(93,96)),
              
              "boxing" : dict(player_x = 32, 
                              player_y=34, 
                              enemy_x=33, 
                              enemy_y=35, 
                              enemy_score=19,
                              clock=17,
                              player_score=18),												
              
              "breakout" : dict(ball_x = 99,
                                ball_y= 101,
                                player_x= 72,
                                blocks_hit_count = 77,
                                block_bit_map=range(30),# see breakout bitmaps tab
                                score= 84), #5 for each hit	
              
              
			 "demonattack":	dict(level= 62,
                                player_x = 22,  
                                enemy_x1 = 17, 
                                enemy_x2 = 18,  
                                enemy_x3 = 19, 
                                missile_y = 21,  
                                enemy_y1= 69,
                                enemy_y2 = 70, 
                                enemy_y3 = 71, 
                                num_lives = 114),

              
#               "enduro" : dict(speed= 22,
#                             cars_remaining= [43,44],
#                             odometer= range(40,43),
#                             colOpponents = range(27,34)),
                              
            "freeway": dict(player_y=14,
                            score=103,
                            enemy_car_x= range(108,118)), # which lane the car collided with player
        "frostbite" : dict(                   
                            top_row_iceflow_x = 34,
                            second_row_iceflow_x = 33,
                            third_row_iceflow_x = 32,
                            fourth_row_iceflow_x = 31,
                            enemy_bear_x = 104, 
                            num_lives= 76,   
                            igloo_blocks_count= 77, #255 is none and 15 is all "
                            enemy_x           =   range(84,88),   #   84  bottom row -   87  top row
                            player_x=   102, 
                            player_y=   100,
                            player_direction = 4,
                            score = [72,73,74]),
                              
                              
    "hero" :dict(player_x = 27, 
                player_y = 31, 
                power_meter = 43, 
                room_number = 28, 
                level_number = 117,
                 dynamite_count=50, 
                 score=[56,57]),
                                                                                                                                                                                                              
    "montezumarevenge": dict(room_number=3,
                              player_x=42,
                              player_y=43,
                              player_direction=52, #72 if facing left, 128 if facing right
                              enemy_skull_x=47,
                              enemy_skull_y=46,
                              key_monster_x=44,
                              key_monster_y=45,
                              level=57,
                              num_lives=58,
                              items_in_inventory_count=61,
                              room_state=62,
                              score_0 = 19,
                              score_1 = 20,
                              score_2 = 21),
                              
    "mspacman": dict(sue_x =  6,
                        enemy_inky_x=  7,
                        enemy_pinky_x  =  8,
                        enemy_blinky_x =  9,
                        enemy_sue_y  = 12,
                        enemy_inky_y = 13,
                        enemy_pinky_y = 14,
                        enemy_blinky_y  = 15,
                        player_x  = 10,
                        player_y = 16,
                        fruit_x  = 11,
                        fruit_y = 17,
                        ghosts_count = 19,                     
                        player_direction=56,
                        dots_eaten_count = 119,
                        player_score = 120, 
                        num_lives = 123),
                              
    "pitfall": dict(player_x= 97, #8-148      
                    player_y= 105, # 21-86 except for when respawning then 0-255 with confusing wraparound
                    enemy_logs_x = 98, # 0-160     
                    enemy_scorpion_x= 99,  
                    #player_y_on_ladder= 108, # 0-20        
                    #player_collided_with_rope= 5, #yes if bit 6 is 1 
                   bottom_of_rope_y= 18, # 0-20 varies even when you can't see rope
                   ),
                              
                              
    "pong": dict(player_y= 51,
                 enemy_y= 50,
                 ball_x=49,
                 ball_y=54,
                 enemy_score=13,
                 player_score= 14),
                              
    "privateeye": dict(agent_x= 63,
                      agent_y= 86,
                      room_number= 92,
                      clock=[67,69],
                      player_direction=58,
                      score=[73,74],
                      dove_x=48,
                      dove_y=39),
              

                              
                              
    "qbert" :  dict( player_x=43, 
                    player_y = 67, 
                    player_column=35,
                    red_enemy_column=69,
                    green_enemy_column=105),
                              
                              
                                                                                                                                                                                                                                                                                                                
    "riverraid" : dict(player_x= 51,
                  missile_x = 117,
                  missile_y=50,
                  fuel_meter_high= 55, # high value displayed 
                  fuel_meter_low= 56 #low value     
                                        ),
                              
                              
    "seaquest" : dict(enemy_obstacle_x=range(30,34),
                      player_x= 70, 
                      player_y= 97,
                      diver_x= range(71,75),
                      player_direction=86,
                      missile_direction= 87,
                      oxygen_meter_value= 102,
                      missile_x= 103,
                      score=[57,58],
                      num_lives=59,
                      divers_collected_count=62),
    
#     "solaris" : dict(player_x= range(42,45),
#                      player_y= range(51,54),
#                      photon_z= range(60,62),
#                      game_state= 88,
#                      num_lives= 89,
#                      fuel_meter=91,
#                      score= range(92,95)), 
                              
                              
    "spaceinvaders" : dict(invaders_left_count=17, 
                            player_score=   104, 
                            num_lives =  125, 
                            player_x =   28,
                            enemies_x = 26,
                            missiles_y = 9,
                            enemies_y = 24 ),
              
    "tennis": dict(enemy_x = 27,
                   enemy_y=25,
                   enemy_score=70,
                   ball_x =16, 
                   ball_y=17,
                   player_x = 26, 
                   player_y = 24, 
                   player_score = 69),	
    
    "venture": dict(sprite0_y        = 20,  
                    sprite1_y        = 21,
                    sprite2_y        = 22,
                    sprite3_y        = 23,
                    sprite4_y        = 24,
                    sprite5_y        = 25,
                    sprite0_x        = 79,
                    sprite1_x        = 80,
                    sprite2_x        = 81,
                    sprite3_x        = 82,
                    sprite4_x        = 83,
                    sprite5_x        = 84,      
                    player_x = 85,
                    player_y = 26,    
                    current_room         = 90,  # The number of the room the player is currently in 0 to 9_
                    num_lives = 70,
                    score_1_2 = 71, 
                    score_3_4 = 72), 
              
                              
    "videopinball" : dict(ball_x= 67,
                          ball_y= 68,
                          player_left_paddle_y =98,
                          player_right_paddle_y = 102,
                          score_1=48,
                          score_2 = 50),
                              
    "yarsrevenge" : dict(player_yar_x=32, 
                          player_yar_y=31,
                          yar_missile_x=38,
                          yar_missile_y=37,
                          enemy_qotile_x=43,
                          enemy_qotile_y=42,
                          qotile_missile_x=47,
                          qotile_missile_y=46) }
																			

update_dict = {k:{} for k in atari_dict.keys()}

remove_dict = {k:[] for k in atari_dict.keys()}

for game, d in atari_dict.items():
    for k,v in d.items():
        if isinstance(v,range) or isinstance(v,list):
            for i, vi in enumerate(v):
                update_dict[game]["%s_%i"%(k,i)] = vi
            remove_dict[game].append(k)

for k in atari_dict.keys():
    atari_dict[k].update(update_dict[k])
    for rk in remove_dict[k]:
        atari_dict[k].pop(rk)
        

from copy import deepcopy

list_of_keys = [list(v.keys()) for v in atari_dict.values()]
all_keys = []
for a in list_of_keys:
    all_keys.extend(a)

small_object_names = [ "ball", "missile"]
agent_names = ["agent", "player"]
localization_keys = [k for k in all_keys if any(coord in k for coord in ["_x","_y","_z","_column"])]
agent_localization_keys = [k for k in localization_keys if any(agent_name in k for agent_name in agent_names) and not any(small_object_name in k for small_object_name in small_object_names)]
small_object_localization_keys = [k for k in localization_keys if any(small_object_name in k for small_object_name in small_object_names) ]
other_localization_keys = [k for k in localization_keys if k not in agent_localization_keys + small_object_localization_keys]

score_keys = [k for k in all_keys if "score" in k]
clock_keys = [k for k in all_keys if "clock" in k]
lives_keys = [k for k in all_keys if "lives" in k or "lifes" in k ]
count_keys = [k for k in all_keys if "count" in k]
meter_keys = [k for k in all_keys if "meter" in k in k]
existence_keys = [k for k in all_keys if "bit_map" in k]
score_clock_lives_keys = score_keys + clock_keys + lives_keys 
direction_keys = [k for k in all_keys if "direction" in k]
level_room_keys = [k for k in all_keys if "level" in k or "room" in k or "game_state" in k]
misc_keys = count_keys + meter_keys + existence_keys + level_room_keys + direction_keys


unused_keys = deepcopy(all_keys)


summary_key_dict = dict(small_object_localization=small_object_localization_keys,
                        agent_localization=agent_localization_keys,
                        other_localization=other_localization_keys,
                        score_clock_lives_display = score_clock_lives_keys,
                        misc_keys = misc_keys 
                        )   
    

detailed_key_dict = dict(overall=all_keys,
                         agent_localization=agent_localization_keys,
                            small_object_localization=small_object_localization_keys,
                            other_localization=other_localization_keys,
                            score=score_keys,
                            clock=clock_keys,
                            lives=lives_keys,
                            count=count_keys,
                            meter=meter_keys,
                            existence=existence_keys,
                            direction=direction_keys,
                            level_room=level_room_keys)