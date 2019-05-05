atari_dict = {"asteroids": dict(asteroids_y=range(3,21), # 224 means end of list
                            asteroids_x= range(21,39),#$e0 means end of list, stored as hmove/delay/LR (mmmmdddl)
                            #asteroids flags= range(39,56),
                            ship_x=73, # stored as HMove/Delay
                            ship_y=74,	
                            player1_lifesdir= 60, #msb are lives, lsb is direction -> LLLLDDDD
                            player1_score_high=61,
                            player1_score_low=62,                
                            xUFO        = 81,
                            yUFO        = 82,	
                            xShot1      = 83,
                            xShot2      = 84,
                            xShotUfo    = 85,
                            yShot1      = 86,
                            yShot2      = 87,
                            yShotUfo    = 88,
                            dirShot1    = 89,
                            dirShot2    = 90,
                            dirShotUfo  = 91),

                "berzerk" : dict(player_x=19, 
                                player_y =11,
                                player_direction= 14 ,
                                player_missile_x=22,
                                player_missile_y=23  ,
                                player_missile_direction= 21 ,
                                player_missile_flight_time= 20 ,
                                robot_missile_direction= 26 ,
                                robot_missile_x=29,
                                robot_missile_y=30  ,
                                num_lives= 90 ,
                                robots_killed= 91 ,
                                game_level= 92 ,
                                robot_missile_flight_time=28  ,
                                evilOtto_x= 46 ,
                                evilOtto_y= 89 ,
                                robots_x= range(65,73),
                                robots_y=range(56,65), 
                                player_score=range(93,96)),
              
              "boxing" : dict(agent_x = 32, 
                              agent_y=34, 
                              agent_left_arm = 73,
                              agent_right_arm = 75,
                              agent_head_pos=74,
                              enemy_x=33, 
                              enemy_y=35, 
                              enemy_right_arm=77, 
                              enemy_left_arm=79,
                              enemy_head_pos=78,
                              enemy_score=19,
                              clock=17,
                              agent_score=18),												
              
              "breakout" : dict(ball_x = 99,
                                ball_y= 101,
                                paddle_x= 72,
                                num_blocks_hit= 77,
                                block_bit_map=range(30),# see breakout bitmaps tab
                                score= 84), #5 for each hit	
              
              "defender": dict(radar1_y=42,    # vertical position of radar dot
                               radar2_y=44,    # vertical position of radar dot
                               enemyWave=82),# Current wave of enemies"
              
			 "demonattack":	dict(level= 62,
                      CurrentEnemy_y   = 68,
                      small_demon_y = 72,
                      small_demon_x= 111,	
                        #"1st level Only?
                        EnemyLeftHPosition        = 13,  # horizontal position ENEMY1 (enemy left side)
                        EnemyLeftHPosition2        = 14, # horizontal position ENEMY2 (enemy left side)
                        EnemyLeftHPosition3        = 15, # horizontal position ENEMY3 (enemy left side)

                        playerX                        = 16,  # horizontal position PLAYER'S SHIP
                        EnemyRightHPosition        = 17, # horizontal position ENEMY1 (enemy right side)
                        EnemyRightHPosition2        = 18,  # horizontal position ENEMY2 (enemy right side)
                        EnemyRightHPosition3        = 19, # horizontal position ENEMY3 (enemy right side)

                        SmallDemonHPosition        = 20, # horizontal position SINGLE SMALL ATTACKING DEMON
                        ShotVerticalPosition    = 21,  # shot vertical position (BALL)"	


                        enemy_right1_y               = 69, # vertical position ENEMY1 (enemy right side)
                        enemy_right2_y               = 70, # vertical position ENEMY2 (enemy right side)
                        enemy_right3_y = 71, # vertical position ENEMY3 (enemy right side)
                        numberOfLives   = 114,
                        TypeofShot = 118,
                        shot_speed = 110),
              
              
              "enduro" : dict(speed= 22,
                            cars_remaining= [43,44],
                            odometer= range(40,43),
                            colOpponents = range(27,34)),
                              
            "freeway": dict(player1_y=14,
                            player2_y=15,
                            carx_dir= 22,
                            car_motions= range(43,53),
                            automobile_x_coords= range(108,118),
                            temp_x_coord = 118,
                            agent_laneCollide=16), # which lane the car collided with agent
        "frostbite" : dict(                   
                            top_row_iceflow_x = 34,
                            second_row_iceflow_x = 33,
                            third_row_iceflow_x = 32,
                            fourth_row_iceflow_x = 31,
                            top_row_enemy_x = 42,
                            second_row_enemy_x = 41,
                            third_row_enemy_x = 40,
                            fourth_row_enemy_x = 39,
                            top_row_iceflow_col = 46,
                            second_row_iceflow_col = 45,
                            third_row_iceflow_col = 44,
                            fourth_row_iceflow_col = 43,
                            bear_col = 61,
                            bear_x = 104,
                            igloo_entrance_x = 62,   
                            num_lives= 76,   
                            num_igloo_blocks= 77, #255 is none and 15 is all "
                            hPosEnemies            =   range(84,88),   #   84  bottom row -   87  top row
                            enemyNusiz              =   88,
                            player_x=   102, 
                            player_y=   100),
                              
                              
    "hero" :dict(player_x = 27, 
                player_y = 31, 
                powermeter = 43, 
                room_number = 28, 
                level_number = 117),
                                                                                                                                                                                                              
    "montezumarevenge": dict(room_number=3,
                              agent_x=42,
                              agent_y=43,
                              agent_facing_direction=52, #72 if facing left, 128 if facing right
                              skull_x=47,
                              skull_y=46,
                              key_monster_x=44,
                              key_monster_y=45,level=57,
                              lives_count=58,
                              items_possessed=61,
                              room_state=62,
                             score_0 = 19,
                              score_1 = 20,
                              score_2 = 21),
                              
    "mspacman": dict(sue_x =  6,
                        inky_x=  7,
                        pinky_x  =  8,
                        blinky_x =  9,
                        msPac_x  = 10,
                        fruit_x  = 11,
                        sue_y  = 12,
                        inky_y = 13,
                        pinky_y = 14,
                        blinky_y  = 15,
                        msPac_y = 16,
                        fruit_y = 17,
                        numberOfGhosts = 19,
                        playerHorizPosValues= range(28,29),
                        msPacmanCurrectDir=58,
                        energizerStatus = 116,
                        energizerValue = 117,
                        gameBoardStatus = 118,
                        numDotsEaten = 119,
                        playerScore = 120, 
                        numberOfLives = 123),
                              
    "pitfall": dict(agent_x= 97, #8-148      
                    agent_y= 105, # 21-86 except for when respawning then 0-255 with confusing wraparound
                    logs_x = 98, # 0-160     
                    scorpion_x= 99,  
                    player_y_on_ladder= 108, # 0-20  
                    quicksand_x= 116,        
                    player_collided_with_rope= 5, #yes if bit 6 is 1 
                    y_bottom_of_rope= 18, # 0-20 varies even when you can't see rope 
                    number_of_ground_obj=84, 
                    player_leg_position= 100, # range 0-8    
                    jump_trajectory_pos= 103),# 0-32
                              
                              
    "pong": dict(player_y= 51,
                 enemy_y= 50,
                 ball_x=49,
                 ball_y=54,
                 enemy_score=13,
                 player_score= 14),
                              
    "privateeye": dict(agent_x=63,
                        agent_y= 86,
                        room_number= 92,
                        inventory= 60,
                        inventory_history= 72,
                        tasks_completed= 93),
                              
                              
    "qbert" :  dict(num_lives = 26, 
                    player_x = 54,
                    EnemySection1_x = 55, 
                    EnemySection2_x = 56, 
                    EnemySection3_x = 57, 
                    EnemySection4_x = 58, 
                    EnemySection5_x = 59, 
                    EnemySection6_x = 60, 
                    colEnemy1 = 41, 
                    colEnemy2 = 42),
                              
                              
                                                                                                                                                                                                                                                                                                                
    "riverraid" : dict(player_x= 51,
                  player_jet_speed_x= 52,
                  player_jet_speed_y= 53,
                  missile_x = 117,
                  missile_y=50,
                  fuel_high= 55, # high value displayed 
                  fuel_low= 56, #low value     
                  difficulty_level= 61),
                              
                              
    "seaquest" : dict(obstacles_x=range(31,35),
                      player_x= 70, 
                      player_y= 97,
                      torpedo_or_diver_x= range(71,75),
                      player_x_facing_direction=86,
                      torpedo_x_facing_direction= 87,
                      oxygen_level= 102,
                      torpedo_x= 103,
                      enemy_sub_x= 118),
    
    "solaris" : dict(player_x= range(42,45),
                     player_y= range(51,54),
                     photon_z= range(60,62),
                     game_state= 88,
                     lives= 89,
                     fuel=91,
                     score= range(92,95)), 
                              
                              
    "spaceinvaders" : dict(num_invaders_left=17, 
                            Score=   102, 
                            Lives =  125, 
                            PlayerX=   75, 
                            ShipPos=  72 ),
    
    "venture": dict(sprite0_Ypos        = 20,  
                    sprite1_Ypos        = 21,
                    sprite2_Ypos        = 22,
                    sprite3_Ypos        = 23,
                    sprite4_Ypos        = 24,
                    sprite5_Ypos        = 25,
                    sprite0_Xpos        = 79,
                    sprite1_Xpos        = 80,
                    sprite2_Xpos        = 81,
                    sprite3_Xpos        = 82,
                    sprite4_Xpos        = 83,
                    sprite5_Xpos        = 84,      
                    ballXpos = 85,
                    ballYpos = 26,    
                    sprite0_Status            = 28,# The direction and speed of each game object_  The upper nybble is the
                    sprite1_Status            = 29,   # direction, and it corresponds to the values generated by a joystick_
                    sprite2_Status      = 30,   # The 3 lowest order bits hold the object's speed 0 - 7_  0 is immobile        
                    sprite3_Status      = 31,  # like an item_  7 is the quickest speed in the game_
                    sprite4_Status      = 32,
                    sprite5_Status      = 33,
                    ball_Status            = 34,
                    currentRoom         = 90),   # The number of the room the player is currently in 0 to 9_
                              
    "videopinball" : dict(ball_x= 67,
                           ball_y= 68,
                          left_paddle_pos=98,
                          right_paddle_pos = 102,
                          score_1=48,
                          score_2 = 50),
                              
    "yarsrevenge" : dict(yar_x=32, 
                          yar_y=31,
                          yar_missile_x=38,
                          yar_missile_y=37,
                          qotile_x=43,
                          qotile_y=42,
                          qotile_missile_x=47,
                          qotile_missile_y=46,
                          zorlon_cannon_x=25, 
                          zorlon_cannon_y=24,
                          shield_y_pos= 26,
                          shield_y= 26) }
																			

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
    
