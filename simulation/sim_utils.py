import math
from scipy.spatial.transform import Rotation as R
import numpy as np

# positive angle is to the right / clockwise
def spawn_point(default_scenario, road_id, reverse=False, seg=1):
    global lanewidth
    if default_scenario == 'cliff':
        return {'pos': (-124.806, 190.554, 465.489), 'rot': None, 'rot_quat': (0, 0, 0.3826834, 0.9238795)}
    elif default_scenario == 'west_coast_usa':
        if road_id == "13242": # surface road (crashes early af)
            return {'pos': (-733.7, -923.8, 163.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, 0.805, 0.592), -20)}
        elif road_id == "8650":
            if reverse:
                return {'pos': (-358.719,-846.965,136.99), 'rot': None, 'rot_quat': turn_X_degrees((-0.008,-0.0144,0.3291,0.9442), 7)}
            else: # yellow lanelines, left curve, driving on left
                return {'pos': (-365.24, -854.45, 136.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 90)}
        elif road_id == "12667": # lanelines, uphill left curve
            return {'pos': (-892.4, -793.4, 114.1), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 70)}
        elif road_id == "8432": # lanelines, wide 2direction highway, long left curve
            if reverse:
                return {'pos': (-871.9, -803.2, 115.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 165 - 180)}
            else:
                return {'pos': (-390.4, -799.1, 139.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 165)}
        elif road_id == "8518":
            if reverse: # up the hill
                return {'pos': (-913.2, -829.6, 118.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 45)}
            else: # starts on right turn, rock walls surrounding road, lanelines
                return {'pos': (-390.5, -896.6, 138.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 20)}
        elif road_id == "8417":
            if reverse: # road surrounding suburb, big intersection
                # return {"pos": (-282.82, -875.592, 134.9), "rot": None, "rot_quat": turn_X_degrees((0, 0, -0.278, 0.961), 155)}
                # past intersection
                # return {"pos": (-306.01910400390625, -862.2595825195312, 135.1), "rot": None, "rot_quat": turn_X_degrees((0, 0, -0.278, 0.961), 155)}
                # return {"pos": (-320.4444885253906, -853.4628295898438, 135.6), "rot": None, "rot_quat": turn_X_degrees((0, 0, -0.278, 0.961), 155)}
                return {"pos": (-348.3262939453125, -831.0289916992188, 137.0), "rot": None, "rot_quat": turn_X_degrees((0, 0, -0.278, 0.961), 165)}
            else: # road surrounding suburb, starts on left side of road
                return {'pos': (-402.7, -780.2, 141.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 0)}
        # elif road_id == "8703":
        #     if reverse:
        #         # return {'pos': (-312.4, -856.8, 135.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 155)}
        #         return {'pos': (-327.926,-846.741,136.099), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 155)}
        #     return {'pos': (-307.8, -784.9, 137.6), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 80)}
        elif road_id == "12641":
            # if reverse:
            #     return {'pos': (-964.2, 882.8, 75.1), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 0)}
            return {'pos': (-366.1753845214844, 632.2236938476562, 75.1), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 170)}
        elif road_id == "13091":
            if reverse:
                return {'pos': (-903.6078491210938, -586.33154296875, 106.6), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 80)}
            return {'pos': (-331.0728759765625, -697.2451782226562, 133.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 0)}
        # elif road_id == "11602":
        #     return {'pos': (-366.4, -858.8, 136.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 0)}
        elif road_id == "12146":
            if reverse:
                return {'pos': (995.7, -855.0, 167.1), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -15)}
            else:
                return {'pos': (-391.0, -798.8, 139.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -15)}
        elif road_id == "13228":
            return {'pos': (-591.5175170898438, -453.1298828125, 114.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 0)}
        elif road_id == "13155":  # middle laneline on highway #12492 11930, 10368 is an edge
            return {'pos': (-390.7796936035156, -36.612098693847656, 109.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -90)}
        elif road_id == "10784":  # track # 13228  suburb edge, 12939 10371 12098 edge
            if reverse:
                return {'pos': (800.905,350.394,156.297), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -135)}
            # return {'pos': (57.05, -150.53, 125.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -115)}
            # return {'pos': (86.1454,-118.969,127.519), 'rot': None, 'rot_quat': (-0.03234875574708,0.022467797622085,-0.8322811126709,0.55295306444168)}
            return {'pos': (144.962, -96.1268, 128.935), 'rot': None, 'rot_quat': (-0.032797202467918, 0.024726673960686, -0.80050182342529, 0.59792125225067)}
        elif road_id == "10673":
            return {'pos': (-21.7, -826.2, 133.1), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -90)}
        elif road_id == "12930":  # 13492 dirt road
            # return {'pos': (-347.2, -824.7, 137.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -100)}
            return {'pos': (-353.731,-830.905,137.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -100)}
        elif road_id == "10988": # track
            # return {'pos': (622.2, -251.1, 147.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -60)}
            # return {'pos': (660.388,-247.67,147.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -160)}
            if seg == 0: # straight portion
                return {'pos': (687.5048828125, -185.7435302734375, 146.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -100)}
            elif seg == 1: # approaching winding portion
                #  crashes around [846.0238647460938, 127.84288787841797, 150.64915466308594]
                # return {'pos': (768.1991577148438, -108.50184631347656, 146.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -100)}
                # return {'pos': (781.2423095703125, -95.72360229492188, 147.4), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -100)}
                return {'pos': (781.2423095703125, -95.72360229492188, 147.4), 'rot': None,'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -105)}
                return {'pos': (790.599,-86.7973,147.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -100)} # slightly better?
            elif seg == 2:
                return {'pos': (854.4083862304688, 136.79324340820312, 152.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -100)}
            elif seg == 3:
                return {'pos': (608.651,-251.865,147.4), 'rot': None, 'rot_quat': (0.0013050610432401,0.0056851170957088,-0.72551721334457,0.68817931413651)}
            else:
                return {'pos': (599.341, -252.333, 147.6), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -60)}
        elif road_id == "13306":
            return {'pos': (-310, -790.0, 137.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), -15)}
        elif road_id == "13341":
            return {'pos': (-393.4, -34.0, 109.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.278, 0.961), 90)}
        elif road_id == "8483": # parking lot aisle
            return {'pos': (203.2150421142578, -389.3501892089844, 143.99), 'rot': None, 'rot_quat': turn_X_degrees((0., 0., 0., 1.0), 0)}
        elif road_id == "8510":
            return {'pos': (1004.8037109375, -815.4000244140625, 166.9), 'rot': None, 'rot_quat': turn_X_degrees((0., 0., 0., 1.0), 0)}
        elif road_id == "8719": # dock area left hand side of road
            return {'pos': (-432.8963623046875, 681.6007080078125, 74.84849548339844), 'rot': None, 'rot_quat': turn_X_degrees((0., 0., 0., 1.0), 140)}
        elif road_id == "10551":
            return {'pos': (-88.7691650390625, 375.8317565917969, 101.8), 'rot': None, 'rot_quat': turn_X_degrees((0., 0., 0., 1.0), 0)}
        elif road_id == "11297":
            return {'pos': (-24.706754684448242, 494.89447021484375, 74.5791702270), 'rot': None, 'rot_quat': turn_X_degrees((0., 0., 0., 1.0), 0)}
        elif road_id == "8576": # too short, good nearby roads though
            return {'pos': (-837.0810546875, -565.4932250976562, 99.7087), 'rot': None, 'rot_quat': turn_X_degrees((0., 0., 0., 1.0), 0)}
        elif road_id == "8418":
            return {'pos': (507.22747802734375, 657.4219360351562, 124.9), 'rot': None, 'rot_quat': turn_X_degrees((0., 0., 0., 1.0), -20)}
        elif road_id == "8409":
            return {'pos': (-234.9193572998047, -190.3572540283203, 119.38419342041), 'rot': None, 'rot_quat': turn_X_degrees((0., 0., 0., 1.0), 0)}
        elif road_id == "8668":
            return {'pos': (-720.2305908203125, 859.3646240234375, 74.822509765), 'rot': None, 'rot_quat': turn_X_degrees((0., 0., 0., 1.0), 0)}
        elif road_id == "8455":
            return {'pos': (-366.32904052734375, -493.8239440917969, 107.78089141845), 'rot': None, 'rot_quat': turn_X_degrees((0., 0., 0., 1.0), 0)}
        elif road_id == "10495":
            return {'pos': (-167.48072814941406, 506.37420654296875, 74.84955596923), 'rot': None, 'rot_quat': turn_X_degrees((0., 0., 0., 1.0), 0)}
        elif road_id == "8714":
            return {'pos': (199.6, 820.4, 102.2), 'rot': None, 'rot_quat': turn_X_degrees((0., 0., 0., 1.0), 0)}
        # todo: check other west_coast_usa poi roads
        elif road_id == "8512":
            return {'pos': (-895.3,-400.5,101.4), 'rot': None, 'rot_quat': turn_X_degrees((0., 0., 0., 1.0), -50)}
        elif road_id == "13349": # highway under underpasses
            return {'pos': (-897.6122436523438,-399.5248107910156,101.3), 'rot': None, 'rot_quat': turn_X_degrees((0., 0., 0., 1.0), -55)}
        elif road_id == "10378": # road edge
            return {'pos': (-899.7344970703125,-781.9215698242188,113.47029876), 'rot': None, 'rot_quat': turn_X_degrees((0., 0., 0., 1.0), 0)}
        elif road_id == "12930":
            return {'pos': (-347.16302490234375,-824.6746215820312,137.0292816162), 'rot': None, 'rot_quat': turn_X_degrees((0., 0., 0., 1.0), -7)}
        elif road_id == "11302": # road edge
            return {'pos': (-952.902587890625,-656.427001953125,106.37081909179688), 'rot': None, 'rot_quat': turn_X_degrees((0., 0., 0., 1.0), 0)}
        elif road_id == "11635":
            return {'pos': (-314.78973388671875,-481.28790283203125,107.0), 'rot': None, 'rot_quat': turn_X_degrees((0., 0., 0., 1.0), 154)}
        elif road_id == "10944": # road edge
            return {'pos': (-706.712158203125,-387.5685119628906,106.1199340820312), 'rot': None, 'rot_quat': turn_X_degrees((0., 0., 0., 1.0), 0)}
        elif road_id == "10939": # dirt road
            return {'pos': (-856.2081909179688,-415.33544921875,98.504989624023), 'rot': None, 'rot_quat': turn_X_degrees((0., 0., 0., 1.0), 0)}
        elif road_id == "11025":
            return {'pos': (-802.132080078125,-381.5849609375,100.934997558), 'rot': None, 'rot_quat': turn_X_degrees((0., 0., 0., 1.0), 0)}
        elif road_id == "ai_path":
            return {'pos': (-809.6217041015625, 11.88548755645752, 117.4980239868164), 'rot': None, 'rot_quat': turn_X_degrees((0., 0., 0., 1.0), 0)}
        # elif road_id == "":
        #     return {'pos': (), 'rot': None, 'rot_quat': turn_X_degrees((0., 0., 0., 1.0), 0)}
        # elif road_id == "":
        #     return {'pos': (), 'rot': None, 'rot_quat': turn_X_degrees((0., 0., 0., 1.0), 0)}
    elif default_scenario == 'smallgrid':
        return {'pos':(0.0, 0.0, 0.0), 'rot':None, 'rot_quat':(0, 0, 0.3826834, 0.9238795)}
    elif default_scenario == 'automation_test_track':
        if road_id == 'startingline':
            return {'pos': (487.25, 178.73, 131.928), 'rot': None, 'rot_quat': (0, 0, -0.702719, 0.711467)}
        elif road_id == "7990" or  road_id == "7991": # immediately crashes into guardrail
            return {'pos': (57.229, 360.560, 128.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.702719, 0.711467), 180)}
        elif road_id == "7846": # immediately leaves track
            return {'pos': (-456.0, -100.3, 117.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.702719, 0.711467), 180)}
        elif road_id == "7811":
            return  {'pos': (-146.2, -255.5, 119.95), 'rot': None, 'rot_quat': turn_X_degrees((-0.021, -0.009, 0.740, 0.672), 180)}
        elif road_id == "8185": # good for saliency testing
            return {'pos': (174.92, -289.7, 120.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.702719, 0.711467), 180)}
            # return {'pos': (-180.4, -253.0, 120.7), 'rot': None, 'rot_quat': (-0.008, 0.004, 0.779, 0.63)}
            return {'pos': (-58.2675, -255.216, 120.175), 'rot': None, 'rot_quat': (-0.021, -0.009, 0.740, 0.672)}
        elif road_id == "8293": # immediately leaves track
            return {'pos': (-556.185, 386.985, 145.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.702719, 0.7115), 120)}
        elif road_id == "8341": # dirt road
            return {'pos': (775.5, -2.2, 132.6), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.702719, 0.711467), 90)}
        # elif road_id == "8287": # road edge
        #     return {'pos': (-198.8, -251.0, 119.8), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.702719, 0.711467), 0)}
        # elif road_id == "7998": # road edge
        #     return {'pos': (-162.6, 108.8, 122.1), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.702719, 0.711467), 60)}
        elif road_id == "8356" or road_id == "8357": # mountain road, immediately crashes into guardrail
            return {'pos': (-450.45, 679.2, 249.45), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 150)}
        elif road_id == "7770": # road edge
            return {'pos': (-443.42, 61.7, 118), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 0)}
        elif road_id == "8000":  # same as 7909
            return {'pos': (-49.1, 223, 127), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), -145)}
        elif road_id == "7905": # dirt road, immediately leaves track
            return {'pos': (768.2, 452.04, 145.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), -110)}
        elif road_id == "8205":
            return {'pos': (501.4, 178.6, 131.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 0)}
        elif road_id == "8353":
            return {'pos': (887.2, 359.8, 159.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), -20)}
        elif road_id == "7882":
            return {'pos': (-546.8, 568.0, 199.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 155)}
        elif road_id == "8179":
            return {'pos': (-738.1, 257.3, 133.4), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 150)}
        #START TESTING NEW ROADS HERE vvvv
        # elif road_id == "8286": # roundabout only
        #     return {"pos": (-137.96646118164062,-27.3629093170166,116.99319458007812), 'rot':None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 0)}
        elif road_id == "8038": #winding service road
            return {"pos": (-487.9,35.3,117.2), 'rot':None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), -30)}
        # elif road_id == "7726":
        #     return {"pos": (-162.8,-103.3,117.6), 'rot':None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 0)}
        # elif road_id == "8339": # road edge
        #     return {"pos": (-550.934326171875,402.41839599609375,146.0393829345703), 'rot':None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 0)}
        # elif road_id == "7927": # road edge
        #     return {"pos": (-432.21502685546875,0.49254098534584045,119.20594787597656), 'rot':None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 0)}
        elif road_id == "7882": # winding mountain road with double line
            return {"pos": (-531.7635498046875,575.9854125976562,199.79685974121094), 'rot':None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 0)}
        # elif road_id == "8372": # road edge
        #     return {"pos": (-303.7345275878906,6.419606685638428,118.55500793457031), 'rot':None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 0)}
        elif road_id == "8290":
            return {"pos": (-630.625,345.876,139.162), 'rot': None, 'rot_quat': (-0.012557459063828,-0.036273058503866,0.92246723175049,-0.3841624557972)}
            return {"pos": (-621.3233032226562,354.85296630859375,139.67283630371094), 'rot':None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), -45)}
        # elif road_id == "8227": #road edge
        #     return {"pos": (-120.01959991455078,475.173095703125,134.49951171875), 'rot':None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 0)}
        # elif road_id == "8018": # road edge-8906,-139.7559051513672,119.69510650634766), 'rot':None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 0)}
        # elif road_id == "7770":
        #     return {"pos": (-453.42, 61.66, 117.5), 'rot':None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 0)}
        elif road_id == "8330":
            return {"pos": (-7.9, 190.61, 126.5), 'rot':None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 40)}
        elif road_id == "8396":
            return {"pos": (585.77, 64.86, 134.2), 'rot':None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), -90)}
        elif road_id == "7804": # right turn
            return {"pos": (330.568, 118.032, 134.2), 'rot':None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 180)}
        elif road_id == "7736": # straight road
            return {"pos": (-679.6494140625, -36.56947326660156, 117.21), 'rot':None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), -25)}
        elif road_id == "7776": # straight to hard left turn
            return {"pos": (249.07, 135.40, 135.1), 'rot':None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), -90)}
        elif road_id == "7909": # same as 8080
            return {"pos": (359.25079345703125, 65.21964263916016, 134.05052185058594), 'rot':None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 0)}
        # elif road_id == "8248": # road edge
        #     return {'pos': (-298.8, 13.6, 118.4), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 180)}
        # elif road_id == "7768": # road edge
        #     return {'pos': (-298.8, 13.6, 118.4), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 180)}
        # elif road_id == "7807": # road edge
        #     return {'pos': (-251.2, -260.0, 119.2), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), 0)}
        # elif road_id == "8049":  # road edge
        #     return {'pos': (-405.0, -26.6, 117.4), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.703, 0.711), -110)}
        elif road_id == 'starting line 30m down':
            return {'pos': (530.25, 178.73, 131.928), 'rot': None, 'rot_quat': (0, 0, -0.702719, 0.711467)}
        elif road_id == 'handlingcircuit':
            # handling circuit
            return {'pos': (-294.031, 10.4074, 118.518), 'rot': None, 'rot_quat': (0, 0, 0.708103, 0.706109)}
        elif road_id == 'handlingcircuit2':
            return {'pos': (-280.704, -25.4946, 118.794), 'rot': None, 'rot_quat': (-0.00862686, 0.0063203, 0.98271, 0.184842)}
        elif road_id == 'handlingcircuit3':
            return {'pos': (-214.929, 61.2237, 118.593), 'rot': None, 'rot_quat': (-0.00947676, -0.00484788, -0.486675, 0.873518)}
        elif road_id == 'handlingcircuit4':
            # return {'pos': (-180.663, 117.091, 117.654), 'rot': None, 'rot_quat': (0.0227101, -0.00198367, 0.520494, 0.853561)}
            # return {'pos': (-171.183,147.699,117.438), 'rot': None, 'rot_quat': (0.001710215350613,-0.039731655269861,0.99312973022461,-0.11005393415689)}
            return {'pos': (-173.009,137.433,116.701), 'rot': None,'rot_quat': (0.0227101, -0.00198367, 0.520494, 0.853561)}
            return {'pos': (-166.679, 146.758, 117.68), 'rot': None,'rot_quat': (0.075107827782631, -0.050610285252333, 0.99587279558182, 0.0058960365131497)}
        elif road_id == 'rally track':
            return {'pos': (-374.835, 84.8178, 115.084), 'rot': None, 'rot_quat': (0, 0, 0.718422, 0.695607)}
        elif road_id == 'highway': #(open, farm-like)
            return {'pos': (-294.791, -255.693, 118.703), 'rot': None, 'rot_quat': (0, 0, -0.704635, 0.70957)}
        elif road_id == 'highwayopp': # (open, farm-like)
            return {'pos': (-542.719,-251.721,117.083), 'rot': None, 'rot_quat': (0.0099,0.0096,0.7215,0.6923)}
        elif road_id == 'default':
            return {'pos': (487.25, 178.73, 131.928), 'rot': None, 'rot_quat': (0, 0, -0.702719, 0.711467)}
    elif default_scenario == 'industrial':
        if road_id == 'west':
            # western industrial area -- didnt work with AI Driver
            return {'pos': (237.131, -379.919, 34.5561), 'rot': None, 'rot_quat': (-0.035, -0.0181, 0.949, 0.314)}
        # open industrial area -- didnt work with AI Driver
        # drift course (dirt and paved)
        elif road_id == 'driftcourse':
            return {'pos': (20.572, 161.438, 44.2149), 'rot': None, 'rot_quat': (-0.003, -0.005, -0.636, 0.771)}
        elif road_id == 'rallycross': # rallycross course/default
            return {'pos': (4.85287, 160.992, 44.2151), 'rot': None, 'rot_quat': (-0.0032, 0.003, 0.763, 0.646)}
        elif road_id == 'racetrackright':
            return {'pos': (184.983, -41.0821, 42.7761), 'rot': None, 'rot_quat': (-0.005, 0.001, 0.299, 0.954)}
        elif road_id == 'racetrackleft':
            return {'pos': (216.578, -28.1725, 42.7788), 'rot': None, 'rot_quat': (-0.0051, -0.003147, -0.67135, 0.74112)}
        elif road_id == 'racetrackstartinggate' or road_id == "7983" or road_id == "7982":
            return {'pos':(160.905, -91.9654, 42.8511), 'rot': None, 'rot_quat':(-0.0036226876545697, 0.0065293218940496, 0.92344760894775, -0.38365218043327)}
        elif road_id == "rc_asphalta":
            return {'pos': (-68.78999328613281,113.09487915039062,43.5), 'rot': None, 'rot_quat':turn_X_degrees((-0.00362, 0.006529, 0.92345, -0.38365), -45)}
        elif road_id == "7978": #narrow patched service road
            return {'pos': (95.4,3.2,42.7), 'rot': None, 'rot_quat': turn_X_degrees((-0.0036226876545697, 0.0065293218940496, 0.92344760894775, -0.38365218043327), 140)}
        elif road_id == "8067": # dirt road
            return {'pos': (-139.33372497558594, -342.8045654296875, 36.7), 'rot': None, 'rot_quat': turn_X_degrees((-0.0036226876545697, 0.0065293218940496, 0.92344760894775, -0.38365218043327), 180)}
        elif road_id == "racetrackstraightaway":
            return {'pos':(262.328, -35.933, 42.5965), 'rot': None, 'rot_quat':(-0.010505940765142, 0.029969356954098, -0.44812294840813, 0.89340770244598)}
        elif road_id == "racetrackcurves":
            return {'pos':(215.912,-243.067,45.8604), 'rot': None, 'rot_quat':(0.029027424752712,0.022241719067097,0.98601061105728,0.16262225806713)}
        elif road_id == "rc_asphaltb":
            return {'pos':(-37.43122482299805,101.19683074951172,42.6), 'rot': None, 'rot_quat':turn_X_degrees((0.0,0.0,0.0,1.0), -90)}
        elif road_id == "rc_asphaltc":
            return {'pos':(-2.0535049438476562,56.74980545043945,42.56134033203125), 'rot': None, 'rot_quat':turn_X_degrees((0.0,0.0,0.0,1.0), -90)}
        elif road_id == "rc_asphaltd":
            return {'pos':(110.4,21.2,43.4), 'rot': None, 'rot_quat':turn_X_degrees((0.0,0.0,0.0,1.0), -90)}
        elif road_id == "7978":
            return {'pos':(95.38813781738281,3.2,42.5), 'rot': None, 'rot_quat':turn_X_degrees((0.0,0.0,0.0,1.0), 0)}
        # elif road_id == "8023": # dirt road
        #     return {'pos':(-67.13399505615234,-301.4559020996094,38.1), 'rot': None, 'rot_quat':turn_X_degrees((0.0,0.0,0.0,1.0), 0)}
        elif road_id == "8022": # white pavement
            return {'pos':(5.042698860168457,-324.50885009765625,34.09881591796875), 'rot': None, 'rot_quat':turn_X_degrees((0.0,0.0,0.0,1.0), -55)}
        elif road_id == "7989":  # dirt road
            return {'pos':(-51.59226989746094,-71.90184783935547,42.68646240234375), 'rot': None, 'rot_quat':turn_X_degrees((0.0,0.0,0.0,1.0), 0)}
        elif road_id == "7978":
            return {'pos':(95.38813781738281,3.2,42.529296875), 'rot': None, 'rot_quat':turn_X_degrees((0.0,0.0,0.0,1.0), 0)}
        elif road_id == "8068":
            return {'pos':(177.36476135253906,-371.0460510253906,34.09881591796875), 'rot': None, 'rot_quat':turn_X_degrees((0.0,0.0,0.0,1.0), -60)}
        elif road_id == "8028": # similar to 7978
            # return {'pos':(-4.2,-329.8,34.1), 'rot': None, 'rot_quat':turn_X_degrees((0.0,0.0,0.0,1.0), -105)}
            return {'pos': (8.1,-328.3,34.7), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), -60)}
        elif road_id == "8079":
            return {'pos':(95.3,2.1,42.5), 'rot': None, 'rot_quat':turn_X_degrees((0.0,0.0,0.0,1.0), 0)}
        elif road_id == "8078":
            return {'pos':(177.36476135253906,-371.0460510253906,34.09881591796875), 'rot': None, 'rot_quat':turn_X_degrees((0.0,0.0,0.0,1.0), 0)}
        elif road_id == "rc_dirtc":
            return {'pos':(238.2580108642578,110.55962371826172,43.604042053222656), 'rot': None, 'rot_quat':turn_X_degrees((0.0,0.0,0.0,1.0), 0)}
        elif road_id == "rc_asphalte": # too short
            return {'pos':(94.68978118896484,105.68899536132812,42.3736572265625), 'rot': None, 'rot_quat':turn_X_degrees((0.0,0.0,0.0,1.0), 0)}
        # elif road_id == "7977": # dirt road
        #     return {'pos':(45.62337112426758,-116.86038970947266,42.67439270019531), 'rot': None, 'rot_quat':turn_X_degrees((0.0,0.0,0.0,1.0), 0)}
        elif road_id == "8008": # too short
            return {'pos':(226.1936492919922,-24.210155487060547,42.648006439208984), 'rot': None, 'rot_quat':turn_X_degrees((0.0,0.0,0.0,1.0), 0)}
        elif road_id == "8009":
            return {'pos':(199.23269653320312,-24.431718826293945,42.56134033203125), 'rot': None, 'rot_quat':turn_X_degrees((0.0,0.0,0.0,1.0), 0)}

    elif default_scenario == "hirochi_raceway":
        # road edges: 9297, 9327, 9286, 9327, 9286, 9266, 9226, 9192, 9356, 9357...
        # dirt roads: 9307, 9225 9047 9235 9166 9168 9060
        if road_id == "9039": # good candidate for input rect.
            if seg == 0: # start of track, right turn; 183m; cutoff at (412.079,-191.549,38.2418)
                return {'pos': (289.327,-281.458, 46.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -130)}
            elif seg == 1: # straight road
                return {'pos': (330.3320007324219, -217.5743408203125, 45.7054443359375), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -130)}
            elif seg == 2: # left turn
                # return {'pos': (439.0, -178.4, 35.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -85)}
                # return {'pos': (448.1, -174.6, 34.6), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -85)}
                return {'pos': (496.2, -150.6, 35.6), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -85)}
            elif seg == 3:
                return {'pos': (538.2, -124.3, 40.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -110)}
            elif seg == 4: # straight; cutoff at vec3(596.333,18.7362,45.6584)
                return {'pos': (561.7396240234375, -76.91995239257812, 44.7), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -130)}
            elif seg == 5: # left turn; cutoff at (547.15234375, 115.24089050292969, 35.97171401977539)
                return {'pos': (598.3154907226562, 40.60638427734375, 43.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -147)}
            elif seg == 6:
                return {'pos': (547.15234375, 115.24089050292969, 36.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -225)}
            elif seg == 7:
                return {'pos': (449.7561340332031, 114.96491241455078, 25.801856994628906), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -225)}
            elif seg == 8: # mostly straight, good behavior; cutoff at  vec3(305.115,304.196,38.4392)
                return {'pos': (405.81732177734375, 121.84907531738281, 25.04170036315918), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -190)}
            elif seg == 9:
                return {'pos': (291.171875, 321.78662109375, 38.6), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -190)}
            elif seg == 10:
                return {'pos': (216.40045166015625, 367.1772155761719, 35.99), 'rot': None, 'rot_quat': (-0.037829957902431,0.0035844487138093,0.87171512842178,0.48853760957718)}
            else:
                return {'pos': (290.558, -277.280, 46.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.277698, 0.961), -130)}
        elif road_id == "9204" or road_id == "9205":
            # return {'pos': (-3, 230.0, 26.2), 'rot': None, 'rot_quat': (0, 0, -0.277698, 0.960669)}
            return {'pos': (-401.98, 243.3, 25.5), 'rot': None, 'rot_quat': (0, 0, -0.277698, 0.960669)}
        elif road_id == "9068":
            return {'pos': (-401.98, 243.3, 25.5), 'rot': None, 'rot_quat': (0, 0, -0.278, 0.961)}
        elif road_id == "9118" or road_id == "9119":
            return {"pos": (-452.972, 16.0, 29.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 210)}
        elif road_id == "9167":
            return {'pos': (105.3, -96.4, 25.3), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 40)}
        elif road_id == "9156":
            return {'pos': (-376.25, 200.8, 25.0), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 45)}
            # return {'pos': (-379.184,208.735,25.4121), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 90)}
        elif road_id == "9189":
            return {'pos': (-383.498, 436.979, 32.1), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 120)}
        elif road_id == "9201" or road_id == "9202": # lanelines
            return {'pos': (-315.2, 80.94, 32.33), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -20)}
        # elif road_id == "9062":
        #     return {'pos': (-315.2, 80.94, 32.33), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -20)}
        elif road_id == "9069": # paved narrow service road
            return {'pos': (-77.27,-135.96,29.65), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "9167": #windy cutthru
            return {'pos': (105.27052307128906,-96.36903381347656,25.5), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), 0)}
        elif road_id == "9062": # forest track big concrete & grass shoulder
            return {'pos': (-3.038116455078125,231.2467498779297,25.9), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -45)}
        elif road_id == "9156": # another windy cutthru, lanelines
            return {'pos': (-376.2503356933594,200.80081176757812,25.2), 'rot': None, 'rot_quat': turn_X_degrees((0, 0, -0.2777, 0.9607), -20)}
        elif road_id == "9095":
            return {'pos': (-150.1,174.6,32.2), 'rot': None, 'rot_quat': turn_X_degrees((0.00, 0.00, 0.00, 1.0), 150)}


        # TODO: test roads below as validation tracks vvvv
        elif road_id == "9198":
            return {'pos': (-427.49517822265625, 17.393047332763672, 31.0), 'rot': None, 'rot_quat': turn_X_degrees((0.00, 0.00, 0.00, 1.0), -11)}
        elif road_id == "9377":
            return {'pos': (186.03909301757812, 57.94004821777344, 26.761920928955078), 'rot': None, 'rot_quat': turn_X_degrees((0.00, 0.00, 0.00, 1.0), 0)}
        elif road_id == "9379":
            return {'pos': (-529.9602661132812, 407.200439453125, 26.696285247802734), 'rot': None, 'rot_quat': turn_X_degrees((0.00, 0.00, 0.00, 1.0), 0)}
        elif road_id == "9441":
            return {'pos': (41.81145095825195, 241.10394287109375, 27.46866798400879), 'rot': None, 'rot_quat': turn_X_degrees((0.00, 0.00, 0.00, 1.0), 0)}
        elif road_id == "9369":
            return {'pos': (380.3826599121094, -164.01087951660156, 38.961448669433594), 'rot': None, 'rot_quat': turn_X_degrees((0.00, 0.00, 0.00, 1.0), 0)}
        elif road_id == "9265":
            return {'pos': (185.29598999023438, 213.83651733398438, 29.49993896484375), 'rot': None, 'rot_quat': turn_X_degrees((0.00, 0.00, 0.00, 1.0), 0)}
        elif road_id == "9211":
            return {'pos': (429.05841064453125, -43.24795913696289, 28.02117347717285), 'rot': None, 'rot_quat': turn_X_degrees((0.00, 0.00, 0.00, 1.0), 0)}
        elif road_id == "9309":
            return {'pos': (-344.4768981933594, 348.6491394042969, 25.605924606323242), 'rot': None, 'rot_quat': turn_X_degrees((0.00, 0.00, 0.00, 1.0), 0)}
        elif road_id == "9338":
            return {'pos': (279.7564697265625, 114.01238250732422, 25.297771453857422), 'rot': None, 'rot_quat': turn_X_degrees((0.00, 0.00, 0.00, 1.0), 0)}
        elif road_id == "9339":
            return {'pos': (60.45085144042969, 30.721210479736328, 25.19574546813965), 'rot': None, 'rot_quat': turn_X_degrees((0.00, 0.00, 0.00, 1.0), 0)}
        elif road_id == "9061":
            return {'pos': (-286.7733154296875, 215.09913635253906, 25.04488182067871), 'rot': None, 'rot_quat': turn_X_degrees((0.00, 0.00, 0.00, 1.0), 0)}
        elif road_id == "9401":
            return {'pos': (193.4588165283203, 115.05908966064453, 25.34110450744629), 'rot': None, 'rot_quat': turn_X_degrees((0.00, 0.00, 0.00, 1.0), 0)}
        elif road_id == "9307":
            return {'pos': (-245.9326934814453, 235.84124755859375, 25.31020164489746), 'rot': None, 'rot_quat': turn_X_degrees((0.00, 0.00, 0.00, 1.0), 0)}
        elif road_id == "9397":
            return {'pos': (390.2131042480469, -113.14376068115234, 32.6632194519043), 'rot': None, 'rot_quat': turn_X_degrees((0.00, 0.00, 0.00, 1.0), 0)}
        else:
            return {'pos': (-453.309, 373.546, 25.3623), 'rot': None, 'rot_quat': (0, 0, -0.2777, 0.9607)}
    elif default_scenario == "small_island":
        if road_id == "int_a_small_island":
            return {"pos": (280.397, 210.259, 35.023), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 110)}
        elif road_id == "ai_1": # super long circular road around island perimeter
            return {"pos": (314.573, 105.519, 37.5), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 155)}
        elif road_id == "17000": # cliffside
            return {"pos": (309.3996276855469, 254.36805725097656, 30.829261779785156), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "coast_a_nw":  # narrow road through hills, scrub vegetation
            return {"pos": (349.7467346191406, 29.1319580078125, 30.6), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), -30)}
        elif road_id == "17101":
            return {"pos": (-53.561851501464844, -398.9946594238281, 32.6), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "trai_ai10":
            return {"pos": (-130.66000366210938, -239.3667449951172, 69.78095245361328), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "trai_ai15":
            return {"pos": (-403.55950927734375, 88.72032928466797, 25.370380401611328), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "17371":
            return {"pos": (-131.87718200683594, -189.1957550048828, 64.87214660644531), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "17238":
            return {"pos": (-400.277099609375, 8.050675392150879, 26.193714141845703), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "17091":
            return {"pos": (35.88029861450195, 382.6633605957031, 27.408042907714844), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "17125":
            return {"pos": (309.3996276855469, 254.36805725097656, 30.829261779785156), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "17273":
            return {"pos": (-324.3287353515625, -348.68072509765625, 30.709070205688477), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "17218":
            return {"pos": (-149.96421813964844, -39.074378967285156, 36.994625091552734), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "16976":
            return {"pos": (181.13726806640625, 190.68841552734375, 39.002593994140625), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "int_d_ind2ind":
            return {"pos": (194.14175415039062, 199.45742797851562, 37.81889724731445), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "coast_a_se":
            return {"pos": (-351.2323303222656, -269.44140625, 37.919189453125), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "ai_5":
            return {"pos": (286.218017578125, 226.80758666992188, 34.25876235961914), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "int_d_ne":
            return {"pos": (-257.8052062988281, -96.90257263183594, 40.792877197265625), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "17156":
            return {"pos": (101.83289337158203, 354.48736572265625, 31.159273147583008), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "int_d_sn_c":
            return {"pos": (-190.1091766357422, 234.6486053466797, 48.3748779296875), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "17087":
            return {"pos": (14.356404304504395, -396.12982177734375, 31.31866455078125), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "17082":
            return {"pos": (-361.6777648925781, -296.4471435546875, 34.90195083618164), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "trail_ai3":
            return {"pos": (268.7442932128906, -403.5768737792969, 25.223878860473633), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "trail_ai8":
            return {"pos": (-410.7110595703125, -87.706787109375, 28.3528499603271), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "int_d_nw":
            return {"pos": (32.715553283691406, -341.3067626953125, 42.874385833740234), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "ai_6":
            return {"pos": (-86.56625366210938, 399.794677734375, 27.000682830810547), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "int_d_mountain_e":
            return {"pos": (-191.31536865234375, -126.09651184082031, 49.82212829589844), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "17118":
            return {"pos": (203.2488555908203, 325.47216796875, 32.08655548095703), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "ai_2":
            return {"pos": (349.98919677734375, 28.93511962890625, 30.297290802001953), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "17222":
            return {"pos": (-343.1805725097656, -31.032434463500977, 37.5694465637207), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "17108":
            return {"pos": (-21.64780616760254, 309.9963073730469, 35.27057647705078), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "17133":
            return {"pos": (-315.55181884765625, 252.47760009765625, 48.91640090942383), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "trail_ai":
            return {"pos": (375.11102294921875, 42.58416748046875, 26.3427734375), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "int_d_mountain_w":
            return {"pos": (215.83319091796875, -212.47642517089844, 45.62574005126953), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "16991":
            return {"pos": (309.3996276855469, 254.36805725097656, 30.829261779785156), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "16975":
            return {"pos": (-146.8442840576172, 380.08917236328125, 29.050579071044922), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "int_d_center_mid":
            return {"pos": (-156.75552368164062, -40.85871887207031, 37.022193908691406), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "17034":
            return {"pos": (-66.46664428710938, 383.8621520996094, 28.15357208251953), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "trai_ai12":
            return {"pos": (-49.940093994140625, -215.75543212890625, 95.97010803222656), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "17235":
            return {"pos": (87.08615112304688, 156.2106475830078, 41.77522659301758), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "trail_ai9":
            return {"pos": (-327.7767028808594, 340.14697265625, 25.586414337158203), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "int_d_center_w":
            return {"pos": (148.710205078125, 19.351869583129883, 33.63618469238281), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "17263":
            return {"pos": (-362.70074462890625, -298.3653259277344, 34.73210144042969), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "16979":
            return {"pos": (-332.97955322265625, 19.397611618041992, 38.34748077392578), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        elif road_id == "trail_ai4":
            return {"pos": (-83.31843566894531, -392.3854675292969, 31.5), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {"pos": (), 'rot': None, 'rot_quat': turn_X_degrees((-0.0, 0.0, -0.0, 1.0), 0)}

        else:
            return {'pos': (254.77, 233.82, 39.5792), 'rot': None, 'rot_quat': (-0.013234630227089, 0.0080483080819249, -0.00034890600363724, 0.99987995624542)}
    elif default_scenario == "jungle_rock_island":
        # road edges:
        # dirt roads: dirt_road_hills, d_lighthous_b, 7930, 8238 7872 8101 dirt_mountain_a 8242
        # if road_id == "8239": # too short?
        #     return {'pos': (792.8975219726562, -678.6703491210938, 159.9837646484375), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        if road_id == "drift_road_op":
            return {'pos': (-876.4090576171875, -502.31439208984375, 140.5927734375), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), -60)}
        elif road_id == "8312": # too short?
            return {'pos': (-459.8533020019531, 824.875, 128.66307067871094), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), -90)}
        elif road_id == "8325":
            return {'pos': (-662.950927734375, 708.9342041015625, 145.57579040527344), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "mountain_road_e":
            return {'pos': (52.57740020751953, 376.9378967285156, 265.7337951660156), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), -70)}
        elif road_id == "drift_road_b":
            return {'pos': (-692.7639770507812, -503.6965026855469, 160.9375), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), -190)}
        elif road_id == "8161":
            return {'pos': (-892.5745239257812, -489.0263671875, 143.31333923339844), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), 160)}
        elif road_id == "8082":
            return {'pos': (-297.006591796875, -632.435546875, 172.7034912109375), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "mountain_alt_f":
            return {'pos': (-107.017822265625, 15.058277130126953, 204.13226318359375), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), -90)}
        elif road_id == "mountain_road_i":
            return {'pos': (360.1, 231.1, 195.5), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), 130)}
        elif road_id == "8114":
            return {'pos': (425.51971435546875, -872.8245849609375, 161.36889648437), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 90)}
        elif road_id == "drift_road_m":
            return {'pos': (215.98255920410156, -674.3016357421875, 156.3968505859375), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), 90)}
        elif road_id == "drift_road_k":
            return {'pos': (213.72064208984375, -674.1602172851562, 156.4460296630), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), -90)}
        elif road_id == "8131":
            return {'pos': (-6.0388641357421875, -856.5894165039062, 161.5219421386), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), -110)}
        elif road_id == "mountain_alt_a":
            return {'pos': (-71.41427612304688, 448.9234313964844, 242.171234130), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), 90)}
        elif road_id == "7994":
            return {'pos': (909.0350952148438, -661.3153686523438, 150.28479003), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), 180)}
        elif road_id == "8000":
            return {'pos': (-781.6738891601562, -564.2924194335938, 198.054534912), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), -15)}
        elif road_id == "outer_road_c":
            return {'pos': (619.7442626953125, 555.2013549804688, 154.5972137451), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), 130)}
        elif road_id == "7849":
            return {'pos': (-505.47186279296875, -178.8316650390625, 135.61096191406), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "8241":
            return {'pos': (800.6476440429688, -672.3448486328125, 159.9921417236328), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "mountain_road_c":
            return {'pos': (442.8168640136719, 539.40380859375, 179.636398315), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), 150)}
        elif road_id == "7902":
            return {'pos': (-541.4922485351562, 537.37841796875, 162.2920074462890), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "7840":
            return {'pos': (51.592140197753906, 376.80413818359375, 265.8583374023), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "8021":
            return {'pos': (-786.2412109375, 912.8829956054688, 143.601959228), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "8240":
            return {'pos': (796.7699584960938, -675.5048217773438, 160.011627197), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "8025":
            return {'pos': (-623.0732421875, 565.7265014648438, 155.86181640), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "7947":
            return {'pos': (-781.8087158203125, -562.53271484375, 197.9423217), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "outer_road_a":
            return {'pos': (-195.96287536621094, -252.71876525878906, 129.5), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), -75)}
        elif road_id == "7971":
            return {'pos': (-183.8403778076172, -256.1764831542969, 128.982543945), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "drift_road_d":
            return {'pos': (-324.958251953125, -731.504150390625, 163.57069396972), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), -50)}
        elif road_id == "outer_road_b":
            return {'pos': (619.1002807617188, 555.6561889648438, 154.64169311523438), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), -50)}
        elif road_id == "drift_road_f":
            return {'pos': (-227.26190185546875, -758.22509765625, 148.8), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), 15)}
        elif road_id == "main_tunnel":
            return {'pos': (-500.2301025390625, -186.64100646972656, 135.9), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 165)}
        elif road_id == "drift_road_s":
            return {'pos': (-459.158,103.86,135.381), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), 180)}
        elif road_id == "drift_road_e":
            return {'pos': (-226.21485900878906, -744.7556762695312, 148.9), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), 180)}
        elif road_id == "drift_road_a":
            # c3(-283.544,-249.771,130.134)
            return {'pos': (-280.0, -246.76296997070312, 130.8), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "mountain_road_h":
            return {'pos': (141.54734802246094, 275.77142333984375, 254.2), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), 180)}
        elif road_id == "drift_road_c":
            return {'pos': (-298.3597717285156, -750.8599243164062, 182.86871337890625), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "mountain_alt_e":
            return {'pos': (-160.021728515625, 317.7889404296875, 240.9942626953125), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "drift_road_p":
            return {'pos': (-874.7346801757812, -503.2436218261719, 140.37405395507812), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {'pos': (), 'rot': None, 'rot_quat':turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        else:
            return {'pos': (-10.0, 580.73, 156.8), 'rot': None, 'rot_quat': (-0.0067, 0.0051, 0.6231, 0.7821)}
    elif default_scenario == 'driver_training': #etk driver experience center
        # TODO: test for validation roads vvvv
        if road_id == "7837": # roundabout
            return {'pos': (-206.64938354492188, 284.364501953125, 52.87702941894531), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "8023": # part of lane of multilane road
            return {'pos': (-308.1035461425781, 125.30075073242188, 54.28969955444336), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        # elif road_id == "7764": # road edge
        #     return {'pos': (-272.9448547363281, 193.0986785888672, 51.736209869384766), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "8013": # weird road with water and grate spouting water, weird lane lines
            return {'pos': (-352.1323547363281, 85.90463256835938, 51.7182159423828), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        # elif road_id == "7878": # road edge
        #     return {'pos': (-86.38534545898438, 284.6070556640625, 51.6995849609375), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        # elif road_id == "7813": # road edge
        #     return {'pos': (-117.55108642578125, 361.30792236328125, 52.8898849487304), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "8017": # USE FOR VALIDATION(?)
            return {'pos': (-276.734130859375, 217.2627716064453, 52.2891845703), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 80)} # farther away
            return {'pos': (-300.05,212.178,53.0684), 'rot': None, 'rot_quat': turn_X_degrees((-0.005521287675947,0.00082905520685017,0.63220649957657,0.77477985620499), 0)}  # closer
        elif road_id == "7785":
            if reverse:
                return {'pos': (19.981637954711914, 251.74679565429688, 50.6), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 150)}
            else:
                return {'pos': (-95.3030776977539, 298.9210510253906, 51.9328689575), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "7752": # roundabout
            return {'pos': (-199.17970275878906, 256.0574035644531, 52.82499313354492), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "7754":
            return {'pos': (81.27847290039062, 44.85991668701172, 38.1568527221679), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 35)}
        elif road_id == "7989":
            return {'pos': (150.19183349609375, 118.05951690673828, 37.6821174621582), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "8005": # same as 7774
            return {'pos': (-383.8293151855469, 147.43643188476562, 51.49815368652), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "7868":
            return {'pos': (-123.0858383178711, -60.699684143066406, 33.26995086669), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "7833":
            return {'pos': (-275.2615966796875, -89.3564453125, 45.14498519897), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "7700":
            return {'pos': (-92.44567108154297, 295.8944091796875, 51.8337593078), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "7952":
            return {'pos': (-289.6571350097656, -143.10055541992188, 44.70738220214844), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "7743":
            return {'pos': (-174.78211975097656, 328.0307312011719, 53.0), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "7646":
            return {'pos': (-93.61753845214844, 171.70712280273438, 49.9917755126953), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "BigRoad_1":
            if seg == 1:
                return {'pos': (58.5842,73.7687,38.4945), 'rot': None,'rot_quat': turn_X_degrees((0.011,0.009,0.864,0.504), 0)}
            elif seg == 2:
                return {'pos': (-267.591, 312.492, 53.2779), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 10)}
            else:
                return {'pos': (85.3, 50.4, 38.04322052001953), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 132)}
        elif road_id == "7774":
            return {'pos': (-508.4, -301.4, 38.5), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), -110)}
        elif road_id == "7927":
            return {'pos': (-131.27334594726562, 182.663818359375, 50.37403869628906), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "7861":
            return {'pos': (323.1593322753906, -177.75326538085938, 36.27228546142578), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), -125)}
        elif road_id == "7990":
            return {'pos': (-414.2222900390625, 58.4100456237793, 50.30381393432617), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 70)}
        elif road_id == "7652": # same as 7861
            return {'pos': (138.69186401367188, -157.484130859375, 35.24814987182617), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), -55)}
        elif road_id == "7820": # weird road markings
            return {'pos': (-357.1138000488281, 102.91966247558594, 52.74211883544922), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "7748": # remainder of circular closed wide track # same as 7852
            return {'pos': (170.18038940429688, 61.981048583984375, 36.38764190673828), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 65)}
        elif road_id == "7852":
            return {'pos': (187.57199096679688, 61.35380554199219, 35.1498870849), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "7645":
            return {'pos': (-221.4468231201172, 49.05771255493164, 45.2898712158203), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), -55)}
        elif road_id == "7643": # same as BigRoad_1
            return {'pos': (57.03415298461914, 75.23177337646484, 38.11738967895508), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "7835":
            return {'pos': (112.60657501220703, 137.0403289794922, 47.866409301757), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), -90)}
        else:
            return {'pos':(60.6395, 70.8329, 38.3048), 'rot': None, 'rot_quat':(0.015, 0.006, 0.884, 0.467)}
    # elif default_scenario == 'east_coast_usa':
    elif default_scenario == 'utah':
        if road_id == "westhighway":
            # return {'pos': (-922.158, -929.868, 135.534), 'rot': None, 'rot_quat': turn_180((0, 0, -0.820165, 0.572127))}
            return {'pos': (-1005.94, -946.995, 135.624), 'rot': None, 'rot_quat': (-0.0087888045236468, -0.0071660503745079, 0.38833409547806, 0.9214488863945)}
        elif road_id == "westhighway2":
            # after tunnel
            # return {'pos': (980.236, -558.879, 148.511), 'rot': None, 'rot_quat': (-0.015679769217968, -0.0069956826046109, 0.59496110677719, 0.80357110500336)}
            # return {'pos': (-241.669, -1163.42, 150.153), 'rot': None, 'rot_quat': (-0.0054957182146609, -0.0061398106627166, -0.69170582294464, 0.72213244438171)}
            return {'pos': (806.49749756, -652.42816162, 147.92123413), 'rot': None, 'rot_quat': (-0.0052490886300802, 0.007554049603641, 0.48879739642143, 0.87234884500504)}
        elif road_id == "15152":
            return {'pos': (-554.169921875,-1143.9346923828125,149.9), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 90)}
        elif road_id == "15144":
            return {'pos': (-984.30859375,-937.9894409179688,135.5), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 90)}
        elif road_id == "14933":
            return {'pos': (-525.8953857421875, -822.3724975585938, 142.1), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), -20)}
        elif road_id == "14979":
            return {'pos': (-884.8267211914062, -275.3382263183594, 266.7), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 67)}
        elif road_id == "14963":
            return {'pos': (723.7115478515625, 346.7457580566406, 144.7), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 110)}
        elif road_id == "14936": # too short
            return {'pos': (-183.26243591308594, -900.5277709960938, 133.2), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "14934": # too short
            return {'pos': (884.0354614257812, -605.5223999023438, 148.3), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "14928":
            return {'pos': (883.9509887695312, -605.48486328125, 148.4), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 64)}
        elif road_id == "14926": #too short
            return {'pos': (-803.3829956054688, -904.5082397460938, 145.4), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        elif road_id == "14912": # exit tunnel driving on right
            return {'pos': (-978.60888671875, -955.3838500976562, 135.6), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), -115)}
        elif road_id == "14923":
            return {'pos': (733.4808349609375, -716.1962890625, 147.6), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), -130)}
        # elif road_id == "":
        #     return {'pos': (), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {'pos': (), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}
        # elif road_id == "":
        #     return {'pos': (), 'rot': None, 'rot_quat': turn_X_degrees((0.0, 0.0, 0.0, 1.0), 0)}

        elif road_id == "buildingsite":
            # return {'pos': (-910.372, 607.927, 265.059), 'rot': None, 'rot_quat': (0, 0, 0.913368, -0.407135)}
            # on road near building site
            return {'pos': (-881.524, 611.674, 264.266), 'rot': None, 'rot_quat': (0, 0, 0.913368, -0.407135)}
        elif road_id == "touristarea":
            return {'pos': (-528.44, 283.886, 298.365), 'rot': None, 'rot_quat': (0, 0, 0.77543, 0.631434)}
        elif road_id == "auto repair zone":
            return {'pos': (771.263, -149.268, 144.291), 'rot': None, 'rot_quat': (0, 0, -0.76648, 0.642268)}
        elif road_id == "campsite":
            return {'pos': (566.186, -530.957, 135.291), 'rot': None, 'rot_quat': ( -0.0444918, 0.0124419, 0.269026, 0.962024)}
        elif road_id == "default":
            return {'pos': ( 771.263, -149.268, 144.291), 'rot': None, 'rot_quat': (0, 0, -0.76648, 0.642268)} #(do not use for training)
        # COLLECTED UTAH8 return {'pos': (835.449, -164.877, 144.57), 'rot': None, 'rot_quat': (-0.003, -0.0048, -0.172, 0.985)}
        # parking lot (do not use for training)
        # return {'pos': (907.939, 773.502, 235.878), 'rot': None, 'rot_quat': (0, 0, -0.652498, 0.75779)} #(do not use for training)
        # COLLECTED UTAH9 return {'pos': (963.22,707.785,235.583), 'rot': None, 'rot_quat': (-0.027, 0.018, -0.038, 0.999)}
        # west highway 2
        # COLLECTED UTAH10 return {'pos': (-151.542, -916.292, 134.561), 'rot': None, 'rot_quat': (0.017533652484417, 0.01487538497895, -0.68549990653992, 0.72770953178406)}
    elif default_scenario == 'italy':
        return {'pos': (-690.403,-1338.64,140.514), 'rot': None, 'rot_quat': turn_X_degrees((-0.00386,0.0038,0.6757,0.7372), 0)}
    elif default_scenario == 'derby':
        if road_id == 'big8':
            return {'pos': (-174.882, 61.4717, 83.5583), 'rot': None, 'rot_quat': (-0.119, -0.001, 0.002, 0.993)}

def get_distance_traveled(traj):
    dist = 0.0
    for i in range(len(traj[:-1])):
        dist += math.sqrt(math.pow(traj[i][0] - traj[i+1][0],2) + math.pow(traj[i][1] - traj[i+1][1],2) + math.pow(traj[i][2] - traj[i+1][2],2))
    return dist

def distance2D(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def turn_X_degrees(rot_quat, degrees=90):
    r = R.from_quat(list(rot_quat))
    r = r.as_euler('xyz', degrees=True)
    r[2] = r[2] + degrees
    r = R.from_euler('xyz', r, degrees=True)
    return tuple(r.as_quat())

def get_topo(topo_id):
    # automation_test_track roads
    if "countryrd" in topo_id:
        default_scenario = "automation_test_track"; road_id="7991"; seg=None; reverse=False
    elif "Rturn_mtnrd" in topo_id:
        default_scenario = "automation_test_track"; road_id="8357"; seg=None; reverse=False
    elif "Lturnyellow" in topo_id:
        default_scenario = "automation_test_track"; road_id="8000"; seg=None; reverse=False
    elif "straightcommercialroad" in topo_id:
        default_scenario = "automation_test_track"; road_id="7909"; seg=None; reverse=False
    elif "Rturninnertrack" in topo_id:
        default_scenario = "automation_test_track"; road_id="7776"; seg=None; reverse=False
    elif "straightwidehighway" in topo_id:
        default_scenario = "automation_test_track"; road_id="7736"; seg=None; reverse=False
    elif "Rturncommercialunderpass" in topo_id:
        default_scenario = "automation_test_track"; road_id="7804"; seg=None; reverse=False
    elif "Lturncommercialcomplex" in topo_id:
        default_scenario = "automation_test_track"; road_id="8396"; seg=None; reverse=False
    elif "Lturnpasswarehouse" in topo_id:
        default_scenario = "automation_test_track"; road_id="8330"; seg=None; reverse=False
    elif "Rturnserviceroad" in topo_id:
        default_scenario = "automation_test_track"; road_id="8038"; seg=None; reverse=False
    elif "Rturnlinedmtnroad" in topo_id:
        default_scenario = "automation_test_track"; road_id="7882"; seg=None; reverse=False
    elif "Rturnrockylinedmtnroad" in topo_id:
        default_scenario = "automation_test_track"; road_id="8290"; seg=None; reverse=False

    # hirochi_raceway roads
    elif "Rturn_hirochitrack" in topo_id:
        default_scenario = "hirochi_raceway"; road_id="9205"; seg=None; reverse=False
    elif "Rturn_sidequest" in topo_id:
        default_scenario = "hirochi_raceway"; road_id="9119"; seg=None; reverse=False
    elif "Rturn_lanelines" in topo_id:
        default_scenario = "hirochi_raceway"; road_id="9202"; seg=None; reverse=False
    elif "Rturn_maintenancerd" in topo_id:
        default_scenario = "hirochi_raceway"; road_id="9069"; seg=None; reverse=False
    elif "Rturn_bridge" in topo_id:
        default_scenario = "hirochi_raceway"; road_id="9095"; seg=None; reverse=False
    elif "Rturn_narrowcutthru" in topo_id:
        default_scenario = "hirochi_raceway"; road_id="9167"; seg=None; reverse=False
    elif "Rturn_bigshoulder" in topo_id:
        default_scenario = "hirochi_raceway"; road_id="9062"; seg=None; reverse=False
    elif "Rturn_servicecutthru" in topo_id:
        default_scenario = "hirochi_raceway"; road_id="9156"; seg=None; reverse=False
    elif "Lturn_narrowservice" in topo_id:
        default_scenario = "hirochi_raceway"; road_id="9198"; seg=None; reverse=False

    # industrial roads
    elif "extrawinding_industrialtrack" in topo_id:
        default_scenario = "industrial"; road_id="7982"; seg=None; reverse=False
    elif "extrawinding_industrialrcasphalta" in topo_id:
        default_scenario = "industrial"; road_id = "rc_asphalta"; seg=None; reverse = False
    elif "extrawinding_industrial7978" in topo_id:
        default_scenario = "industrial"; road_id = "7978"; seg=None; reverse = False
    elif "Rturn_industrial7978" in topo_id: #   TODO: Reverse this one, dupe of above
        default_scenario = "industrial"; road_id="7978"; seg=None; reverse=False
    elif "Rturn_industrialrc_asphaltd" in topo_id:
        default_scenario = "industrial"; road_id="rc_asphaltd"; seg=None; reverse=False
    elif "Rturn_industrialrc_asphaltc" in topo_id:
        default_scenario = "industrial"; road_id="rc_asphaltc"; seg=None; reverse=False
    elif "Rturn_industrialrc_asphaltb" in topo_id:
        default_scenario = "industrial"; road_id="rc_asphaltb"; seg=None; reverse=False

    elif "Rturn_industrial8022whitepave" in topo_id:
        default_scenario = "industrial"; road_id="8022"; seg=None; reverse=False
    elif "Rturn_industrial8068widewhitepave" in topo_id:
        default_scenario = "industrial"; road_id="8068"; seg=None; reverse=False
    elif "Rturn_industrialnarrowservicerd" in topo_id:
        default_scenario = "industrial"; road_id="8079"; seg=None; reverse=False
    # elif "extra_test0" in topo_id:
    #     default_scenario = "industrial"; road_id="8009"; seg=None; reverse=False
    # TODO: finish testing industrial roads or use for validation
    
    # small_island roads
    elif "Rturn_int_a_small_island" in topo_id:
        default_scenario = "small_island"; road_id="int_a_small_island"; seg=None; reverse=False
    elif "Rturn_small_island_ai_1" in topo_id: # super long circular road around island perimeter
        default_scenario = "small_island"; road_id="ai_1"; seg=None; reverse=False
    # TODO: finish testing small_island roads or use for validation
    elif "extra_small_islandcoast_a_nw" in topo_id: # narrow road through hills, scrub vegetation
        default_scenario = "small_island"; road_id="coast_a_nw"; seg=None; reverse=False
    elif "extra_test1" in topo_id:
        default_scenario = "small_island"; road_id="17101"; seg=None; reverse=False
    elif "extra_test1" in topo_id:
        default_scenario = "small_island"; road_id="trai_ai10"; seg=None; reverse=False
    elif "extra_test1" in topo_id:
        default_scenario = "small_island"; road_id="trai_ai15"; seg=None; reverse=False
    elif "extra_test1" in topo_id:
        default_scenario = "small_island"; road_id="17371"; seg=None; reverse=False
    elif "extra_test1" in topo_id:
        default_scenario = "small_island"; road_id="17238"; seg=None; reverse=False
    elif "extra_test1" in topo_id:
        default_scenario = "small_island"; road_id="17091"; seg=None; reverse=False
    elif "extra_test1" in topo_id:
        default_scenario = "small_island"; road_id="17125"; seg=None; reverse=False
    elif "extra_test1" in topo_id:
        default_scenario = "small_island"; road_id="17273"; seg=None; reverse=False
    elif "extra_test1" in topo_id:
        default_scenario = "small_island"; road_id="17218"; seg=None; reverse=False
    elif "extra_test1" in topo_id:
        default_scenario = "small_island"; road_id="16976"; seg=None; reverse=False
    elif "extra_test1" in topo_id:
        default_scenario = "small_island"; road_id="int_d_ind2ind"; seg=None; reverse=False
    elif "extra_test1" in topo_id:
        default_scenario = "small_island"; road_id="coast_a_se"; seg=None; reverse=False
    elif "extra_test1" in topo_id:
        default_scenario = "small_island"; road_id="ai_5"; seg=None; reverse=False
    elif "extra_test1" in topo_id:
        default_scenario = "small_island"; road_id="int_d_ne"; seg=None; reverse=False
    elif "extra_test1" in topo_id:
        default_scenario = "small_island"; road_id="17156"; seg=None; reverse=False
    elif "extra_test1" in topo_id:
        default_scenario = "small_island"; road_id="int_d_sn_c"; seg=None; reverse=False
    elif "extra_test1" in topo_id:
        default_scenario = "small_island"; road_id="17087"; seg=None; reverse=False
    elif "extra_test1" in topo_id:
        default_scenario = "small_island"; road_id="17082"; seg=None; reverse=False
    elif "extra_test1" in topo_id:
        default_scenario = "small_island"; road_id="trail_ai3"; seg=None; reverse=False
    elif "extra_test1" in topo_id:
        default_scenario = "small_island"; road_id="trail_ai8"; seg=None; reverse=False
    elif "extra_test1" in topo_id:
        default_scenario = "small_island"; road_id="int_d_nw"; seg=None; reverse=False
    elif "extra_test1" in topo_id:
        default_scenario = "small_island"; road_id="ai_6"; seg=None; reverse=False
    elif "extra_test1" in topo_id:
        default_scenario = "small_island"; road_id="int_d_mountain_e"; seg=None; reverse=False
    elif "extra_test1" in topo_id:
        default_scenario = "small_island"; road_id="17118"; seg=None; reverse=False
    elif "extra_test1" in topo_id:
        default_scenario = "small_island"; road_id="ai_2"; seg=None; reverse=False

    # jungle_rock_island roads
    elif "narrowjungleroad1" in topo_id:
        default_scenario = "jungle_rock_island"; road_id="drift_road_op"; seg=None; reverse=False
    elif "narrowjungleroad2" in topo_id:
        default_scenario = "jungle_rock_island"; road_id="8312"; seg=None; reverse=False
    elif "Lturn_junglemountain_road_e" in topo_id:
        default_scenario = "jungle_rock_island"; road_id="mountain_road_e"; seg=None; reverse=False
    elif "extra_jungledrift_road_b" in topo_id:
        default_scenario = "jungle_rock_island"; road_id="drift_road_b"; seg=None; reverse=False
    elif "extra_jungle8161" in topo_id:
        default_scenario = "jungle_rock_island"; road_id="8161"; seg=None; reverse=False
    elif "extra_windyjungle8082" in topo_id:
        default_scenario = "jungle_rock_island"; road_id="8082"; seg=None; reverse=False
    elif "extra_junglemountain_alt_f" in topo_id:
        default_scenario = "jungle_rock_island"; road_id="mountain_alt_f"; seg=None; reverse=False
    elif "extra_junglemountain_road_i" in topo_id:
        default_scenario = "jungle_rock_island"; road_id="mountain_road_i"; seg=None; reverse=False
    elif "extra_junglemeander8114" in topo_id: # lanelines
        default_scenario = "jungle_rock_island"; road_id="8114"; seg=None; reverse=False
    elif "extra_jungledrift_road_m" in topo_id:
        default_scenario = "jungle_rock_island"; road_id="drift_road_m"; seg=None; reverse=False
    elif "extra_jungledrift_road_k" in topo_id:
        default_scenario = "jungle_rock_island"; road_id="drift_road_k"; seg=None; reverse=False
    elif "extra_jungle8131" in topo_id:
        default_scenario = "jungle_rock_island"; road_id="8131"; seg=None; reverse=False
    elif "extra_junglemountain_alt_a" in topo_id:
        default_scenario = "jungle_rock_island"; road_id="mountain_alt_a"; seg=None; reverse=False
    elif "extra_junglemeander7994" in topo_id:
        default_scenario = "jungle_rock_island"; road_id="7994"; seg=None; reverse=False
    elif "extra_jungle8000" in topo_id: # no lane lines
        default_scenario = "jungle_rock_island"; road_id="8000"; seg=None; reverse=False
    elif "extra_jungleouter_road_c" in topo_id:
        default_scenario = "jungle_rock_island"; road_id="outer_road_c"; seg=None; reverse=False
    # elif "extra_test2" in topo_id: # test later
    #     default_scenario = "jungle_rock_island"; road_id="8241"; seg=None; reverse=False
    elif "extra_junglemountain_road_c" in topo_id:
        default_scenario = "jungle_rock_island"; road_id="mountain_road_c"; seg=None; reverse=False
    elif "extra_jungleouter_road_a" in topo_id:
        default_scenario = "jungle_rock_island"; road_id="outer_road_a"; seg=None; reverse=False
    elif "extra_jungledrift_road_d" in topo_id:
        default_scenario = "jungle_rock_island"; road_id="drift_road_d"; seg=None; reverse=False
    elif "extra_jungleouter_road_b" in topo_id:
        default_scenario = "jungle_rock_island"; road_id="outer_road_b"; seg=None; reverse=False
    # TODO: Finish testing jungle_rock_island, or use for validation?
    elif "extra_jungledrift_road_f" in topo_id:
        default_scenario = "jungle_rock_island"; road_id="drift_road_f"; seg=None; reverse=False
    elif "extra_junglemain_tunnel" in topo_id:
        default_scenario = "jungle_rock_island"; road_id="main_tunnel"; seg=None; reverse=False
    elif "extra_jungledrift_road_s" in topo_id:
        default_scenario = "jungle_rock_island"; road_id="drift_road_s"; seg=None; reverse=False
    elif "extra_jungledrift_road_e" in topo_id:
        default_scenario = "jungle_rock_island"; road_id="drift_road_e"; seg=None; reverse=False
    elif "extra_jungledrift_road_a" in topo_id: #"extra_jungledrift_road_a"
        default_scenario = "jungle_rock_island"; road_id="drift_road_a"; seg=None; reverse=False
    elif "extra_junglemountain_road_h" in topo_id:
        default_scenario = "jungle_rock_island"; road_id="mountain_road_h"; seg=None; reverse=False
    elif "extra_test2" in topo_id:
        default_scenario = "jungle_rock_island"; road_id="drift_road_c"; seg=None; reverse=False
    elif "extra_test2" in topo_id:
        default_scenario = "jungle_rock_island"; road_id="mountain_alt_e"; seg=None; reverse=False
    elif "extra_test2" in topo_id:
        default_scenario = "jungle_rock_island"; road_id="drift_road_p"; seg=None; reverse=False


    # driver_training roads
    elif "Lturn_test3" in topo_id: # USE FOR VALIDATION
        default_scenario = "driver_training"; road_id="8017"; seg=None; reverse=False
    elif "extra_driver_trainingvalidation2" in topo_id: # USE FOR VALIDATION
        default_scenario = "driver_training"; road_id="7785"; seg=None; reverse=True
    elif "extra_multilanehighway" in topo_id: # multi lane
        default_scenario = "driver_training"; road_id="7754"; seg=None; reverse=False
    elif "extra_multilanehighway2" in topo_id:
        default_scenario = "driver_training"; road_id="7774"; seg=None; reverse=False
    elif "extra_wideclosedtrack" in topo_id: # circular closed wide track
        default_scenario = "driver_training"; road_id="7861"; seg=None; reverse=False
    elif "extra_windingtrack" in topo_id: # lane lines
        default_scenario = "driver_training"; road_id="7990"; seg=None; reverse=False
    elif topo_id == "extra_BigRoad_1":
        default_scenario = "driver_training"; road_id="BigRoad_1"; seg = 1; reverse=False
    elif "extra_wideclosedtrack2" in topo_id: # remainder of circular closed wide track
        default_scenario = "driver_training"; road_id="7748"; seg=None; reverse=False
    elif "extra_windingnarrowtrack" in topo_id: # winding narrow no lanelines
        default_scenario = "driver_training"; road_id="7645"; seg=None; reverse=False
    elif "extra_lefthandperimeter" in topo_id:
        default_scenario = "driver_training"; road_id="7835"; seg=None; reverse=False

    # italy roads
    # TODO: finish testing italy roads
    elif "extra_test4" in topo_id:
        default_scenario = "italy"; road_id=""; seg=None; reverse=False

    # utah roads
    # TODO: finish testing utah roads
    elif "extra_utahtunnel" in topo_id:
        default_scenario = "utah"; road_id = "15152"; seg=None; reverse = False
    elif topo_id == "extra_utahlong":
        default_scenario = "utah"; road_id = "14933"; seg=None; reverse = False
    elif topo_id == "extra_utahswitchback":
        default_scenario = "utah"; road_id = "14979"; seg=None; reverse = False
    elif topo_id == "extra_utahlong2":
        default_scenario = "utah"; road_id = "14963"; seg=None; reverse = False
    elif topo_id == "extra_utahexittunnel":
        default_scenario = "utah"; road_id = "14928"; seg=None; reverse = False
    elif topo_id == "extra_utahexittunnelright": # extra_utahexittunnelright
        default_scenario = "utah"; road_id = "14912"; seg=None; reverse = False
    elif topo_id == "extra_utahturnlane": # turning lane near tunnel, too short
        default_scenario = "utah"; road_id = "14923"; seg=None; reverse = False
    elif topo_id == "extra_test6":
        default_scenario = "utah"; road_id = ""; seg=None; reverse = False
    # elif topo_id == "extra_test6":
    #     default_scenario = "utah"; road_id = ""; seg=None; reverse = False
    # elif topo_id == "extra_test6":
    #     default_scenario = "utah"; road_id = ""; seg=None; reverse = False
    # elif topo_id == "extra_test6":
    #     default_scenario = "utah"; road_id = ""; seg=None; reverse = False
    # elif topo_id == "extra_test6":
    #     default_scenario = "utah"; road_id = ""; seg=None; reverse = False
    # elif topo_id == "extra_test6":
    #     default_scenario = "utah"; road_id = ""; seg=None; reverse = False
    # elif topo_id == "extra_test6":
    #     default_scenario = "utah"; road_id = ""; seg=None; reverse = False
    # elif topo_id == "extra_test6":
    #     default_scenario = "utah"; road_id = ""; seg=None; reverse = False
    # elif topo_id == "extra_test6":
    #     default_scenario = "utah"; road_id = ""; seg=None; reverse = False
    # elif topo_id == "extra_test6":
    #     default_scenario = "utah"; road_id = ""; seg=None; reverse = False
    # elif topo_id == "extra_test6":
    #     default_scenario = "utah"; road_id = ""; seg=None; reverse = False
    # elif topo_id == "extra_test6":
    #     default_scenario = "utah"; road_id = ""; seg=None; reverse = False
    # elif topo_id == "extra_test6":
    #     default_scenario = "utah"; road_id = ""; seg=None; reverse = False
    # elif topo_id == "extra_test6":
    #     default_scenario = "utah"; road_id = ""; seg=None; reverse = False
    # elif topo_id == "extra_test6":
    #     default_scenario = "utah"; road_id = ""; seg=None; reverse = False
    # elif topo_id == "extra_test6":
    #     default_scenario = "utah"; road_id = ""; seg=None; reverse = False
    # elif topo_id == "extra_test6":
    #     default_scenario = "utah"; road_id = ""; seg=None; reverse = False
    # elif topo_id == "extra_test6":
    #     default_scenario = "utah"; road_id = ""; seg=None; reverse = False
    # elif topo_id == "extra_test6":
    #     default_scenario = "utah"; road_id = ""; seg=None; reverse = False
    # elif topo_id == "extra_test6":
    #     default_scenario = "utah"; road_id = ""; seg=None; reverse = False
    # elif topo_id == "extra_test6":
    #     default_scenario = "utah"; road_id = ""; seg=None; reverse = False
    # elif topo_id == "extra_test6":
    #     default_scenario = "utah"; road_id = ""; seg=None; reverse = False
    # elif topo_id == "extra_test6":
    #     default_scenario = "utah"; road_id = ""; seg=None; reverse = False
    # elif topo_id == "extra_test6":
    #     default_scenario = "utah"; road_id = ""; seg=None; reverse = False
    # elif topo_id == "extra_test6":
    #     default_scenario = "utah"; road_id = ""; seg=None; reverse = False
    # elif topo_id == "extra_test6":
    #     default_scenario = "utah"; road_id = ""; seg=None; reverse = False
    # elif topo_id == "extra_test6":
    #     default_scenario = "utah"; road_id = ""; seg=None; reverse = False
    #

    # west_coast_usa roads
    # TODO: finish testing west_coast_usa roads
    elif "Lturn_uphill" in topo_id:
        default_scenario = "west_coast_usa"; road_id="12667"; seg=None; reverse=False
    elif "extra_westcoastrocks" in topo_id:
        default_scenario = "west_coast_usa"; road_id="8518"; seg=None; reverse=True
    elif "extra_dock" in topo_id:
        default_scenario = "west_coast_usa"; road_id="12641"; seg=None; reverse=False
    # todo: explore west_coast_usa nearby roads
    # elif topo_id == "extra_test7": # too short, good nearby roads though
    #     default_scenario = "west_coast_usa"; road_id = "8510"; seg=None; reverse = False
    # elif topo_id == "extra_test7": # too short, good nearby roads though
    #     default_scenario = "west_coast_usa"; road_id = "10551"; seg=None; reverse = False
    # elif topo_id == "extra_test7": # too short, good nearby roads though
    #     default_scenario = "west_coast_usa"; road_id = "11297"; seg=None; reverse = False
    # elif topo_id == "extra_test7": # too short, good nearby roads though
    #     default_scenario = "west_coast_usa"; road_id = "8576"; seg=None; reverse = False
    elif topo_id == "extra_westmtnroad":
        default_scenario = "west_coast_usa"; road_id = "8518"; seg=None; reverse = False
    elif topo_id == "extra_westdockleftside":
        default_scenario = "west_coast_usa"; road_id = "8719"; seg=None; reverse = False
    elif topo_id == "extra_westgrassland":
        default_scenario = "west_coast_usa"; road_id = "8418"; seg=None; reverse = False
    elif topo_id == "extra_westoutskirts":
        default_scenario = "west_coast_usa"; road_id = "8512"; seg=None; reverse = False
    elif topo_id == "extra_westsuburbs":
        default_scenario = "west_coast_usa"; road_id = "13306"; seg=None; reverse = False
    elif topo_id == "extra_westunderpasses":
        default_scenario = "west_coast_usa"; road_id = "13349"; seg=None; reverse = False
    elif topo_id == "extra_westLturnway":
        default_scenario = "west_coast_usa"; road_id = "12930"; seg=None; reverse = False
    elif topo_id == "extra_westofframp":
        default_scenario = "west_coast_usa"; road_id = "11635"; seg=None; reverse = False

    # original roads
    elif "straight" in topo_id:
        default_scenario = "automation_test_track"; road_id = "8185"; seg=None; reverse=False
    elif "extra_winding" in topo_id:
        default_scenario = "west_coast_usa"; road_id = "10988"; seg = 3; reverse=False
    elif "Rturn" in topo_id:
        default_scenario="hirochi_raceway"; road_id="9039"; seg=0; reverse=False
    elif "Lturn" in topo_id:
        default_scenario = "west_coast_usa"; road_id = "12930"; seg=None; reverse=False
    elif "extra_whatever" in topo_id: # might cut, involves intentional turns
        default_scenario = "west_coast_usa"; road_id="13091"; seg=None; reverse=False

    return default_scenario, road_id, seg, reverse


def get_transf(transf_id):
    if transf_id == "regular" or transf_id is None:
        img_dims = (240, 135); fov = 51; transf = "None"
    elif transf_id == "medium":
        img_dims = (192, 108); fov = 51; transf = "None"
    elif transf_id == "mediumfisheye":
        img_dims = (192, 108); fov = 75; transf = "None"
    elif transf_id == "small":
        img_dims = (144, 81); fov = 51; transf = "None"
    elif "fisheye" in transf_id:
        img_dims = (240,135); fov=75; transf = "fisheye"
    elif "resdec" in transf_id:
        img_dims = (96, 54); fov = 51; transf = "resdec"
    elif "resinc" in transf_id:
        img_dims = (480,270); fov = 51; transf = "resinc"
    elif "depth" in transf_id:
        img_dims = (240, 135); fov = 51; transf = "depth"
    return img_dims, fov, transf


def lineseg_dists(p, a, b):
    """Cartesian distance from point to line segment
    Edited to support arguments as series, from:
    https://stackoverflow.com/a/54442561/11208892
    Args:
        - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
        - a: np.array of shape (x, 2)
        - b: np.array of shape (x, 2)
    """
    # normalized tangent vectors
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1, 1)))

    # signed parallel distance components
    # rowwise dot products of 2D vectors
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(s))])

    # perpendicular distance component
    # rowwise cross products of 2D vectors
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

    return np.hypot(h, c)


#return distance between two any-dimenisonal points
def distance(a, b):
    sqr = sum([math.pow(ai-bi, 2) for ai, bi in zip(a,b)])
    # return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)
    return math.sqrt(sqr)


def dist_from_line(centerline, point):
    a = [[x[0],x[1]] for x in centerline[:-1]]
    b = [[x[0],x[1]] for x in centerline[1:]]
    a = np.array(a)
    b = np.array(b)
    dist = lineseg_dists([point[0], point[1]], a, b)
    return dist


def diff_damage(damage, damage_prev):
    if damage is None or damage_prev is None:
        return 0
    else:
        return damage['damage'] - damage_prev['damage']


def ms_to_kph(wheelspeed):
    return wheelspeed * 3.6