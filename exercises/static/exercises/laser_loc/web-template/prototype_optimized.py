from GUI import GUI
from HAL import HAL
import numpy as np
import math
from time import time
import random
# Enter sequential code!
line_1 = np.cross([104,729 - 330,1],[104,729 - 497,1])
line_2 = np.cross([104,729 - 497,1],[157,729 - 497,1])
line_3 = np.cross([157,729 - 497,1],[157,729 - 550,1])
line_4 = np.cross([157,729 - 550,1],[211,729 - 550,1])
line_5 = np.cross([211,729 - 550,1],[211,729 - 604,1])
line_6 = np.cross([211,729 - 604,1],[265,729 - 604,1])
line_7 = np.cross([265,729 - 604,1],[265,729 - 659,1])
line_8 = np.cross([265,729 - 659,1],[431,729 - 659,1])
line_9 = np.cross([431,729 - 659,1],[431,729 - 604,1])
line_10 = np.cross([431,729 - 604,1],[484,729 - 604,1])
line_11 = np.cross([484,729 - 604,1],[484,729 - 547,1])
line_12 = np.cross([484,729 - 547,1],[538,729 - 547,1])
line_13 = np.cross([538,729 - 547,1],[538,729 - 439,1])
line_14 = np.cross([538,729 - 439,1],[647,729 - 439,1])
line_15 = np.cross([647,729 - 439,1],[647,729 - 385,1])
line_16 = np.cross([647,729 - 385,1],[700,729 - 385,1])
line_17 = np.cross([700,729 - 385,1],[700,729 - 283,1])
line_18 = np.cross([747,729 - 277,1],[535,729 - 277,1])
line_19 = np.cross([591,729 - 328,1],[591,729 - 281,1])
line_20 = np.cross([537,729 - 223,1],[537,729 - 281,1])
line_21 = np.cross([590,729 - 224,1],[644,729 - 224,1])
line_22 = np.cross([747,729 - 277,1],[747,729 - 224,1])
line_23 = np.cross([747,729 - 224,1],[698,729 - 224,1])
line_24 = np.cross([698,729 - 224,1],[698,729 - 171,1])
line_25 = np.cross([698,729 - 171,1],[644,729 - 171,1])
line_26 = np.cross([644,729 - 171,1],[644,729 - 117,1])
line_27 = np.cross([644,729 - 117,1],[590,729 - 116,1])
line_28 = np.cross([590,729 - 116,1],[590,729 - 62,1])
line_29 = np.cross([590,729 - 62,1],[482,729 - 62,1]) 
line_30 = np.cross([482,729 - 62,1],[482,729 - 116,1]) 
line_31 = np.cross([482,729 - 116,1],[536,729 - 116,1])
line_32 = np.cross([536,729 - 116,1],[536,729 - 8,1])
line_33 = np.cross([536,729 - 8,1],[428,729 - 8,1])
line_34 = np.cross([428,729 - 8,1],[428,729 - 62,1])
line_35 = np.cross([428,729 - 62,1],[374,729 - 62,1])
line_36 = np.cross([374,729 - 62,1],[374,729 - 116,1])
line_37 = np.cross([374,729 - 116,1],[320,729 - 116,1])
line_38 = np.cross([320,729 - 116,1],[320,729 - 224,1])
line_39 = np.cross([320,729 - 224,1],[212,729 - 224,1])
line_40 = np.cross([212,729 - 224,1],[212,729 - 278,1])
line_41 = np.cross([212,729 - 278,1],[158,729 - 278,1])
line_42 = np.cross([158,729 - 278,1],[158,729 - 332,1])
line_43 = np.cross([104,729 - 332,1],[212,729 - 332,1])
line_44 = np.cross([212,729 - 332,1],[212,729 - 440,1])
line_45 = np.cross([158,729 - 440,1],[266,729 - 440,1])
line_46 = np.cross([158,729 - 440,1],[158,729 - 384,1])
line_47 = np.cross([266,729 - 385,1],[266,729 - 549,1])
line_48 = np.cross([266,729 - 385,1],[212,729 - 385,1])
line_49 = np.cross([266,729 - 549,1],[374,729 - 549,1])
line_50 = np.cross([320,729 - 603,1],[320,729 - 549,1])
line_51 = np.cross([374,729 - 549,1],[374,729 - 495,1])
line_52 = np.cross([374,729 - 495,1],[482,729 - 495,1])
line_53 = np.cross([482,729 - 495,1],[482,729 - 387,1])
line_54 = np.cross([482,729 - 387,1],[431,729 - 387,1])
line_55 = np.cross([431,729 - 387,1],[431,729 - 116,1])
line_56 = np.cross([431,729 - 333,1],[370,729 - 333,1])
line_57 = np.cross([370,729 - 333,1],[370,729 - 279,1])
line_58 = np.cross([370,729 - 279,1],[262,729 - 279,1])
line_59 = np.cross([262,729 - 279,1],[262,729 - 333,1])
line_60 = np.cross([262,729 - 333,1],[316,729 - 333,1])
line_61 = np.cross([316,729 - 333,1],[316,729 - 279,1])
line_62 = np.cross([482,729 - 222,1],[374,729 - 222,1])
line_63 = np.cross([374,729 - 222,1],[374,729 - 168,1])
line_64 = np.cross([482,729 - 170,1],[482,729 - 334,1])
line_65 = np.cross([482,729 - 170,1],[428,729 - 170,1])
line_66 = np.cross([482,729 - 334,1],[536,729 - 334,1])
line_67 = np.cross([536,729 - 334,1],[536,729 - 388,1])
line_68 = np.cross([536,729 - 388,1],[590,729 - 388,1])
line_69 = np.cross([320,729 - 493,1],[320,729 - 439,1])
line_70 = np.cross([373,729 - 437,1],[428,729 - 437,1])
Q = np.array([([104,729 - 330,1],[104,729 - 497,1]),([104,729 - 497,1],[157,729 - 497,1]),([157,729 - 497,1],[157,729 - 550,1]),([157,729 - 550,1],[211,729 - 550,1]),([211,729 - 550,1],[211,729 - 604,1]),
                        ([211,729 - 604,1],[265,729 - 604,1]),([265,729 - 604,1],[265,729 - 659,1]),([265,729 - 659,1],[431,729 - 659,1]),([431,729 - 659,1],[431,729 - 604,1]),([431,729 - 604,1],[484,729 - 604,1]),
                        ([484,729 - 604,1],[484,729 - 547,1]),([484,729 - 547,1],[538,729 - 547,1]),([538,729 - 547,1],[538,729 - 439,1]),([538,729 - 439,1],[647,729 - 439,1]),([647,729 - 439,1],[647,729 - 385,1]),
                        ([647,729 - 385,1],[700,729 - 385,1]),([700,729 - 385,1],[700,729 - 283,1]),([747,729 - 277,1],[535,729 - 277,1]),([591,729 - 328,1],[591,729 - 281,1]),([537,729 - 223,1],[537,729 - 281,1]),
                        ([590,729 - 224,1],[644,729 - 224,1]),([747,729 - 277,1],[747,729 - 224,1]),([747,729 - 224,1],[698,729 - 224,1]),([698,729 - 224,1],[698,729 - 171,1]),([698,729 - 171,1],[644,729 - 171,1]),
                        ([644,729 - 171,1],[644,729 - 117,1]),([644,729 - 117,1],[590,729 - 116,1]),([590,729 - 116,1],[590,729 - 62,1]),([590,729 - 62,1],[482,729 - 62,1]),([482,729 - 62,1],[482,729 - 116,1]),
                        ([482,729 - 116,1],[536,729 - 116,1]),([536,729 - 116,1],[536,729 - 8,1]),([536,729 - 8,1],[428,729 - 8,1]),([428,729 - 8,1],[428,729 - 62,1]),([428,729 - 62,1],[374,729 - 62,1]),
                        ([374,729 - 62,1],[374,729 - 116,1]),([374,729 - 116,1],[320,729 - 116,1]),([320,729 - 116,1],[320,729 - 224,1]),([320,729 - 224,1],[212,729 - 224,1]),([212,729 - 224,1],[212,729 - 278,1]),
                        ([212,729 - 278,1],[158,729 - 278,1]),([158,729 - 278,1],[158,729 - 332,1]),([104,729 - 332,1],[212,729 - 332,1]),([212,729 - 332,1],[212,729 - 440,1]),([158,729 - 440,1],[266,729 - 440,1]),
                        ([158,729 - 440,1],[158,729 - 384,1]),([266,729 - 385,1],[266,729 - 549,1]),([266,729 - 385,1],[212,729 - 385,1]),([266,729 - 549,1],[374,729 - 549,1]),([320,729 - 603,1],[320,729 - 549,1]),
                        ([374,729 - 549,1],[374,729 - 495,1]),([374,729 - 495,1],[482,729 - 495,1]),([482,729 - 495,1],[482,729 - 387,1]),([482,729 - 387,1],[431,729 - 387,1]),([431,729 - 387,1],[431,729 - 116,1]),
                        ([431,729 - 333,1],[370,729 - 333,1]),([370,729 - 333,1],[370,729 - 279,1]),([370,729 - 279,1],[262,729 - 279,1]),([262,729 - 279,1],[262,729 - 333,1]),([262,729 - 333,1],[316,729 - 333,1]),
                        ([316,729 - 333,1],[316,729 - 279,1]),([482,729 - 222,1],[374,729 - 222,1]),([374,729 - 222,1],[374,729 - 168,1]),([482,729 - 170,1],[482,729 - 334,1]),([482,729 - 170,1],[428,729 - 170,1]),
                        ([482,729 - 334,1],[536,729 - 334,1]),([536,729 - 334,1],[536,729 - 388,1]),([536,729 - 388,1],[590,729 - 388,1]),([320,729 - 493,1],[320,729 - 439,1]),([373,729 - 437,1],[428,729 - 437,1])], dtype=object).tolist()
L1 = np.array((line_1, line_2, line_3, line_4, line_5, line_6, line_7, line_8, line_9, line_10, 
                line_11,line_12,line_13,line_14,line_15,line_16,line_17,line_18,line_19,line_20,
                line_21,line_22,line_23,line_24,line_25,line_26,line_27,line_28,line_29,line_30,
                line_31,line_32,line_33,line_34,line_35,line_36,line_37,line_38,line_39,line_40,
                line_41,line_42,line_43,line_44,line_45,line_46,line_47,line_48,line_49,line_50,
                line_51,line_52,line_53,line_54,line_55,line_56,line_57,line_58,line_59,line_60,
                line_61,line_62,line_63,line_64,line_65,line_66,line_67,line_68,line_69,line_70), dtype=object).tolist()
num_particles = 50
particles = []
robot_x = 0
robot_y = 0
robot_yaw = 0
x_min = 102
x_max = 748
y_min = 9
y_max = 657
P1x = 8 / 0.03
P1y = 8 / 0.03
sum_weight_particles = 0
angleincrement = 0.0157473180443
# Posicion inicial del robot
robot_pose_x = HAL.getPose3d().x
robot_pose_y =  HAL.getPose3d().y
robot_pose_yaw = HAL.getPose3d().yaw
while(robot_pose_yaw < 0):
    robot_pose_yaw = math.pi*2 + robot_pose_yaw
while(robot_pose_yaw > math.pi*2):
    robot_pose_yaw = robot_pose_yaw - math.pi*2

# Incertidumbre de la posicion incial
var_x = 10
var_y = 10
# var_yaw = 2.0 * math.pi / 180.0
var_yaw = math.pi
# Etapa de generacion de las particulas
# Se generan las primeras partículas aleatoriamente alrededor del punto incial del robot
wo = 1.0 / float(num_particles)
w_persist = wo
for i in range(num_particles):
    x = np.random.normal(round(robot_pose_x/0.03),var_x)
    y = np.random.normal(round(robot_pose_y/0.03),var_y)
    yaw = np.random.normal(robot_pose_yaw, var_yaw)
    while(yaw < 0):
        yaw = math.pi*2 + yaw
    while(yaw > math.pi*2):
        yaw = yaw - math.pi*2
    particles.append((round(x), round(y), yaw, wo, w_persist))
# Mostrar por el GUI las particulas generadas.
particles = np.array(particles).tolist()
# GUI.showParticles(particles)
Di = 729**2 + 729**2 + 1
D_min = 729**2 + 729**2
first_time = 0
while True:
    # Enter iterative code!
    posx = HAL.getPose3d().x
    posy = HAL.getPose3d().y
    actual_pose_yaw = HAL.getPose3d().yaw
    while(actual_pose_yaw < 0):
        actual_pose_yaw = math.pi*2 + actual_pose_yaw
    while(actual_pose_yaw > math.pi*2):
        actual_pose_yaw = actual_pose_yaw - math.pi*2
    distance_control = math.sqrt((posx - robot_pose_x)**2 + (posy - robot_pose_y)**2)
    distance_yaw_control = abs(robot_pose_yaw - actual_pose_yaw)
    # GUI.update()
    # if(distance_control >= 0.5 or distance_yaw_control >= math.pi/6 or first_time == 0):
    if(distance_control >= 0.5 or distance_yaw_control >= math.pi/6):
        start_time = time()
        first_time = 1
        # Etapa de observacion. Obtencion de las medidas de la observacion
        angle_real = actual_pose_yaw
        laser_rays = HAL.getLaserData()
        found = False
        distance = 8/0.03
        ang_iter = 0
        vector_teorico = [[] for _ in range(num_particles)]
        laser_teorico = [[] for _ in range(num_particles)]
        point_x = 0
        point_y = 0
        angle = 0
        num_lineas = 0
        pnt_impacto = []
        aux = True
        for z, particle in enumerate(particles):
            while(ang_iter < 5): # Debe haber 5 rayos en cada particula
                xr = math.cos(angle)*distance
                yr = math.sin(angle)*distance
                # Se podria introducir la comprobacion de si la "x" o la "y" son negativas
                px = particle[0] + math.cos(particle[2]) * xr - math.sin(particle[2]) * yr
                py = particle[1] + math.sin(particle[2]) * xr + math.cos(particle[2]) * yr
                L2 = ([(particle[1]*1 - 1*py),(1*px - particle[0]*1),(particle[0]*py - particle[1]*px)])
                for j in range (70): # Se recorre cada una de las lineas del mapa
                    dx0 = Q[j][1][0] - Q[j][0][0]
                    dx1 = px - particle[0]
                    dy0 = Q[j][1][1] - Q[j][0][1]
                    dy1 = py - particle[1]
                    p0 = dy1 * (px - Q[j][0][0]) - dx1 * (py - Q[j][0][1])
                    p1 = dy1 * (px - Q[j][1][0]) - dx1 * (py - Q[j][1][1])
                    p2 = dy0 * (Q[j][1][0] - particle[0]) - dx0 * (Q[j][1][1] - particle[1])
                    p3 = dy0 * (Q[j][1][0] - px) - dx0 * (Q[j][1][1] - py)
                    if(p0*p1 <= 0 and (p2*p3 <= 0)):
                        P = ([(L1[j][1]*L2[2] - L1[j][2]*L2[1]),(L1[j][2]*L2[0] - L1[j][0]*L2[2]),(L1[j][0]*L2[1] - L1[j][1]*L2[0])])
                        P = np.array(P)
                        P = P / P[2] # Se normaliza
                        num_lineas = num_lineas + 1
                        if(P[0] > 0 and P[1] > 0): # Solo coordenadas positivas
                            if(Q[j][0][0] != Q[j][1][0]): # Si es una linea horizontal entonces y1 == y2
                                if((min(Q[j][0][0],Q[j][1][0]) <= P[0] <= max(Q[j][0][0],Q[j][1][0]))): # El punto se encuentra dentro de la linea horizontal del mapa
                                    if(min(particle[0],px) <= P[0] <= max(particle[0],px) and min(particle[1],py) <= P[1] <= max(particle[1],py)): # El punto se encuentra dentro del segmento del laser
                                        Di = (particle[0] - P[0])**2 + (particle[1] - P[1])**2
                                        if(Di < D_min):
                                            D_min = Di
                                            pnt_impacto = np.array((round(P[0]), round(P[1]))).tolist()
                            elif(Q[j][0][0] == Q[j][1][0]): # Si es una linea vertical entonces x1 == x2
                                if((min(Q[j][0][1],Q[j][1][1]) <= P[1] <= max(Q[j][0][1],Q[j][1][1]))):
                                    if(min(particle[0],px) <= P[0] <= max(particle[0],px) and min(particle[1],py) <= P[1] <= max(particle[1],py)): # El punto se encuentra del segmento del laser
                                        Di = (particle[0] - P[0])**2 + (particle[1] - P[1])**2
                                        if(Di < D_min):
                                            D_min = Di
                                            pnt_impacto = np.array((round(P[0]), round(P[1]))).tolist()
                print(num_lineas)
                num_lineas = 0
                laser_teorico[z].append((pnt_impacto))
                if pnt_impacto:
                    manhattan = abs(round(particle[0] - pnt_impacto[0])) + abs(round(particle[0] - pnt_impacto[1]))
                    vector_teorico[z].append((manhattan))
                else:
                    pnt_impacto = np.array((round(px), round(py))).tolist()
                    manhattan = abs(round(particle[0] - pnt_impacto[0])) + abs(round(particle[0] - pnt_impacto[1]))
                    vector_teorico[z].append((manhattan))
                angle = angle + angleincrement * 80
                Di = 729**2 + 729**2 + 1
                D_min = 729**2 + 729**2
                ang_iter += 1
            ang_iter = 0
        # GUI.showEstimatedLaser((particles,laser_teorico))
        observation_time = time() - start_time
        print("El tiempo de observacion es:")
        print(observation_time)

        # Obtencion del vector real con distancias manhattan
        vector_real = []
        for i in range(0, 400, 80):
            if(laser_rays.values[i] == float("-inf") or laser_rays.values[i] == float("inf")):
                xr = math.cos(angle_real) * 8
                yr = math.cos(angle_real) * 8
            else:
                xr = math.cos(angle_real) * laser_rays.values[i]
                yr = math.cos(angle_real) * laser_rays.values[i]

            x = posx + math.cos(actual_pose_yaw) * xr - math.sin(actual_pose_yaw) * yr
            y = posy + math.sin(actual_pose_yaw) * xr + math.cos(actual_pose_yaw) * yr

            manhattan = round((abs(posx - x) + abs(posy - y))/0.03)
            vector_real.append((manhattan))
            angle_real = angle_real + angleincrement * 80 # Debe haber 5 rayos en cada particula
            while(angle_real < 0):
                angle_real = math.pi*2 + angle_real
            while(angle_real > math.pi*2):
                angle_real = angle_real - math.pi*2
        # Comparacion entre el vector teorico y el vector real
        vector_real_time = time() - (start_time + observation_time)
        print("El tiempo de generacion del vector real es:")
        print(vector_real_time)
        umbral = 8 # 4
        v = 0
        distancia = [[] for _ in range(num_particles)]
        for i, item in enumerate(vector_teorico):
            for j, dist in enumerate(item):
                dif = abs(dist - vector_real[j])
                distancia[v].append((dif * 0.03)) # Diferencia en metros
            v = v + 1
        comparition_time = time() - (start_time + observation_time + vector_real_time)
        print("El tiempo de comparacion es:")
        print(comparition_time)
        print("Vector distancia")
        print(distancia)
        # Se suma la distancia para obtener la probabilidad de cada particula
        array_prob = []
        for i, element in enumerate(distancia):
            exponente = np.sum(np.array(element))
            prob = math.e **(-exponente)
            array_prob.append((prob))
        # print(prob)
        # print(array_prob)
        normalizers = np.sum(array_prob)
        # print(normalizers)
        array_prob = array_prob / normalizers
        # print(array_prob)
        for i, probability in enumerate(array_prob):
            particles[i][3] = probability
            particles[i][4] = probability
        # El valor de la media de la lista de probabilidades
        mean = sum(array_prob)/len(array_prob)
        probability_time = time() - (start_time + observation_time + vector_real_time + comparition_time)
        print("El tiempo de probabilidad es:")
        print(probability_time)
        ############################################################################
        ############################################################################
        # Resampling
        array_resampled = []
        iteracion = 0
        greater_particles = []
        less_particles = []
        for i, particle in enumerate(particles): # De todas las particulas 
            if(particle[3]>0.3*np.max(np.array(particles)[:][:,3])): # Si la probabilidad de la particula es mayor que un determinado umbral:
                greater_particles.append((particle,i)) # Entonces se almacena en las particulas que son mayores que un determinado umbral.
            elif(particle[3]<=0.3*np.max(np.array(particles)[:][:,3])): # Si la probabilidad es menor que un determinado umbral:
                less_particles.append((particle,i)) # Se almacena en las particulas que son menores que un determinado umbral.

        while (iteracion < 0.9*np.array(greater_particles).size): # Mientras no se coja una cantidad determinada random de particulas mayores que un determinado umbral.
            particle_resampled = random.choice(greater_particles)
            array_resampled.append((particle_resampled))
            iteracion += 1
        iteracion = 0
        while (iteracion < 0.1 * np.array(less_particles).size): # Mientras no se coja una cantidad determinada random de particulas menores que un determinado umbral.
            particle_resampled = random.choice(less_particles)
            array_resampled.append((particle_resampled))
            iteracion += 1
        ############################################################################
        ############################################################################
        ############################################################################
        ############################################################################
        # Se aplica el modelo de movimiento a las particulas
        h = distance_control
        if(h != 0):
            d_x = abs(posx - robot_pose_x)
            angulo = math.acos(d_x/h)
        
            xr = math.cos(angulo)*h
            yr = math.sin(angulo)*h
        else:
            pass
        
        # particle[0][0] = particle[0][0] + math.cos(particle[0][2])*xr - math.sin(particle[0][2])*yr
        # particle[0][1] = particle[0][1] + math.sin(particle[0][2])*xr + math.cos(particle[0][2])*yr
        
        
        # Se obtiene el giro que se ha realizado desde la última iteracion
        pos_actual_yaw = angle_real # Orientacion actual del robot
        move_yaw = pos_actual_yaw - robot_pose_yaw # Incremento del movimiento de la orientacion

        # Se calcula la distancia recorrida por el robot, todo en pixeles.
        move_x = abs(round(robot_pose_x/0.03) - round(posx/0.03))
        move_y = abs(round(robot_pose_y/0.03) - round(posy/0.03))
        
        # Se aplica el movimiento calculado a las particulas
        # Dependiendo de como este orientada la particula, se aplica un movimiento u otro
        array_resampled = np.array(array_resampled).tolist()
        v_x = 0
        v_y = 0
        sum_weight_particles = 0
        resampling_time = time() - (start_time + observation_time + vector_real_time + comparition_time + probability_time)
        print("El tiempo de resampleo es:")
        print(resampling_time)
        for i, particle in enumerate(array_resampled):
            particle[0][0] = particle[0][0] + math.cos(particle[0][2])*xr - math.sin(particle[0][2])*yr
            particle[0][1] = particle[0][1] + math.sin(particle[0][2])*xr + math.cos(particle[0][2])*yr
            particle[0][2] = particle[0][2] + move_yaw
            # Si la particula no esta dentro del rango del mapa o la probabilidad es menor que un determinado valor:
            if ((particle[0][0]<x_min or particle[0][0]>x_max) or (particle[0][1]<y_min or particle[0][1]>y_max) or (particle[0][3]<0.1*(1/num_particles))):
                # Se remuestrea la particula alrededor de la particula con mayor peso
                index = np.argmax(np.array(particles)[:][:,3])
                x = np.random.normal(particles[index][0], v_x)
                y = np.random.normal(particles[index][1], v_y)
                yaw = np.random.normal(particles[index][2], var_yaw)
                while(yaw < 0):
                    yaw = math.pi*2 + yaw
                while(yaw > math.pi*2):
                    yaw = yaw - math.pi*2
                particle[0][0] = round(x)
                particle[0][1] = round(y)
                particle[0][2] = yaw
            particles[particle[1]] = particle[0]

            # Sumatorio de los pesos de las particulas que se han considerado en esa iteracion
            sum_weight_particles += particle[0][3]
        movement_time = time() - (start_time + observation_time + vector_real_time + comparition_time + probability_time + resampling_time)
        print("El tiempo de movimiento es:")
        print(movement_time)
        # Se actualizan los valores para la proxima iteracion
        robot_pose_x = posx
        robot_pose_y = posy
        robot_pose_yaw = pos_actual_yaw

        # Se estima la posicion del robot
        # GUI.showParticles(particles)
        for i in range(num_particles):
            robot_x += particles[i][0] * particles[i][3] / sum_weight_particles
            robot_y += particles[i][1] * particles[i][3] / sum_weight_particles
            robot_yaw += particles[i][2] * particles[i][3] 
        # GUI.showEstimatedPose((robot_x, robot_y, robot_yaw))
        estimation_time = time() - (start_time + observation_time + vector_real_time + comparition_time + probability_time + resampling_time + movement_time)
        print("El tiempo de estimacion es:")
        print(estimation_time)
        # Se actualiza el peso de las particulas a su estado original
        for i, particle in enumerate(particles):
            particle[3] = 1.0 / float(num_particles)
        robot_x = 0
        robot_y = 0
        robot_yaw = 0
        a = 0
        elapsed_time = time() - start_time
        print("El codigo ha tardado:")
        print(elapsed_time)