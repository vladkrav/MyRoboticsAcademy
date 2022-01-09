from GUI import GUI
from HAL import HAL
import numpy as np
import math
from time import time
import random
from LUT import Q, L1
# Enter sequential code!

num_particles = 300
particles = []
robot_x = 0
robot_y = 0
robot_yaw = 0
x_min = 102
x_max = 748
y_min = 9
y_max = 657
sum_weight_particles = 0
angleincrement = 0.0157473180443
numero_iteraciones = 0
threshold_weight = 0
new_x = 0
new_y = 0
new_yaw = 0
new_weight = 0

# Posicion inicial del robot
robot_pose_x = HAL.getPose3d().x / 0.03
robot_pose_y =  HAL.getPose3d().y / 0.03
robot_pose_yaw = HAL.getPose3d().yaw
while(robot_pose_yaw < 0):
    robot_pose_yaw = math.pi*2 + robot_pose_yaw
while(robot_pose_yaw > math.pi*2):
    robot_pose_yaw = robot_pose_yaw - math.pi*2
num_rayos = 15
ray_multiplier = math.ceil(2*math.pi/(num_rayos * angleincrement))

# Incertidumbre de la posicion incial
var_x = 25
var_y = 25
var_yaw = math.pi
std_x = 10
std_y = 10
std_yaw = math.pi

# Etapa de generacion de las particulas
# Se generan las primeras partículas aleatoriamente alrededor del punto incial del robot
wo = 1.0 / float(num_particles)
w_persist = wo
for i in range(num_particles):
    x = np.random.normal(round(robot_pose_x), var_x)
    y = np.random.normal(round(robot_pose_y), var_y)
    yaw = np.random.normal(robot_pose_yaw, var_yaw)
    while(yaw < 0):
        yaw = math.pi*2 + yaw
    while(yaw > math.pi*2):
        yaw = yaw - math.pi*2
    particles.append((round(x), round(y), yaw, wo, w_persist))
# Mostrar por el GUI las particulas generadas.
particles = np.array(particles).tolist()
GUI.showParticles(particles)
Di = 729**2 + 729**2 + 1
D_min = 729**2 + 729**2
first_time = 0
while True:
    # Enter iterative code!
    posx = HAL.getPose3d().x / 0.03
    posy = HAL.getPose3d().y / 0.03
    actual_pose_yaw = HAL.getPose3d().yaw
    while(actual_pose_yaw < 0):
        actual_pose_yaw = math.pi*2 + actual_pose_yaw
    while(actual_pose_yaw > math.pi*2):
        actual_pose_yaw = actual_pose_yaw - math.pi*2
    distance_control = math.sqrt((posx - robot_pose_x)**2 + (posy - robot_pose_y)**2)
    distance_yaw_control = abs(robot_pose_yaw - actual_pose_yaw)
    GUI.update()
    # if(distance_control >= 0.5 or distance_yaw_control >= math.pi/6 or first_time == 0):
    if(distance_control >= 0.1/0.03 or distance_yaw_control >= math.pi/6):
        numero_iteraciones += 1 
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
            while(ang_iter < num_rayos): # Debe haber 5 rayos en cada particula
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
                num_lineas = 0
                laser_teorico[z].append((pnt_impacto))
                if pnt_impacto:
                    manhattan = abs(round(particle[0] - pnt_impacto[0])) + abs(round(particle[0] - pnt_impacto[1]))
                    vector_teorico[z].append((manhattan))
                else:
                    pnt_impacto = np.array((round(px), round(py))).tolist()
                    manhattan = abs(round(particle[0] - pnt_impacto[0])) + abs(round(particle[0] - pnt_impacto[1]))
                    vector_teorico[z].append((manhattan))
                angle = angle + angleincrement * ray_multiplier
                Di = 729**2 + 729**2 + 1
                D_min = 729**2 + 729**2
                ang_iter += 1
            ang_iter = 0
        observation_time = time() - start_time
        print("El tiempo de observacion es:")
        print(observation_time)

        # Obtencion del vector real con distancias manhattan
        vector_real = []
        for i in range(num_rayos):
            k = int(i * 400 / num_rayos) # Selecciona la posicion del array que se debe coger.
            if(laser_rays.values[k] == float("-inf") or laser_rays.values[k] == float("inf")):
                xr = math.cos(angleincrement*k) * 8 / 0.03
                yr = math.cos(angleincrement*k) * 8 / 0.03
            else:
                xr = math.cos(angleincrement*k) * laser_rays.values[k] / 0.03
                yr = math.cos(angleincrement*k) * laser_rays.values[k] / 0.03

            x = posx + math.cos(actual_pose_yaw) * xr - math.sin(actual_pose_yaw) * yr
            y = posy + math.sin(actual_pose_yaw) * xr + math.cos(actual_pose_yaw) * yr

            manhattan = round((abs(posx - x) + abs(posy - y)))
            vector_real.append((manhattan))
            angle_real = angle_real + angleincrement * ray_multiplier
            while(angle_real < 0):
                angle_real = math.pi*2 + angle_real
            while(angle_real > math.pi*2):
                angle_real = angle_real - math.pi*2
        # Comparacion entre el vector teorico y el vector real
        vector_real_time = time() - (start_time + observation_time)
        print("El tiempo de generacion del vector real es:")
        print(vector_real_time)
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
        # Se suma la distancia para obtener la probabilidad de cada particula
        array_prob = []
        for i, element in enumerate(distancia):
            exponente = np.sum(np.array(element))
            prob = math.e **(-exponente)
            array_prob.append((prob))
        normalizers = np.sum(array_prob)
        array_prob = array_prob / normalizers
        effective_sample_size = 0.0
        for i, normalized_weight in enumerate(array_prob):
            particles[i][3] = normalized_weight
            particles[i][4] = normalized_weight
            normalized_weight2 = normalized_weight * normalized_weight
            effective_sample_size += normalized_weight2
        effective_sample_size = 1.0 / effective_sample_size
        print("El valor de effective_sample_size es:", effective_sample_size)
        # El valor de la media de la lista de probabilidades
        mean = sum(array_prob)/len(array_prob)
        print("el valor medio de las particulas es:", mean)
        probability_time = time() - (start_time + observation_time + vector_real_time + comparition_time)
        print("El tiempo de probabilidad es:")
        print(probability_time)
        ############################################################################
        ############################################################################
        # # Resampling
        # array_resampled = []
        # iteracion = 0
        # greater_particles = []
        # less_particles = []
        # for i, particle in enumerate(particles): # De todas las particulas 
        #     if(particle[3] > 0.001*mean): #0.5*np.max(np.array(particles)[:][:,3])): Si la probabilidad de la particula es mayor que un determinado umbral:
        #         greater_particles.append((particle,i)) # Entonces se almacena en las particulas que son mayores que un determinado umbral.
        #     elif(particle[3] <= 0.001*mean): #0.5*np.max(np.array(particles)[:][:,3])): Si la probabilidad es menor que un determinado umbral:
        #         less_particles.append((particle,i)) # Se almacena en las particulas que son menores que un determinado umbral.
        # print("Numero de particulas greater", len(greater_particles))
        # print("Numero de particulas less:", len(less_particles))
        
        # while ((iteracion < 0.7* len(greater_particles)) and len(array_resampled) <= num_particles - 1): # Mientras no se coja una cantidad determinada random de particulas mayores que un determinado umbral.
        #     particle_resampled = random.choice(greater_particles)
        #     array_resampled.append((particle_resampled))
        #     iteracion += 1
        # iteracion = 0
        # while ((iteracion < 0.3 * len(less_particles)) and len(array_resampled) <= num_particles - 1): # Mientras no se coja una cantidad determinada random de particulas menores que un determinado umbral.
        #     particle_resampled = random.choice(less_particles)
        #     array_resampled.append((particle_resampled))
        #     iteracion += 1
        # print("Numero de particulas que mueve el algoritmo", len(array_resampled))
        ############################################################################
        ############################################################################
        # Se aplica el modelo de movimiento a las particulas
        turn_noise = 1.
        forward_noise = 1.
        h = distance_control
        if(h != 0):        
            xr = math.cos(0)*h + random.gauss(0.0, forward_noise)
            yr = math.sin(0)*h + random.gauss(0.0, forward_noise)
        else:
            xr = 0
            yr = 0
        
        # Se obtiene el giro que se ha realizado desde la última iteracion
        pos_actual_yaw = angle_real # Orientacion actual del robot
        move_yaw = pos_actual_yaw - robot_pose_yaw + random.gauss(0.0, turn_noise) % (2 * math.pi) # Incremento del movimiento de la orientacion
        
        # Se aplica el movimiento calculado a las particulas
        sum_weight_particles = 0
        resampling_time = time() - (start_time + observation_time + vector_real_time + comparition_time + probability_time)
        print("El tiempo de resampleo es:")
        print(resampling_time)
        for i, particle in enumerate(particles):
            particle[0] = round(particle[0] + math.cos(particle[2])*xr - math.sin(particle[2])*yr)
            particle[1] = round(particle[1] + math.sin(particle[2])*xr + math.cos(particle[2])*yr)
            particle[2] = particle[2] + move_yaw
            # Sumatorio de los pesos de las particulas que se han considerado en esa iteracion
            sum_weight_particles += particle[3]
        movement_time = time() - (start_time + observation_time + vector_real_time + comparition_time + probability_time + resampling_time)
        print("El tiempo de movimiento es:")
        print(movement_time)
        # Se actualizan los valores para la proxima iteracion
        robot_pose_x = posx
        robot_pose_y = posy
        robot_pose_yaw = pos_actual_yaw

        # Se estima la posicion del robot
        GUI.showParticles(particles)
        for i in range(num_particles):
            robot_x += particles[i][0] * particles[i][3] / 1 #sum_weight_particles
            robot_y += particles[i][1] * particles[i][3] / 1 #sum_weight_particles
            robot_yaw += particles[i][2] * particles[i][3]
        GUI.showEstimatedPose((robot_x, robot_y, robot_yaw))
        estimation_time = time() - (start_time + observation_time + vector_real_time + comparition_time + probability_time + resampling_time + movement_time)
        print("El tiempo de estimacion es:")
        print(estimation_time)
        if(numero_iteraciones >= 10):
            numero_iteraciones = 0
            new_particles = 0
            tmp_particles = []
            index = np.argmax(np.array(particles)[:][:,3])
            threshold_weight = 0.1*mean
            print("La particula con mayor peso es:", particles[index][3])
            new_weight = 1.0 / float(num_particles)
            while new_particles <= 0.90 * num_particles:
                indice = np.random.choice(np.arange(0,len(particles)))
                print("El indice que ha elegido ha sido:", indice)
                if(particles[indice][3] > threshold_weight):
                    new_x = np.random.normal(particles[indice][0], std_x)
                    new_y = np.random.normal(particles[indice][1], std_y)
                    new_yaw = np.random.normal(particles[indice][2], std_yaw)
                    if not ((new_x < x_min or new_x > x_max) or (new_y < y_min or new_y > y_max)):
                        tmp_particles.append((new_x, new_y, new_yaw, new_weight, new_weight))
                        new_particles += 1
            while new_particles <= num_particles:
                indice = np.random.choice(np.arange(0,len(particles)))
                print("El indice que ha elegido ha sido:", indice)
                new_x = np.random.normal(particles[indice][0], std_x)
                new_y = np.random.normal(particles[indice][1], std_y)
                new_yaw = np.random.normal(particles[indice][2], std_yaw)
                if not ((new_x < x_min or new_x > x_max) or (new_y < y_min or new_y > y_max)):
                        tmp_particles.append((new_x, new_y, new_yaw, new_weight, new_weight))
                        new_particles += 1
            print("El valor de new_particles es:", new_particles)
            for i, particle in enumerate(particles):
                particle[0] = tmp_particles[i][0]
                particle[1] = tmp_particles[i][1]
                particle[2] = tmp_particles[i][2]
                particle[3] = tmp_particles[i][3]
                particle[4] = tmp_particles[i][4]
            GUI.showParticles(particles)
        elapsed_time = time() - start_time
        print("El codigo ha tardado:")
        print(elapsed_time)
        print("")
        print("")
        robot_x = 0
        robot_y = 0
        robot_yaw = 0
        a = 0