import random
import math
import time
import threading
import pygame
import sys
import os
import datetime
import csv

# Default values of signal times
defaultRed = 150
defaultYellow = 5
defaultGreen = 20
defaultMinimum = 10
defaultMaximum = 60

signals = []
noOfSignals = 4
simTime = 300
timeElapsed = 0

currentGreen = 0
nextGreen = (currentGreen + 1) % noOfSignals
currentYellow = 0

# Average times for vehicles to pass the intersection
carTime = 2
bikeTime = 1
rickshawTime = 2.25
busTime = 2.5
truckTime = 2.5
ambulanceTime = 1.5  # Faster crossing time for ambulances

# Count of vehicles at a traffic signal
noOfCars = 0
noOfBikes = 0
noOfBuses = 0
noOfTrucks = 0
noOfRickshaws = 0
noOfAmbulances = 0
noOfLanes = 2

# Red signal time at which vehicles are detected
detectionTime = 5

speeds = {'car': 2.25, 'bus': 1.8, 'truck': 1.8, 'rickshaw': 2, 'bike': 2.5, 'ambulance': 3.0}  # Ambulance has higher speed

# Coordinates of start
x = {'right': [0, 0, 0], 'down': [755, 727, 697], 'left': [1400, 1400, 1400], 'up': [602, 627, 657]}
y = {'right': [348, 370, 398], 'down': [0, 0, 0], 'left': [498, 466, 436], 'up': [800, 800, 800]}

#In-Memory Storage with vehicles Dictionary:
vehicles = {'right': {0: [], 1: [], 2: [], 'crossed': 0}, 'down': {0: [], 1: [], 2: [], 'crossed': 0},
            'left': {0: [], 1: [], 2: [], 'crossed': 0}, 'up': {0: [], 1: [], 2: [], 'crossed': 0}}
vehicleTypes = {0: 'car', 1: 'bus', 2: 'truck', 3: 'rickshaw', 4: 'bike', 5: 'ambulance'}
directionNumbers = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}

# Coordinates of signal image, timer, and vehicle count
signalCoods = [(530, 230), (810, 230), (810, 570), (530, 570)]
signalTimerCoods = [(530, 210), (810, 210), (810, 550), (530, 550)]
vehicleCountCoods = [(480, 210), (880, 210), (880, 550), (480, 550)]
vehicleCountTexts = ["0", "0", "0", "0"]

# Coordinates of stop lines
stopLines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
defaultStop = {'right': 580, 'down': 320, 'left': 810, 'up': 545}
stops = {'right': [580, 580, 580], 'down': [320, 320, 320], 'left': [810, 810, 810], 'up': [545, 545, 545]}

mid = {'right': {'x': 705, 'y': 445}, 'down': {'x': 695, 'y': 450}, 'left': {'x': 695, 'y': 425}, 'up': {'x': 695, 'y': 400}}
rotationAngle = 3

# Gap between vehicles
gap = 15
gap2 = 15

pygame.init()
simulation = pygame.sprite.Group()

class TrafficSignal:
    def __init__(self, red, yellow, green, minimum, maximum):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.minimum = minimum
        self.maximum = maximum
        self.signalText = "30"
        self.totalGreenTime = 0

class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction, will_turn):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = 0
        self.willTurn = will_turn
        self.turned = 0
        self.rotateAngle = 0
        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1
        path = "images/" + direction + "/" + vehicleClass + ".png"
        self.originalImage = pygame.image.load(path)
        self.currentImage = pygame.image.load(path)

        if direction == 'right':
            if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index - 1].crossed == 0:
                self.stop = vehicles[direction][lane][self.index - 1].stop - vehicles[direction][lane][self.index - 1].currentImage.get_rect().width - gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().width + gap
            x[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif direction == 'left':
            if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index - 1].crossed == 0:
                self.stop = vehicles[direction][lane][self.index - 1].stop + vehicles[direction][lane][self.index - 1].currentImage.get_rect().width + gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().width + gap
            x[direction][lane] += temp
            stops[direction][lane] += temp
        elif direction == 'down':
            if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index - 1].crossed == 0:
                self.stop = vehicles[direction][lane][self.index - 1].stop - vehicles[direction][lane][self.index - 1].currentImage.get_rect().height - gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif direction == 'up':
            if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index - 1].crossed == 0:
                self.stop = vehicles[direction][lane][self.index - 1].stop + vehicles[direction][lane][self.index - 1].currentImage.get_rect().height + gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane] += temp
            stops[direction][lane] += temp
        simulation.add(self)

    def render(self, screen):
        screen.blit(self.currentImage, (self.x, self.y))

#Vehicle Movement Synchronization in Vehicle.move():
    def move(self):
        if self.direction == 'right':
            if self.crossed == 0 and self.x + self.currentImage.get_rect().width > stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if self.willTurn == 1:
                if self.crossed == 0 or self.x + self.currentImage.get_rect().width < mid[self.direction]['x']:
                    if (self.x + self.currentImage.get_rect().width <= self.stop or (currentGreen == 0 and currentYellow == 0) or self.crossed == 1) and \
                       (self.index == 0 or self.x + self.currentImage.get_rect().width < (vehicles[self.direction][self.lane][self.index - 1].x - gap2) or \
                        vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                        self.x += self.speed
                else:
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 2
                        self.y += 1.8
                        if self.rotateAngle == 90:
                            self.turned = 1
                    else:
                        if self.index == 0 or self.y + self.currentImage.get_rect().height < (vehicles[self.direction][self.lane][self.index - 1].y - gap2) or \
                           self.x + self.currentImage.get_rect().width < (vehicles[self.direction][self.lane][self.index - 1].x - gap2):
                            self.y += self.speed
            else:
                if (self.x + self.currentImage.get_rect().width <= self.stop or self.crossed == 1 or (currentGreen == 0 and currentYellow == 0)) and \
                   (self.index == 0 or self.x + self.currentImage.get_rect().width < (vehicles[self.direction][self.lane][self.index - 1].x - gap2) or \
                    vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                    self.x += self.speed

        elif self.direction == 'down':
            if self.crossed == 0 and self.y + self.currentImage.get_rect().height > stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if self.willTurn == 1:
                if self.crossed == 0 or self.y + self.currentImage.get_rect().height < mid[self.direction]['y']:
                    if (self.y + self.currentImage.get_rect().height <= self.stop or (currentGreen == 1 and currentYellow == 0) or self.crossed == 1) and \
                       (self.index == 0 or self.y + self.currentImage.get_rect().height < (vehicles[self.direction][self.lane][self.index - 1].y - gap2) or \
                        vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                        self.y += self.speed
                else:
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 2.5
                        self.y += 2
                        if self.rotateAngle == 90:
                            self.turned = 1
                    else:
                        if self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index - 1].x + vehicles[self.direction][self.lane][self.index - 1].currentImage.get_rect().width + gap2) or \
                           self.y < (vehicles[self.direction][self.lane][self.index - 1].y - gap2):
                            self.x -= self.speed
            else:
                if (self.y + self.currentImage.get_rect().height <= self.stop or self.crossed == 1 or (currentGreen == 1 and currentYellow == 0)) and \
                   (self.index == 0 or self.y + self.currentImage.get_rect().height < (vehicles[self.direction][self.lane][self.index - 1].y - gap2) or \
                    vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                    self.y += self.speed

        elif self.direction == 'left':
            if self.crossed == 0 and self.x < stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if self.willTurn == 1:
                if self.crossed == 0 or self.x > mid[self.direction]['x']:
                    if (self.x >= self.stop or (currentGreen == 2 and currentYellow == 0) or self.crossed == 1) and \
                       (self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index - 1].x + vehicles[self.direction][self.lane][self.index - 1].currentImage.get_rect().width + gap2) or \
                        vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                        self.x -= self.speed
                else:
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 1.8
                        self.y -= 2.5
                        if self.rotateAngle == 90:
                            self.turned = 1
                    else:
                        if self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index - 1].y + vehicles[self.direction][self.lane][self.index - 1].currentImage.get_rect().height + gap2) or \
                           self.x > (vehicles[self.direction][self.lane][self.index - 1].x + gap2):
                            self.y -= self.speed
            else:
                if (self.x >= self.stop or self.crossed == 1 or (currentGreen == 2 and currentYellow == 0)) and \
                   (self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index - 1].x + vehicles[self.direction][self.lane][self.index - 1].currentImage.get_rect().width + gap2) or \
                    vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                    self.x -= self.speed

        elif self.direction == 'up':
            if self.crossed == 0 and self.y < stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if self.willTurn == 1:
                if self.crossed == 0 or self.y > mid[self.direction]['y']:
                    if (self.y >= self.stop or (currentGreen == 3 and currentYellow == 0) or self.crossed == 1) and \
                       (self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index - 1].y + vehicles[self.direction][self.lane][self.index - 1].currentImage.get_rect().height + gap2) or \
                        vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                        self.y -= self.speed
                else:
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 1
                        self.y -= 1
                        if self.rotateAngle == 90:
                            self.turned = 1
                    else:
                        if self.index == 0 or self.x < (vehicles[self.direction][self.lane][self.index - 1].x - vehicles[self.direction][self.lane][self.index - 1].currentImage.get_rect().width - gap2) or \
                           self.y > (vehicles[self.direction][self.lane][self.index - 1].y + gap2):
                            self.x += self.speed
            else:
                if (self.y >= self.stop or self.crossed == 1 or (currentGreen == 3 and currentYellow == 0)) and \
                   (self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index - 1].y + vehicles[self.direction][self.lane][self.index - 1].currentImage.get_rect().height + gap2) or \
                    vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                    self.y -= self.speed

def initialize():
    ts1 = TrafficSignal(0, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts1)
    ts2 = TrafficSignal(ts1.red + ts1.yellow + ts1.green, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts2)
    ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts3)
    ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts4)
    repeat()

def setTime():
    global noOfCars, noOfBikes, noOfBuses, noOfTrucks, noOfRickshaws, noOfAmbulances, noOfLanes
    global carTime, busTime, truckTime, rickshawTime, bikeTime, ambulanceTime
    noOfCars, noOfBikes, noOfBuses, noOfTrucks, noOfRickshaws, noOfAmbulances = 0, 0, 0, 0, 0, 0
    for j in range(len(vehicles[directionNumbers[nextGreen]][0])):
        vehicle = vehicles[directionNumbers[nextGreen]][0][j]
        if vehicle.crossed == 0:
            vclass = vehicle.vehicleClass
            if vclass == 'bike':
                noOfBikes += 1
            elif vclass == 'ambulance':
                noOfAmbulances += 1
    for i in range(1, 3):
        for j in range(len(vehicles[directionNumbers[nextGreen]][i])):
            vehicle = vehicles[directionNumbers[nextGreen]][i][j]
            if vehicle.crossed == 0:
                vclass = vehicle.vehicleClass
                if vclass == 'car':
                    noOfCars += 1
                elif vclass == 'bus':
                    noOfBuses += 1
                elif vclass == 'truck':
                    noOfTrucks += 1
                elif vclass == 'rickshaw':
                    noOfRickshaws += 1
                elif vclass == 'ambulance':
                    noOfAmbulances += 1
    greenTime = math.ceil(((noOfCars * carTime) + (noOfRickshaws * rickshawTime) + (noOfBuses * busTime) + \
                           (noOfTrucks * truckTime) + (noOfBikes * bikeTime) + (noOfAmbulances * ambulanceTime)) / (noOfLanes + 1))
    if greenTime < defaultMinimum:
        greenTime = defaultMinimum
    elif greenTime > defaultMaximum:
        greenTime = defaultMaximum
    signals[nextGreen].green = greenTime
    return noOfCars + noOfBikes + noOfBuses + noOfTrucks + noOfRickshaws + noOfAmbulances

#FCFS for Ambulances in checkAmbulances() and repeat():
def checkAmbulances():
    ambulanceQueue = []
    for i in range(noOfSignals):
        direction = directionNumbers[i]
        for lane in range(3):
            for vehicle in vehicles[direction][lane]:
                if vehicle.vehicleClass == 'ambulance' and vehicle.crossed == 0:
                    ambulanceQueue.append((i, vehicle))
    return ambulanceQueue

#Signal Synchronization in repeat() and currentGreen:
def repeat():
    global currentGreen, currentYellow, nextGreen
    while True:
        ambulanceQueue = checkAmbulances()
        if ambulanceQueue:
            for signalIndex, ambulance in ambulanceQueue:
                # Set green for ambulance's signal, red for others (FCFS for ambulances)
                currentGreen = signalIndex
                currentYellow = 0
                for i in range(noOfSignals):
                    if i == currentGreen:
                        signals[i].green = defaultGreen
                        signals[i].yellow = 0
                        signals[i].red = 0
                    else:
                        signals[i].red = defaultRed
                        signals[i].green = 0
                        signals[i].yellow = 0
                # Wait until ambulance crosses
                while ambulance.crossed == 0:
                    printStatus()
                    updateValues()
                    time.sleep(1)
                # After ambulance crosses, set yellow for its signal
                signals[currentGreen].green = 0
                signals[currentGreen].yellow = defaultYellow
                signals[currentGreen].red = 0
                currentYellow = 1
                # Yellow phase for ambulance's signal
                while signals[currentGreen].yellow > 0:
                    printStatus()
                    updateValues()
                    time.sleep(1)
                currentYellow = 0
                # After yellow, set red for ambulance's signal
                signals[currentGreen].yellow = 0
                signals[currentGreen].red = defaultRed
                vehicleCountTexts[currentGreen] = "0"
                for i in range(3):
                    stops[directionNumbers[currentGreen]][i] = defaultStop[directionNumbers[currentGreen]]
                    for vehicle in vehicles[directionNumbers[currentGreen]][i]:
                        vehicle.stop = defaultStop[directionNumbers[currentGreen]]
        else:
            # No ambulances: select signal with most vehicles (Maximum Vehicles First)
            maxVehicles = -1
            nextGreen = (currentGreen + 1) % noOfSignals  # Default to cyclic if all empty
            hasVehicles = False
            for i in range(noOfSignals):
                vehicleCount = sum(len(vehicles[directionNumbers[i]][lane]) for lane in range(3)) - vehicles[directionNumbers[i]]['crossed']
                if vehicleCount > 0 and vehicleCount > maxVehicles:
                    maxVehicles = vehicleCount
                    nextGreen = i
                    hasVehicles = True
            if not hasVehicles:
                # All lanes empty: keep current signal or cycle
                for i in range(noOfSignals):
                    signals[i].green = 0
                    signals[i].yellow = 0
                    signals[i].red = defaultRed
                printStatus()
                updateValues()
                time.sleep(1)
                continue
            currentGreen = nextGreen
            currentYellow = 0
            # Set green time based on vehicle count
            setTime()
            for i in range(noOfSignals):
                if i == currentGreen:
                    signals[i].yellow = 0
                    signals[i].red = 0
                else:
                    signals[i].red = defaultRed
                    signals[i].green = 0
                    signals[i].yellow = 0
            # Green phase
            while signals[currentGreen].green > 0:
                printStatus()
                updateValues()
                time.sleep(1)
            # Yellow phase
            currentYellow = 1
            signals[currentGreen].green = 0
            signals[currentGreen].yellow = defaultYellow
            vehicleCountTexts[currentGreen] = "0"
            for i in range(3):
                stops[directionNumbers[currentGreen]][i] = defaultStop[directionNumbers[currentGreen]]
                for vehicle in vehicles[directionNumbers[currentGreen]][i]:
                    vehicle.stop = defaultStop[directionNumbers[currentGreen]]
            while signals[currentGreen].yellow > 0:
                printStatus()
                updateValues()
                time.sleep(1)
            # After yellow, set red
            currentYellow = 0
            signals[currentGreen].yellow = 0
            signals[currentGreen].red = defaultRed

def printStatus():
    for i in range(noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                print(" GREEN TS", i + 1, "-> r:", signals[i].red, " y:", signals[i].yellow, " g:", signals[i].green)
            else:
                print("YELLOW TS", i + 1, "-> r:", signals[i].red, " y:", signals[i].yellow, " g:", signals[i].green)
        else:
            if signals[i].yellow > 0:
                print("YELLOW TS", i + 1, "-> r:", signals[i].red, " y:", signals[i].yellow, " g:", signals[i].green)
            else:
                print("   RED TS", i + 1, "-> r:", signals[i].red, " y:", signals[i].yellow, " g:", signals[i].green)
    print()

def updateValues():
    for i in range(noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                signals[i].green -= 1
                signals[i].totalGreenTime += 1
            else:
                signals[i].yellow -= 1
        else:
            if signals[i].yellow > 0:
                signals[i].yellow -= 1
            elif signals[i].red > 0:
                signals[i].red -= 1

def generateVehicles():
    while True:
        vehicle_type = random.randint(0, 5)
        if vehicle_type == 5:  # Ambulance
            lane_number = 1  # Ambulances only in lane 1
            if random.random() > 0.75:  # 25% chance for ambulance
                will_turn = 0
                direction_number = random.choice([0, 1, 2, 3])  # Random direction
                Vehicle(lane_number, vehicleTypes[vehicle_type], direction_number, directionNumbers[direction_number], will_turn)
        else:
            if vehicle_type == 4:
                lane_number = 0
            else:
                lane_number = random.randint(0, 1) + 1
            will_turn = 0
            if lane_number == 2:
                temp = random.randint(0, 4)
                if temp <= 2:
                    will_turn = 1
            temp = random.randint(0, 999)
            direction_number = 0
            a = [400, 800, 900, 1000]
            if temp < a[0]:
                direction_number = 0
            elif temp < a[1]:
                direction_number = 1
            elif temp < a[2]:
                direction_number = 2
            elif temp < a[3]:
                direction_number = 3
            Vehicle(lane_number, vehicleTypes[vehicle_type], direction_number, directionNumbers[direction_number], will_turn)
        time.sleep(0.75)


#CO1 File I/O in simulationTime()
def simulationTime():
    global timeElapsed, simTime
    while True:
        timeElapsed += 1
        time.sleep(1)
        if timeElapsed == simTime:
            totalVehicles = 0
            output_lines = ['Lane-wise Vehicle Counts']
            for i in range(noOfSignals):
                lane_output = f'Lane {i + 1}: {vehicles[directionNumbers[i]]["crossed"]}'
                output_lines.append(lane_output)
                print(lane_output)
                totalVehicles += vehicles[directionNumbers[i]]['crossed']
            total_output = f'Total vehicles passed: {totalVehicles}'
            time_output = f'Total time passed: {timeElapsed}'
            throughput_output = f'No. of vehicles passed per unit time: {(float(totalVehicles) / float(timeElapsed))}'
            output_lines.extend([total_output, time_output, throughput_output])
            print(total_output)
            print(time_output)
            print(throughput_output)

            # Save output to a text file
            output_dir = r'C:\Users\malig\OneDrive\Desktop\OS\Outputs'
            os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f'traffic_output_{timestamp}.txt')
            with open(output_file, 'w') as f:
                f.write('\n'.join(output_lines))

            os._exit(1)


#The code simulates system calls by performing file I/O (open, write) and process management (via threads)
class Main:
    thread4 = threading.Thread(name="simulationTime", target=simulationTime, args=())
    thread4.daemon = True
    thread4.start()

    thread2 = threading.Thread(name="initialization", target=initialize, args=())
    thread2.daemon = True
    thread2.start()

    black = (0, 0, 0)
    white = (255, 255, 255)

    screenWidth = 1400
    screenHeight = 800
    screenSize = (screenWidth, screenHeight)

    background = pygame.image.load('images/mod_int.png')

    screen = pygame.display.set_mode(screenSize)
    pygame.display.set_caption("SIMULATION")

    redSignal = pygame.image.load('images/signals/red.png')
    yellowSignal = pygame.image.load('images/signals/yellow.png')
    greenSignal = pygame.image.load('images/signals/green.png')
    font = pygame.font.Font(None, 30)

    thread3 = threading.Thread(name="generateVehicles", target=generateVehicles, args=())
    thread3.daemon = True
    thread3.start()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        screen.blit(background, (0, 0))
        for i in range(noOfSignals):
            if i == currentGreen:
                if currentYellow == 1:
                    if signals[i].yellow == 0:
                        signals[i].signalText = "STOP"
                    else:
                        signals[i].signalText = signals[i].yellow
                    screen.blit(yellowSignal, signalCoods[i])
                else:
                    if signals[i].green == 0:
                        signals[i].signalText = "SLOW"
                    else:
                        signals[i].signalText = signals[i].green
                    screen.blit(greenSignal, signalCoods[i])
            else:
                if signals[i].yellow > 0:
                    signals[i].signalText = signals[i].yellow
                    screen.blit(yellowSignal, signalCoods[i])
                elif signals[i].red <= 10:
                    if signals[i].red == 0:
                        signals[i].signalText = "GO"
                    else:
                        signals[i].signalText = signals[i].red
                    screen.blit(redSignal, signalCoods[i])
                else:
                    signals[i].signalText = "---"
                    screen.blit(redSignal, signalCoods[i])
        signalTexts = ["", "", "", ""]

        for i in range(noOfSignals):
            signalTexts[i] = font.render(str(signals[i].signalText), True, white, black)
            screen.blit(signalTexts[i], signalTimerCoods[i])
            displayText = vehicles[directionNumbers[i]]['crossed']
            vehicleCountTexts[i] = font.render(str(displayText), True, black, white)
            screen.blit(vehicleCountTexts[i], vehicleCountCoods[i])

        timeElapsedText = font.render(("Time Elapsed: " + str(timeElapsed)), True, black, white)
        screen.blit(timeElapsedText, (1100, 50))

        for vehicle in simulation:
            screen.blit(vehicle.currentImage, [vehicle.x, vehicle.y])
            vehicle.move()
        pygame.display.update()

def simulationTime():
    global timeElapsed, simTime
    while True:
        timeElapsed += 1
        time.sleep(1)
        if timeElapsed == simTime:
            totalVehicles = 0
            # Console output for lane-wise total counts
            print('Lane-wise Vehicle Counts')
            output_lines = ['Lane-wise Vehicle Counts']
            for i in range(noOfSignals):
                lane_total = vehicles[directionNumbers[i]]['crossed']
                lane_output = f'Lane {i + 1}: {lane_total}'
                print(lane_output)
                output_lines.append(lane_output)
                totalVehicles += lane_total
            total_output = f'Total vehicles passed: {totalVehicles}'
            time_output = f'Total time passed: {timeElapsed}'
            throughput = float(totalVehicles) / float(timeElapsed)
            throughput_output = f'No. of vehicles passed per unit time: {throughput}'
            output_lines.extend([total_output, time_output, throughput_output])
            print(total_output)
            print(time_output)
            print(throughput_output)

            # Prepare output directory
            output_dir = r'C:\Users\malig\OneDrive\Desktop\OS\Outputs'
            try:
                os.makedirs(output_dir, exist_ok=True)
                print(f"Output directory ready: {output_dir}")
            except Exception as e:
                print(f"Error creating output directory: {e}")

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save to text file and File Buffering in simulationTime():
            txt_file = os.path.join(output_dir, f'traffic_output_{timestamp}.txt')
            try:
                with open(txt_file, 'w') as f:
                    f.write('\n'.join(output_lines))
                print(f"Text file saved: {txt_file}")
            except Exception as e:
                print(f"Error saving text file: {e}")



            os._exit(1)

Main()
