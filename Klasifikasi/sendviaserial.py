import serial

ser = serial.Serial(port='COM8',baudrate= 115200, timeout=1)

while True:
    command = input("Enter command: ")
    ser.write(command.encode())
    print(ser.readline().decode('utf-8'))

