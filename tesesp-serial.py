import socket
import time
import serial

import datetime

host = "192.168.4.1" # Set to ESP32 Access Point IP Address
port = 80
ser = serial.Serial(port='COM8',baudrate= 115200, timeout=1)

# Create a socket connection
# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    # Connect to the ESP32 server
    # s.connect((host, port))
    
while True:
    # Send two data values
    read = ser.read(1)
    print(read)
    arah = input("Enter arah : ")
    #kecepatan = input("Enter kecepatan : ")
    #message = f"{arah},{kecepatan}"
    date = datetime.datetime.now()
    print(date)

    if arah == "q" :
        break

    # s.send(arah.encode('utf-8'))
    ser.write(arah.encode("utf-8"))

    time.sleep(1)
  
# s.close()