import asyncio

esp_ip = "192.168.137.216"  # Replace with the actual IP address of your ESP32
port = 8080


async def send_command(command):
    try:
        reader, writer = await asyncio.open_connection(esp_ip, port)
        # Send a newline character
        writer.write((command + '\n').encode('utf-8'))
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        print(f"Command sent: {command}")
    except Exception as e:
        print(f"Error sending command: {e}")


async def main():
    await send_command("Hello, ESP32!")

asyncio.run(main())
