import asyncio

host = "127.0.0.1"
port = 8080


async def send_command(command):
    try:
        reader, writer = await asyncio.open_connection(host, port)
        writer.write(command.encode('utf-8'))
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        print(f"Command sent: {command}")
    except Exception as e:
        print(f"Error sending command: {e}")


async def handle_client(reader, writer):
    data = await reader.read(100)
    message = data.decode('utf-8')
    addr = writer.get_extra_info('peername')
    print(f"Received {message!r} from {addr!r}")
    writer.close()
    await writer.wait_closed()


async def run_server():
    server = await asyncio.start_server(handle_client, host, port)
    addr = server.sockets[0].getsockname()
    print(f'Serving on {addr}')

    async with server:
        await server.serve_forever()


async def main():
    # Start the server
    server_task = asyncio.create_task(run_server())

    # Wait for a moment to ensure the server is fully running
    await asyncio.sleep(1)

    # Now send the command to the server
    await send_command("Hello, Server!")

    # Optionally send another command
    # await send_command("Hello again, Server!")

    # Shutdown the server after sending the command
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        print("Server has been cancelled")

# Run everything in a single event loop
asyncio.run(main())
