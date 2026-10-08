"""Expose the resident local service on a specific LAN address without reloading GPU weights."""
import argparse
import asyncio


async def relay(reader, writer):
    while data := await reader.read(65536):
        writer.write(data)
        await writer.drain()
    if writer.can_write_eof():
        writer.write_eof()


async def serve(host, port, upstream_port):
    async def connection(reader, writer):
        upstream = None
        tasks = []
        try:
            upstream_reader, upstream = await asyncio.wait_for(
                asyncio.open_connection('127.0.0.1', upstream_port), timeout=5)
            tasks = [asyncio.create_task(relay(reader, upstream)),
                     asyncio.create_task(relay(upstream_reader, writer))]
            await asyncio.gather(*tasks)
        except (OSError, asyncio.TimeoutError):
            pass
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            for stream in (writer, upstream):
                if stream:
                    stream.close()
                    try:
                        await stream.wait_closed()
                    except OSError:
                        pass

    server = await asyncio.start_server(connection, host, port)
    print(f'LAN access: http://{host}:{port} -> http://127.0.0.1:{upstream_port}', flush=True)
    async with server:
        await server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--listen-host', required=True, help='A specific LAN interface address')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--upstream-port', type=int, default=7860)
    args = parser.parse_args()
    asyncio.run(serve(args.listen_host, args.port, args.upstream_port))
