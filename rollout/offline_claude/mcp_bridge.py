"""Expose only tools advertised by this episode's environment, over MCP stdio."""
import asyncio
import json
import socket
import sys

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server


def request(payload):
    with socket.socket(socket.AF_UNIX) as conn:
        conn.connect(sys.argv[1])
        conn.sendall((json.dumps(payload) + "\n").encode())
        reply = json.loads(conn.makefile("rb").readline())
    if "error" in reply:
        raise ValueError(reply["error"])
    return reply["value"]


async def main():
    server = Server("episode")

    @server.list_tools()
    async def list_tools():
        return [types.Tool(**tool) for tool in request({"op": "list"})]

    @server.call_tool()
    async def call_tool(name, arguments):
        value = request({"op": "call", "name": name, "arguments": arguments})
        return [types.TextContent(type="text", text=value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))]

    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
