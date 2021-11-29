print("Intenta escribir el archivo gui")
f = open("/home/vlad/logs/ws_code.log", 'w')
f.write("websocket_code=ready")
f.close()
logged = True