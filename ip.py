import socket

hostname = socket.gethostname()              # Get system hostname
IPAddr = socket.gethostbyname(hostname)      # Resolve hostname to local IP

print("Your Computer Name is:", hostname)
print("Your Computer IP Address is:",IPAddr)