import subprocess
process = subprocess.Popen(['ps', 'aux'], stdout=subprocess.PIPE)
output, error = process.communicate()

for line in output.decode().split('\n'):
    if 'nrmd' in line and not 'grep' in line:
        parts = line.split()
        pid = parts[1]  # The PID is usually in the second column
        print(pid)