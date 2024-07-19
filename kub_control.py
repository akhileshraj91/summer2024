from kubernetes import client, config, stream
import subprocess
import time


def rebuild_deployment(deployment_file):
    try:
        # Apply the deployment configuration
        subprocess.run(["kubectl", "apply", "-f", deployment_file], check=True)
        # Verify the deployment status
        subprocess.run(["kubectl", "rollout", "status", f"deployment/{deployment_name}"], check=True)
        print("Deployment rebuilt successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def execute_command_in_pod(namespace, pod_name, command):
    try:
        exec_command = ['/bin/sh', '-c', command]
        resp = stream.stream(v1.connect_get_namespaced_pod_exec,
                             pod_name,
                             namespace,
                             command=exec_command,
                             stderr=True, stdin=False,
                             stdout=True, tty=False)
        return resp
    except client.exceptions.ApiException as e:
        print(f"An error occurred: {e}")


deployment_file = "/home/cc/charon/workloads/consumer/deployment.yaml"
deployment_name = "consumer"  # Ensure this name matches the name in the deployment.yaml
# namespace = "consumer"  # Ensure this namespace matches the namespace in the deployment.yaml

# Load the kubeconfig file (assumes it's in the default location)
config.load_kube_config()

# Alternatively, if running inside a pod, use:
# config.load_incluster_config()

# Create an API client
v1 = client.CoreV1Api()
apps_v1 = client.AppsV1Api()

# Rebuild the deployment and wait for it to reconfigure and start running
rebuild_deployment(deployment_file)

# List all pods
print("Listing pods with their IPs:")
ret = v1.list_pod_for_all_namespaces(watch=False)
for i in ret.items:
    if "consumer" in i.metadata.name:
        print(f"{i.status.pod_ip}\t{i.metadata.namespace}\t{i.metadata.name}")
        start_time = time.time()
        result = execute_command_in_pod(f"{i.metadata.namespace}", f"{i.metadata.name}", "ones-stream-full 10000 100")
        end_time = time.time()
        print(result)
        execution_time = end_time - start_time
        
        # Append execution time to a file
        if "VALIDATION" in result:
            print(f"-------{execution_time}-------")
            with open("execution_times.log", "a") as log_file:
                log_file.write(f"{execution_time}\n")