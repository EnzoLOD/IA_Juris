import os
import subprocess

def deploy_application():
    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()

    # Build the Docker image
    print("Building the Docker image...")
    subprocess.run(["docker", "build", "-t", "juris_oracle", "."], check=True)

    # Run the Docker container
    print("Running the Docker container...")
    subprocess.run(["docker", "run", "-d", "--name", "juris_oracle_container", "-p", "8000:8000", "juris_oracle"], check=True)

    print("Deployment completed successfully.")

if __name__ == "__main__":
    deploy_application()