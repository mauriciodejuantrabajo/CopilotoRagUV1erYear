import os, sys

IN_PIPE = "/tmp/rag_in"
OUT_PIPE = "/tmp/rag_out"

if len(sys.argv) < 2:
    print("Usage: python3 agent_rag_client.py '<pregunta>'")
    sys.exit(1)

prompt = " ".join(sys.argv[1:])

# Send prompt
with open(IN_PIPE, "w") as pipe_in:
    pipe_in.write(prompt + "\n")

# Read asnwer
with open(OUT_PIPE, "r") as pipe_out:
    response = pipe_out.read().strip()

print(response)