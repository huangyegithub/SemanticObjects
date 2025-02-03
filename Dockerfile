# syntax=docker/dockerfile:1.3-labs

# To build the container:
#     docker build -t smol .
# To run smol in the current directory:
#     docker run -it --rm -v "$PWD":/root/smol smol
FROM ubuntu:latest
RUN <<EOF
    apt-get -y update
    DEBIAN_FRONTEND=noninteractive apt-get -y install openjdk-11-jdk-headless python3 python3-pip python3-venv liblapack3
    rm -rf /var/lib/apt/lists/*
    python3 -m venv /opt/venv
    . /opt/venv/bin/activate
    pip3 install pyzmq
EOF
COPY . /usr/local/src/smol
WORKDIR /usr/local/src/smol
RUN ./gradlew --no-daemon assemble
WORKDIR /root/smol
CMD ["java", "-jar", "/usr/local/src/smol/build/libs/smol-0.2-all.jar"]
# CMD java -jar /usr/local/src/smol/build/libs/smol-0.2-all.jar -i examples/House/osphouseV2.smol -e -b examples/House/rooms.owl -p asset=https://github.com/Edkamb/SemanticObjects/Asset# -m

