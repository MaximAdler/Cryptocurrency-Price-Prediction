#!/bin/bash
#cd docker && sudo docker-compose build && sudo docker-compose up -d
cd docker && sudo docker-compose up -d

L=0
sleep 5;
until timeout 1 bash -c 'sudo docker cp docker_predictor_1:/opt/predictor/*.png ./'; do
sleep 1
L=$((L+1))
if [[ $L -ge 30 ]]; then
        stop
fi
done

sudo docker-compose down
