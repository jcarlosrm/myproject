# myproject
to compile files:

g++-6 CPUGPUalpha.cpp -o CPUGPUalpha -ltbb -std=c++11 -framework OpenCL -w


#Estado actual de desarrollo:

-Estoy atacando el problema de la ineficiencia, pero me está costando un poco más de lo que pensaba. Si quito los memcopy que tenía, empiezan a aparecer segmentation fault y problemas con la función “assert” de la segunda etapa de ViVid, que antes no aparecían. Así que estoy volviendo a revisar y rehacer los 3 nodos de las etapas de ViVid en CPU, a ver si consigo encontrar los errores que haya cometido ahi.


