killall CCR.x
rm *.x

ifort -o CCR.x -fast LIF_CombinedCR_Sept21.f90

./CCR.x 2 1 4 1000 1 &
