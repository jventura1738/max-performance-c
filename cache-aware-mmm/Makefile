CC = gcc
CFLAGS = -O3 -std=c99
CPP = g++ -o

all: 
	$(CC) $(CFLAGS) $(OBJS) kernel_driver.c  -o mmm.x -march=native -mfma -mavx
run:
	./mmm.x
check:
	$(CPP) check check.cpp -mavx -mavx2
	./check

clean:
	rm -f *.x *~ *.o check
