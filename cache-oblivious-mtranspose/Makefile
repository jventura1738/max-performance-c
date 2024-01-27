CC = gcc
CFLAGS = -O3 -std=c99 -mavx -mavx2

OBJS = morton.o

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

compile: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) driver.c  -o transpose.x -march=native

run:
	./transpose.x 1000  4 1024

clean:
	rm -f *.x *~ *.o
