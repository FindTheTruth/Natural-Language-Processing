CC=		g++
SHARED=  -shared -fPIC
OBJS=		strtokenizer.o dataset.o utils.o model.o
MAIN=		lda
TARGET=     lib$(MAIN).so 
all:	$(OBJS) $(MAIN).cpp
	# $(CC) -o $(MAIN) $(MAIN).cpp $(OBJS)
	# strip $(MAIN)
	$(CC) $(SHARED) $^ $(LIB) -o $(TARGET)

strtokenizer.o:	strtokenizer.h strtokenizer.cpp
	$(CC) $(SHARED) -c -o strtokenizer.o strtokenizer.cpp

dataset.o:	dataset.h dataset.cpp
	$(CC) $(SHARED) -c -o dataset.o dataset.cpp

utils.o:	utils.h utils.cpp
	$(CC) $(SHARED) -c -o utils.o utils.cpp

model.o:	model.h model.cpp
	$(CC)  $(SHARED) -c -o model.o model.cpp

test:
	

clean:
	rm $(OBJS) 
	rm $(MAIN)

