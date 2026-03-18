.PHONY: all clean

SOURCES := $(wildcard *.c)
INCLUDES := $(wildcard *.h)

BINDIR := ../bin
M := $(BINDIR)/main
MS := $(BINDIR)/main-simd
MO := $(BINDIR)/main-o3

all: $(M) $(MS) $(MO)

$(M): $(SOURCES) $(INCLUDES)
	gcc -o $@ -O0 $(SOURCES)

$(MS): $(SOURCES) $(INCLUDES)
	gcc -o $@ -O0 -DSIMD $(SOURCES)

$(MO): $(SOURCES) $(INCLUDES)
	gcc -o $@ -O3 -DSIMD $(SOURCES)

clean:
	rm $(M) $(MS) $(MO)

include MakefileTasks
