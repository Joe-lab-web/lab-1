.PHONY: all clean

SOURCES := $(wildcard *.c)
INCLUDES := $(wildcard *.h)

BINDIR := ../bin
M := $(BINDIR)/main
MS := $(BINDIR)/main-simd
MO := $(BINDIR)/main-o3
M2 := $(BINDIR)/main-o2

all: $(M) $(MS) $(MO) $(M2)

$(M): $(SOURCES) $(INCLUDES)
	gcc -o $@ -O0 $(SOURCES)

$(MS): $(SOURCES) $(INCLUDES)
	gcc -o $@ -O0 -DSIMD $(SOURCES)

$(MO): $(SOURCES) $(INCLUDES)
	gcc -o $@ -O3 -DSIMD $(SOURCES)

$(M2): $(SOURCES) $(INCLUDES)
	gcc -o $@ -O2 -DSIMD $(SOURCES)

clean:
	rm $(M) $(MS) $(MO) $(M2)

include MakefileTasks
