# This is a makefile for Peano's PoissonWithJacobi component
# generated by the Peano Development Tools (PDT) 

# Include files
-include files.mk


# Set Paths
# ---------
# Please adopt these directories to your Peano installation. The values are 
# used by the compiler and linker calls.
PEANO_HOME   = ../p3/src
PROJECT_HOME = $(PWD)
PYCLAW_HOME = ../src/clawpack/pyclaw

# Set build mode
# -------------
BUILD_MODE=debug

# Set Dimension
# -------------
DIM=-DDim2
DIMENSIONS=dim2
#DIM=-DDim3

# Set multicore
# -------------
MULTICORE=multicore_no

# Set parallel
# -------------
PARALLEL=parallel_no

# Configure Peano
#----------------
PROJECT_CFLAGS = -DDebug -DAsserts -fPIC -shared -Wno-long-long -Isrc
PROJECT_LFLAGS = -shared
PYTHON_CFLAGS = $(shell python-config --cflags | sed "s/-Wstrict-prototypes //g")
EXECUTABLE=libpeano-claw-2d.so


# Configure System
# ----------------
# These values are used to assemble the symbols SYSTEM_CFLAGS and SYSTEM_LFLAGS.
INCLUDE_TBB=$(TBB_INC)
#INCLUDE_OMP=$(OMP_INC)
INCLUDE_OMP=-fopenmp
INCLUDE_MPI=$(MPI_INC)


LINK_TBB=$(TBB_SHLIB)
#LINK_OMP=$(OMP_SHLIB)
LINK_OMP=-fopenmp
#LINK_MPI=-lpthread -lrt -lmpich
LINK_MPI=


# Assemble Compiler Flags
# -----------------------
SYSTEM_CFLAGS = $(INCLUDE_TBB) $(INCLUDE_MPI) 
#SYSTEM_CFLAGS =  $(INCLUDE_OMP) $(INCLUDE_MPI)
SYSTEM_LFLAGS = $(LINK_TBB)    $(LINK_MPI)
#SYSTEM_LFLAGS =  $(LINK_OMP)    $(LINK_MPI)



# Settings for the GNU Compiler (Debug Mode)
# ------------------------------------------
#CC=g++
#COMPILER_CFLAGS=-O2 -pedantic -Wall -Wstrict-aliasing -fstrict-aliasing -ggdb
#COMPILER_LFLAGS=

# Settings for the GNU Compiler (Release Mode)
# --------------------------------------------
CC=g++
CFLAGS=-O3 -fstrict-aliasing -fno-rtti -fno-exceptions
LFLAGS=


# Settings for the Intel Compiler (Debug Mode)
# --------------------------------------------
#CC=icpc
#CFLAGS=-O0 -fstrict-aliasing 
#LFLAGS=


# Settings for the Intel Compiler (Release Mode)
# ----------------------------------------------
#CC=icpc
#CFLAGS=-fast -fstrict-aliasing 
#LFLAGS=


# Settings for the XLC Compiler (Debug Mode)
# --------------------------------------------
#CC=xlcc
#CFLAGS=-O0 -fstrict-aliasing -qpack_semantic=gnu 
#LFLAGS=


# Settings for the XLC Compiler (Release Mode)
# ----------------------------------------------
#CC=xlcc
#CFLAGS=-fast -fstrict-aliasing -qpack_semantic=gnu 
#LFLAGS=-fast 
#EXECUTABLE=peano-PoissonWithJacobi-release


# Setup build path
BUILD_PATH = build/$(BUILD_MODE)/$(DIMENSIONS)/$(MULTICORE)/$(PARALLEL)/$(CC)

# Object files
PEANOCLAW_OBJECTS=$(PEANOCLAW_SOURCES:%.cpp=$(BUILD_PATH)/peanoclaw/%.o)
PEANO_OBJECTS=$(PEANO_SOURCES:%.cpp=$(BUILD_PATH)/%.o)

all: header build copy

print:
	@echo $(PEANOCLAW_OBJECTS)
	@echo $(PEANO_OBJECTS)
	@echo $(BUILD_PATH)

files.mk:
	touch files.mk
	echo -n PEANOCLAW_SOURCES= > files.mk
	cd $(PROJECT_HOME)/src; find . -name '*.cpp' -printf '%p ' >> $(PROJECT_HOME)/files.mk
	echo >> files.mk
	echo -n PEANO_SOURCES= >> files.mk
	cd $(PEANO_HOME); find . -name '*.cpp' -printf '%p ' >> $(PROJECT_HOME)/files.mk
	find $(PROJECT_HOME) -name '*.cpp' -printf 'COMPILE %p TO %p OBJECT FILE\n' >> compiler-minutes.txt
	find $(PEANO_HOME) -name '*.cpp' -printf 'COMPILE %p TO %p OBJECT FILE\n' >> compiler-minutes.txt
	echo -n '\n\nLINK ' >> compiler-minutes.txt
	find $(PEANO_HOME) -name '*.cpp' -printf '%p ' >> compiler-minutes.txt
	find $(PROJECT_HOME) -name '*.cpp' -printf '%p ' >> compiler-minutes.txt



header:
	@echo $(BUILD_PATH)
	@echo  --- This is PeanoClaw based on Peano 3 ---


build: files.mk $(PEANOCLAW_OBJECTS) $(PEANO_OBJECTS)
	$(CC) $(PROJECT_LFLAGS) $(COMPILER_LFLAGS) $(shell python-config --ldflags) $(SYSTEM_LFLAGS) $(PEANOCLAW_OBJECTS) $(PEANO_OBJECTS) -o $(BUILD_PATH)/$(EXECUTABLE)
	@echo
	@echo build of PeanoClaw library successful
	@echo visit Peano\'s homepage at http://www.peano-framework.org


clean:
	rm -f $(EXECUTABLE)
	rm -f $(PEANOCLAW_OBJECTS)
	rm -f $(PEANO_OBJECTS)
	rm -f files.mk
	rm -f compiler-minutes.txt

copy:
	cp $(BUILD_PATH)/$(EXECUTABLE) $(PYCLAW_HOME)/src/peanoclaw/
	
$(BUILD_PATH)/peanoclaw/%.o: src/%.cpp
	@echo $@
	@echo DIR_NAME=$(shell dirname $@)
	@echo $(BUILD_PATH)
	mkdir -p $(shell dirname $@)
	$(CC) $(DIM) $(PYTHON_CFLAGS) $(PROJECT_CFLAGS) $(COMPILER_CFLAGS) $(SYSTEM_CFLAGS) -I$(PROJECT_HOME) -I$(PEANO_HOME) -c $< -o $@

$(BUILD_PATH)/%.o: $(PEANO_HOME)/%.cpp
	@echo $@
	@echo DIR_NAME=$(shell dirname $@)
	@echo $(BUILD_PATH)
	mkdir -p $(shell dirname $@)
	$(CC) $(DIM) $(PYTHON_CFLAGS) $(PROJECT_CFLAGS) $(COMPILER_CFLAGS) $(SYSTEM_CFLAGS) -I$(PROJECT_HOME) -I$(PEANO_HOME) -c $< -o $@
	
	
	
