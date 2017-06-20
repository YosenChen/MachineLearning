CXX ?= g++

CXXFLAGS += -c -std=c++11 -Wall $(shell pkg-config --cflags opencv)
LDFLAGS += $(shell pkg-config --libs --static opencv)

all: opencv_knn_svr_dm demo

opencv_knn_svr_dm: opencv_knn_svr_dm.o; $(CXX) $< -o $@ $(LDFLAGS)
demo: demo.o; $(CXX) $< -o $@ $(LDFLAGS)

%.o: %.cpp; $(CXX) $< -o $@ $(CXXFLAGS)

clean: ; rm -f opencv_knn_svr_dm.o opencv_knn_svr_dm demo.o demo
