EXE=video_parallel

INCLUDE=$(HOME)/local/boost_1_57_0/include
LIBLINK=$(HOME)/local/boost_1_57_0/lib -lboost_filesystem  -lboost_system -lboost_regex
$(EXE):$(EXE).cpp
	mpic++ -I $(INCLUDE) -L $(LIBLINK)  -o $@ $^
