#!/usr/bin/python

import re
line = " [kristofLaptop],rank:0 peanoclaw::mappings::Remesh::prepareSendToWorker(...)           in:fineGridCell:(cellDescriptionIndex:0,isInside:1,state:Leaf,level:1,evenFlags:[1,1],accessNumber:[1,-2,-1,2],responsibleRank:1,subtreeHoldsWorker:0,nodeWorkload:1,localWorkload:1,totalWorkload:1),fineGridVerticesEnumerator.toString():(domain-offset:[-1,-1],discrete-offset:[1,1],cell-size:[1,1],level:1,adj-flags:-3)[type=SingleLevelEnumerator],fineGridVerticesEnumerator.getVertexPosition(0):[0,0],coarseGridCell:(cellDescriptionIndex:-2,isInside:0,state:Root,level:0,evenFlags:[0,0],accessNumber:[0,0,0,0],responsibleRank:0,subtreeHoldsWorker:0,nodeWorkload:0,localWorkload:0,totalWorkload:0),coarseGridVerticesEnumerator.toString():(domain-offset:[-1,-1],discrete-offset:[0,0],cell-size:[3,3],level:0,adj-flags:-3)[type=SingleLevelEnumerator],fineGridPositionOfCell:[1,1],worker:1 (file:src/peanoclaw/mappings/Remesh.cpp,line:961)"

pattern =  re.compile("rank:(\d+) peanoclaw::mappings::Remesh::prepareSendToWorker\(...\)\s*in:.*level:(\d+).*fineGridVerticesEnumerator\.getVertexPosition\(0\):(.*).*worker:(\d+)")
#pattern =  re.compile("rank:(\d+) peanoclaw::mappings::Remesh::prepareSendToWorker\(...\)\s*in:.*level:(\d+).*fineGridVerticesEnumerator\.getVertexPosition\(0\):")

m = pattern.search(line)
print(m)
print(m.group(0))
print("level:", m.group(2))
